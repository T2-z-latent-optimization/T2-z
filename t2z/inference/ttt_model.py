"""Training-faithful TTT-KVB support-write/query-rollout inference."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from safetensors.torch import load_file
import torch
from t2z.inference.standard import _parse_sample_indices, _run_autoregressive, build_infer_dataset, prepare_sample_for_rollout
from t2z.training.ttt import add_ttt_kvb_config
from scripts.train import WanTrainingModule
from wan_video_action.methods.baselines.ttt_kvb.wan_adapter import install_ttt_kvb
from wan_video_action.parsers import add_general_config, merge_yaml_and_args, prepare_runtime_config
from wan_video_action.utils import set_global_seed
_FROZEN_DIT_KEYS = {'text_embedding.0.weight', 'text_embedding.0.bias', 'text_embedding.2.weight', 'text_embedding.2.bias'}
_FROZEN_ACTION_KEYS = {'action_embedding.0.weight', 'action_embedding.0.bias', 'action_embedding.2.weight', 'action_embedding.2.bias'}

def load_ttt_kvb_checkpoint(model, installation, checkpoint_path: str) -> None:
    """Load the partial training checkpoint without hiding key mismatches."""
    state = load_file(checkpoint_path, device='cpu')
    action_prefix = 'pipe.action_encoder.'
    action_state = {key[len(action_prefix):]: value for key, value in state.items() if key.startswith(action_prefix)}
    unsupported_pipe_keys = sorted((key for key in state if key.startswith('pipe.') and (not key.startswith(action_prefix))))
    if unsupported_pipe_keys:
        raise RuntimeError(f'Unsupported pipe-prefixed keys in TTT-KVB checkpoint: {unsupported_pipe_keys}')
    if not action_state:
        raise RuntimeError('TTT-KVB checkpoint contains no action-encoder weights.')
    dit_state = {key: value for key, value in state.items() if not key.startswith('pipe.')}
    checkpoint_layers = sorted((int(key.split('.', 2)[1]) for key in dit_state if key.startswith('blocks.') and key.endswith('.self_attn.ttt_kvb_gate')))
    installed_layers = sorted((int(index) for index in installation.layer_indices))
    if checkpoint_layers != installed_layers:
        raise RuntimeError(f'TTT-KVB checkpoint/install layer mismatch: checkpoint={checkpoint_layers}, installed={installed_layers}.')
    for layer_index in installed_layers:
        prefix = f'blocks.{layer_index}.self_attn.ttt_kvb_memory.'
        required = {f'blocks.{layer_index}.self_attn.ttt_kvb_gate', prefix + 'q_proj.weight', prefix + 'k_proj.weight', prefix + 'v_proj.weight', prefix + 'out_proj.weight', prefix + 'lr_weight', prefix + 'lr_bias', prefix + 'initial_w1', prefix + 'initial_b1', prefix + 'initial_w2', prefix + 'initial_b2', prefix + 'fast_norm_weight', prefix + 'fast_norm_bias'}
        missing = sorted(required.difference(dit_state))
        if missing:
            raise RuntimeError(f'TTT-KVB checkpoint is incomplete for layer {layer_index}: {missing}')
    dit_incompatible = model.pipe.dit.load_state_dict(dit_state, strict=False)
    if set(dit_incompatible.missing_keys) != _FROZEN_DIT_KEYS or dit_incompatible.unexpected_keys:
        raise RuntimeError(f'Invalid TTT-KVB DiT checkpoint: missing={sorted(dit_incompatible.missing_keys)}, unexpected={sorted(dit_incompatible.unexpected_keys)}')
    action_incompatible = model.pipe.action_encoder.load_state_dict(action_state, strict=False)
    if set(action_incompatible.missing_keys) != _FROZEN_ACTION_KEYS or action_incompatible.unexpected_keys:
        raise RuntimeError(f'Invalid TTT-KVB action checkpoint: missing={sorted(action_incompatible.missing_keys)}, unexpected={sorted(action_incompatible.unexpected_keys)}')
    print(f'[checkpoint] loaded TTT-KVB dit={len(dit_state)} action={len(action_state)} layers={installed_layers}', flush=True)

def build_training_faithful_model(args):
    runtime_config = prepare_runtime_config(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = WanTrainingModule(model_paths=json.dumps(runtime_config['model_paths_list']), model_id_with_origin_paths=args.model_id_with_origin_paths, tokenizer_path=runtime_config['tokenizer_path'], enable_text=args.enable_text, trainable_models=args.trainable_models, lora_base_model=args.lora_base_model, lora_target_modules=args.lora_target_modules, lora_rank=args.lora_rank, lora_checkpoint=args.lora_checkpoint, preset_lora_path=args.preset_lora_path, preset_lora_model=args.preset_lora_model, use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False, extra_inputs=args.extra_inputs, modules=runtime_config['modules'], fp8_models=args.fp8_models, offload_models=args.offload_models, ckpt_path=args.ckpt_path, task=args.task, device=device, max_timestep_boundary=args.max_timestep_boundary, min_timestep_boundary=args.min_timestep_boundary, num_history_frames=args.num_history_frames, args=args)
    model.requires_grad_(False)
    installation = install_ttt_kvb(model.pipe.dit, layer_spec=args.ttt_layers, expansion=args.ttt_expansion, base_inner_lr=args.ttt_base_inner_lr, inner_batch_size=args.ttt_inner_batch_size, write_token_budget=args.ttt_write_token_budget, gate_init=args.ttt_gate_init, gate_vector=args.ttt_gate_vector, serial_after_attention=args.ttt_serial_after_attention)
    load_ttt_kvb_checkpoint(model, installation, args.ttt_checkpoint_path)
    model.use_gradient_checkpointing = False
    model.use_gradient_checkpointing_offload = False
    model.pipe.use_gradient_checkpointing = False
    model.pipe.use_gradient_checkpointing_offload = False
    model.eval()
    model.pipe.eval()
    return (model, installation)
