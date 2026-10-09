"""One-Minute-style TTT-KVB support/query inference.

For every query denoising timestep, this runner rebuilds the fast state from
the observed support at the same diffusion timestep, then lets the query
perform an update-then-read scan. Each query is an independent branch from
the same support; query fast weights never leak into another query.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
from typing import Any
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import torch
from t2z.inference.standard import _parse_sample_indices, _run_autoregressive, build_infer_dataset, prepare_sample_for_rollout
from t2z.inference.ttt_model import build_training_faithful_model
from t2z.training.ttt import add_ttt_kvb_config
from wan_video_action.parsers import add_general_config, merge_yaml_and_args
from wan_video_action.utils import set_global_seed

def _mean_state_norm(norms: dict[str, float]) -> float:
    if not norms:
        return 0.0
    return sum(norms.values()) / float(len(norms))

def _update_counts(statistics: dict[str, list[dict[str, float]]]) -> dict[str, int]:
    return {layer_id: len(values) for layer_id, values in statistics.items()}

def _prepare_support_model_inputs(model, support: dict[str, Any], noise_seed: int) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    """Encode a support clip once and retain its clean latent and conditions."""
    pipe = model.pipe
    with torch.no_grad():
        inputs = model.get_pipeline_inputs(support)
        inputs = model.transfer_data_to_device(inputs, pipe.device, pipe.torch_dtype)
        for unit in pipe.units:
            inputs = pipe.unit_runner(unit, pipe, *inputs)
    inputs_shared, inputs_posi, _ = inputs
    model_inputs = {**inputs_shared, **inputs_posi}
    input_latents = model_inputs.get('input_latents')
    if not isinstance(input_latents, torch.Tensor):
        raise RuntimeError('Support preprocessing did not produce input_latents; the full observed support video is required for same-timestep TTT replay.')
    generator = torch.Generator(device=input_latents.device)
    generator.manual_seed(int(noise_seed))
    support_noise = torch.randn(input_latents.shape, dtype=input_latents.dtype, device=input_latents.device, generator=generator)
    return (model_inputs, input_latents.detach(), support_noise)

class SameTimestepSupportQueryModelFn:
    """Wrap one Wan model call with support replay and query adaptation."""

    def __init__(self, *, pipe, controller, original_model_fn, support_model_inputs: dict[str, Any], support_input_latents: torch.Tensor, support_noise: torch.Tensor) -> None:
        self.pipe = pipe
        self.controller = controller
        self.original_model_fn = original_model_fn
        self.support_model_inputs = support_model_inputs
        self.support_input_latents = support_input_latents
        self.support_noise = support_noise
        self.trace: list[dict[str, Any]] = []

    def _support_call_inputs(self, timestep: torch.Tensor) -> dict[str, Any]:
        support_inputs = dict(self.support_model_inputs)
        support_latents = self.pipe.scheduler.add_noise(self.support_input_latents, self.support_noise, timestep)
        first_frame_latents = support_inputs.get('first_frame_latents')
        if isinstance(first_frame_latents, torch.Tensor):
            support_latents = support_latents.clone()
            support_latents[:, :, 0:1] = first_frame_latents[:, :, 0:1]
        support_inputs['latents'] = support_latents
        support_inputs['noise'] = self.support_noise
        support_inputs['timestep'] = timestep
        support_inputs['use_gradient_checkpointing'] = False
        support_inputs['use_gradient_checkpointing_offload'] = False
        for model_name in self.pipe.in_iteration_models:
            support_inputs[model_name] = getattr(self.pipe, model_name)
        return support_inputs

    def __call__(self, *args, **query_inputs):
        timestep = query_inputs.get('timestep')
        if not isinstance(timestep, torch.Tensor):
            raise RuntimeError('Wan model_fn call did not provide a tensor timestep.')
        self.controller.reset(batch_size=1)
        try:
            with self.controller.causal_scan(differentiable=False):
                self.original_model_fn(**self._support_call_inputs(timestep))
            support_statistics = self.controller.write_statistics()
            support_counts = _update_counts(support_statistics)
            support_norms = self.controller.state_norms()
            with self.controller.causal_scan(differentiable=False):
                prediction = self.original_model_fn(*args, **query_inputs)
            complete_statistics = self.controller.write_statistics()
            complete_counts = _update_counts(complete_statistics)
            query_counts = {layer_id: complete_counts.get(layer_id, 0) - support_counts.get(layer_id, 0) for layer_id in complete_counts}
            query_norms = self.controller.state_norms()
            self.trace.append({'denoising_call': len(self.trace), 'timestep': float(timestep.detach().float().reshape(-1)[0].cpu()), 'support_updates_per_layer': support_counts, 'query_updates_per_layer': query_counts, 'support_state_norm_mean': _mean_state_norm(support_norms), 'query_state_norm_mean': _mean_state_norm(query_norms)})
            return prediction
        finally:
            self.controller.clear()
