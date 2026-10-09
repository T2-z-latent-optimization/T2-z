from __future__ import annotations
import argparse
import contextlib
import csv
from collections import defaultdict
import json
import os
import random
import shutil
import time
import sys
from datetime import datetime, timedelta
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'scripts'))
import accelerate
from accelerate.utils import InitProcessGroupKwargs
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from diffsynth.diffusion.runner import initialize_deepspeed_gradient_checkpointing
from t2z.curriculum import phase_for_step as _curriculum_phase_for_step, activation_period_steps as _curriculum_activation_period_steps
from wan_video_action.parsers import merge_yaml_and_args, prepare_runtime_config
from wan_video_action.utils import set_global_seed
from scripts.train import TimedRetentionModelLogger, WanTrainingModule, wan_parser
from t2z.data import build_dataset
from t2z.sampling import sample_update as _sample_update_indices

def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows

@torch.no_grad()
def _load_grouped_context_table(model, path: str) -> bool:
    with open(path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    table = model.friction_context_table
    seen: set[int] = set()
    for record in payload.get('records', []):
        value = float(record['friction_mu'])
        distances = torch.abs(table.friction_values - value)
        index = int(torch.argmin(distances).item())
        if float(distances[index]) > 1e-05:
            raise ValueError(f'Context-table value {value} is absent from the current model table.')
        context = torch.tensor(record['context'], dtype=table.contexts.dtype, device=table.contexts.device)
        if tuple(context.shape) != tuple(table.contexts[index].shape):
            raise ValueError(f'Context shape mismatch for group {value}: checkpoint={tuple(context.shape)} model={tuple(table.contexts[index].shape)}')
        table.contexts[index].copy_(context)
        seen.add(index)
    if len(seen) != int(table.friction_values.numel()):
        raise ValueError(f'Context table {path} restored {len(seen)} groups, expected {int(table.friction_values.numel())}.')
    global_context = getattr(table, 'global_context', None)
    saved_global_context = payload.get('global_context')
    if global_context is None or saved_global_context is None:
        return False
    restored_global = torch.tensor(saved_global_context, dtype=global_context.dtype, device=global_context.device)
    if tuple(restored_global.shape) != tuple(global_context.shape):
        raise ValueError(f'Global context shape mismatch: checkpoint={tuple(restored_global.shape)} model={tuple(global_context.shape)}')
    global_context.copy_(restored_global)
    return True

@torch.no_grad()
def _load_background_context_table(model, path: str) -> bool:
    table = getattr(model, 'background_context_table', None)
    if table is None:
        return False
    with open(path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    section = payload.get('background_context_table')
    if not section:
        raise ValueError(f'Context table {path} does not contain background_context_table.')
    seen: set[int] = set()
    for record in section.get('records', []):
        value = float(record['background_index'])
        distances = torch.abs(table.friction_values - value)
        index = int(torch.argmin(distances).item())
        if float(distances[index]) > 1e-05:
            raise ValueError(f'Background-table value {value} is absent from the current model table.')
        context = torch.tensor(record['context'], dtype=table.contexts.dtype, device=table.contexts.device)
        if tuple(context.shape) != tuple(table.contexts[index].shape):
            raise ValueError(f'Background context shape mismatch for index {value}: checkpoint={tuple(context.shape)} model={tuple(table.contexts[index].shape)}')
        table.contexts[index].copy_(context)
        seen.add(index)
    if len(seen) != int(table.friction_values.numel()):
        raise ValueError(f'Background context table {path} restored {len(seen)} groups, expected {int(table.friction_values.numel())}.')
    return True

def _unique_friction_values(metadata_path: str) -> list[float]:
    values = sorted({float(row['friction_mu']) for row in _read_jsonl(metadata_path)})
    if not values:
        raise ValueError(f'No friction_mu values found in {metadata_path}.')
    return values

def _unique_background_values(metadata_path: str) -> list[float]:
    values = sorted({float(row['environment_index']) for row in _read_jsonl(metadata_path)})
    if not values:
        raise ValueError(f'No environment_index values found in {metadata_path}.')
    return values

def _as_float_tensor(value, *, device: torch.device | str) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().to(device=device, dtype=torch.float32).flatten()
    if isinstance(value, (list, tuple)):
        return torch.tensor([float(item) for item in value], device=device, dtype=torch.float32).flatten()
    return torch.tensor([float(value)], device=device, dtype=torch.float32)

class FrictionContextTable(nn.Module):

    def __init__(self, *, friction_values: list[float], context_dim: int, num_tokens: int, init_mode: str, init_value: float, init_std: float, init_min: float, init_max: float, enable_global_context: bool=False):
        super().__init__()
        if context_dim <= 0:
            raise ValueError(f'context_dim must be positive, got {context_dim}.')
        if num_tokens <= 0:
            raise ValueError(f'num_tokens must be positive, got {num_tokens}.')
        values = torch.tensor([float(value) for value in friction_values], dtype=torch.float32)
        self.register_buffer('friction_values', values, persistent=True)
        shape = (len(friction_values), int(num_tokens), int(context_dim))
        mode = str(init_mode).strip().lower()
        if mode == 'uniform':
            contexts = torch.empty(shape, dtype=torch.float32)
            contexts.uniform_(float(init_min), float(init_max))
        elif mode == 'shared_uniform':
            base_context = torch.empty((1, int(num_tokens), int(context_dim)), dtype=torch.float32)
            base_context.uniform_(float(init_min), float(init_max))
            contexts = base_context.repeat(len(friction_values), 1, 1)
        elif mode in ('shared_normal', 'shared_gaussian'):
            base_context = torch.full((1, int(num_tokens), int(context_dim)), float(init_value), dtype=torch.float32)
            if init_std > 0:
                base_context.normal_(mean=float(init_value), std=float(init_std))
            contexts = base_context.repeat(len(friction_values), 1, 1)
        elif mode in ('normal', 'gaussian'):
            contexts = torch.full(shape, float(init_value), dtype=torch.float32)
            if init_std > 0:
                contexts.normal_(mean=float(init_value), std=float(init_std))
        elif mode in ('constant', 'fixed'):
            contexts = torch.full(shape, float(init_value), dtype=torch.float32)
        elif mode in ('ordered_linear', 'linear_ordered'):
            ordered = torch.linspace(float(init_min), float(init_max), steps=len(friction_values), dtype=torch.float32)
            contexts = torch.full(shape, float(init_value), dtype=torch.float32)
            contexts[..., 0] = ordered[:, None].repeat(1, int(num_tokens))
        elif mode in ('ordered_initial_random_rest', 'curriculum_ordered_initial_random_rest'):
            contexts = torch.empty(shape, dtype=torch.float32)
            contexts.uniform_(float(init_min), float(init_max))
        else:
            raise ValueError(f'Unsupported grouped_context_init_mode={init_mode!r}.')
        self.contexts = nn.Parameter(contexts)
        if enable_global_context:
            self.global_context = nn.Parameter(contexts.detach().mean(dim=0).clone())
        else:
            self.register_parameter('global_context', None)

    def lookup(self, friction_mu, *, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        query = _as_float_tensor(friction_mu, device=self.friction_values.device)
        distances = torch.abs(query[:, None] - self.friction_values[None, :])
        indices = torch.argmin(distances, dim=1)
        context = self.contexts[indices]
        return context.to(device=device, dtype=dtype)

    @torch.no_grad()
    def clamp_(self, min_value: float | None, max_value: float | None) -> None:
        if min_value is None and max_value is None:
            return
        self.contexts.clamp_(min=min_value, max=max_value)

    def to_records(self) -> list[dict]:
        values = self.friction_values.detach().float().cpu().tolist()
        contexts = self.contexts.detach().float().cpu().tolist()
        return [{'friction_mu': float(mu), 'context': context} for (mu, context) in zip(values, contexts)]

class GroupedContextStage1Module(WanTrainingModule):

    def __init__(self, *args, grouped_args: argparse.Namespace, friction_values: list[float], background_values: list[float] | None=None, **kwargs):
        super().__init__(*args, **kwargs)
        if getattr(self.pipe, 'physical_context_encoder', None) is None:
            raise ValueError("Grouped-C stage1 requires physical_context_mode != 'none'.")
        self.last_self_correction_metrics: dict | None = None
        self.friction_context_table = FrictionContextTable(friction_values=friction_values, context_dim=int(grouped_args.physical_context_dim), num_tokens=int(grouped_args.physical_context_tokens), init_mode=str(grouped_args.grouped_context_init_mode), init_value=float(grouped_args.grouped_context_init_value), init_std=float(grouped_args.grouped_context_init_std), init_min=float(grouped_args.grouped_context_init_min), init_max=float(grouped_args.grouped_context_init_max), enable_global_context=False)
        self.background_context_table = None
        if getattr(self.pipe, 'background_context_encoder', None) is not None:
            if not background_values:
                raise ValueError('Background context is enabled, but no environment_index values were provided.')
            self.background_context_table = FrictionContextTable(friction_values=background_values, context_dim=int(grouped_args.background_context_dim), num_tokens=int(grouped_args.background_context_tokens), init_mode=str(grouped_args.grouped_context_init_mode), init_value=float(grouped_args.grouped_context_init_value), init_std=float(grouped_args.grouped_context_init_std), init_min=float(grouped_args.grouped_context_init_min), init_max=float(grouped_args.grouped_context_init_max))

    def get_pipeline_inputs(self, data):
        data = data.copy()
        direct_context = data.pop('_direct_physical_context', None)
        if direct_context is None:
            data['physical_context'] = self.friction_context_table.lookup(data['friction_mu'], dtype=self.pipe.torch_dtype, device=self.pipe.device)
        else:
            data['physical_context'] = torch.as_tensor(direct_context, dtype=self.pipe.torch_dtype, device=self.pipe.device)
        if self.background_context_table is not None:
            data['background_context'] = self.background_context_table.lookup(data['environment_index'], dtype=self.pipe.torch_dtype, device=self.pipe.device)
        return super().get_pipeline_inputs(data)

    def _prepare_pipeline_inputs(self, data):
        inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        inputs = self.apply_video_light_augmentation(inputs)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        return inputs

    def forward(self, data, inputs=None):
        if isinstance(data, list):
            from wan_video_action.grouped_tensor_batch import grouped_tensor_batch_loss
            if inputs is not None:
                raise ValueError('Tensor microbatches prepare their own pipeline inputs.')
            return grouped_tensor_batch_loss(self, data)
        return super().forward(data, inputs=inputs)

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_names = {name for (name, param) in self.named_parameters() if param.requires_grad}
        keep_names = set(trainable_names)
        keep_names.add('friction_context_table.friction_values')
        keep_names.add('background_context_table.friction_values')
        keep_names.update((name for name in state_dict if name.startswith('pipe.background_context_encoder.')))
        exported = {key: value for (key, value) in state_dict.items() if key in keep_names}
        if remove_prefix:
            prefix = str(remove_prefix)
            exported = {key[len(prefix):] if key.startswith(prefix) else key: value for (key, value) in exported.items()}
        return exported

class GroupedContextModelLogger(TimedRetentionModelLogger):

    def save_model(self, accelerator, model, file_name):
        unwrapped = accelerator.unwrap_model(model)
        has_non_context_trainable = any((param.requires_grad for (name, param) in unwrapped.named_parameters() if name not in ('friction_context_table.contexts', 'friction_context_table.global_context', 'background_context_table.contexts')))
        completion_marker = Path(self.output_path) / f'.{file_name}.complete'
        if accelerator.is_main_process:
            try:
                completion_marker.unlink()
            except FileNotFoundError:
                pass
        accelerator.wait_for_everyone()
        if has_non_context_trainable:
            super().save_model(accelerator, model, file_name)
        elif accelerator.is_main_process:
            print('[checkpoint] skipped model checkpoint during context-only phase; saving context table only', flush=True)
        if accelerator.is_main_process:
            table = getattr(unwrapped, 'friction_context_table', None)
            if table is None:
                raise RuntimeError('Grouped-context checkpoint is missing its context table.')
            path = os.path.join(self.output_path, file_name.replace('.safetensors', '.context_table.json'))
            self.save_context_table(accelerator, model, path)
            if has_non_context_trainable:
                self._prune_context_tables_without_checkpoints()
            temporary = completion_marker.with_name(f'.{completion_marker.name}.tmp-{os.getpid()}')
            with temporary.open('w', encoding='utf-8') as handle:
                handle.write(f'{file_name}\n')
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, completion_marker)
        else:
            deadline = time.monotonic() + 7200.0
            while not completion_marker.is_file():
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'Timed out waiting for rank 0 to finish checkpoint {file_name}.')
                time.sleep(2.0)
        accelerator.wait_for_everyone()

    def save_context_table(self, accelerator, model, path: str) -> None:
        if not accelerator.is_main_process:
            return
        unwrapped = accelerator.unwrap_model(model)
        table = getattr(unwrapped, 'friction_context_table', None)
        if table is None:
            return
        background_table = getattr(unwrapped, 'background_context_table', None)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {'num_groups': int(table.friction_values.numel()), 'context_shape': list(table.contexts.shape), 'records': table.to_records()}
        if getattr(table, 'global_context', None) is not None:
            payload['global_context'] = table.global_context.detach().float().cpu().tolist()
        if background_table is not None:
            payload['background_context_table'] = {'num_groups': int(background_table.friction_values.numel()), 'context_shape': list(background_table.contexts.shape), 'records': [{'background_index': float(record['friction_mu']), 'context': record['context']} for record in background_table.to_records()]}
        temporary = destination.with_name(f'.{destination.name}.tmp-{os.getpid()}')
        try:
            with temporary.open('w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, destination)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        print(f'[checkpoint] saved {destination}', flush=True)

    def _prune_context_tables_without_checkpoints(self):
        checkpoint_stems = {name[:-len('.safetensors')] for name in os.listdir(self.output_path) if name.startswith('step-') and name.endswith('.safetensors')}
        for path in Path(self.output_path).glob('step-*.context_table.json'):
            stem = path.name[:-len('.context_table.json')]
            if stem in checkpoint_stems:
                continue
            try:
                path.unlink()
                print(f'[checkpoint] pruned {path}', flush=True)
            except FileNotFoundError:
                pass

class StagedGroupedContextModelLogger(GroupedContextModelLogger):

    def __init__(self, *args, local_checkpoint_root: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_checkpoint_root = str(local_checkpoint_root)

    def save_model(self, accelerator, model, file_name):
        shared_output = Path(self.output_path)
        local_output = Path(self.local_checkpoint_root) / file_name.replace('.safetensors', '')
        if accelerator.is_main_process:
            shutil.rmtree(local_output, ignore_errors=True)
            local_output.mkdir(parents=True, exist_ok=False)
            print(f'[checkpoint] staging {file_name} on node-local {local_output}', flush=True)
        accelerator.wait_for_everyone()
        local_logger = GroupedContextModelLogger(str(local_output), remove_prefix_in_ckpt=self.remove_prefix_in_ckpt, save_minutes=0, keep_last=0, log_steps=0)
        local_logger.save_model(accelerator, model, file_name)
        context_name = file_name.replace('.safetensors', '.context_table.json')
        local_marker_name = f'.{file_name}.complete'
        shared_marker = shared_output / local_marker_name
        if accelerator.is_main_process:
            temporary = shared_output / f'.{file_name}.publish-{os.getpid()}'
            shutil.rmtree(temporary, ignore_errors=True)
            temporary.mkdir(parents=True, exist_ok=False)
            shutil.copy2(local_output / file_name, temporary / file_name)
            shutil.copy2(local_output / context_name, temporary / context_name)
            shutil.copy2(local_output / local_marker_name, temporary / local_marker_name)
            for final_path in (shared_output / file_name, shared_output / context_name, shared_marker):
                try:
                    final_path.unlink()
                except FileNotFoundError:
                    pass
            os.replace(temporary / file_name, shared_output / file_name)
            os.replace(temporary / context_name, shared_output / context_name)
            os.replace(temporary / local_marker_name, shared_marker)
            shutil.rmtree(temporary)
            shutil.rmtree(local_output, ignore_errors=True)
            self._prune_old_checkpoints()
            self._prune_context_tables_without_checkpoints()
            print(f'[checkpoint] published staged checkpoint {shared_output / file_name}', flush=True)
        else:
            deadline = time.monotonic() + 20.0 * 60.0 * 60.0
            while not shared_marker.is_file():
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'Timed out waiting for staged checkpoint {shared_marker}.')
                time.sleep(5.0)

def add_grouped_context_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('grouped_context_stage1')
    group.add_argument('--grouped_context_init_mode', type=str, default='normal')
    group.add_argument('--grouped_context_init_value', type=float, default=0.5)
    group.add_argument('--grouped_context_init_std', type=float, default=0.02)
    group.add_argument('--grouped_context_init_min', type=float, default=0.0)
    group.add_argument('--grouped_context_init_max', type=float, default=1.0)
    group.add_argument('--grouped_context_clamp_min', type=float, default=0.0)
    group.add_argument('--grouped_context_clamp_max', type=float, default=1.0)
    group.add_argument('--grouped_context_lr', type=float, default=None)
    group.add_argument('--grouped_background_context_lr', type=float, default=None)
    group.add_argument('--grouped_background_freeze_after_steps', type=int, default=0)
    group.add_argument('--grouped_context_new_context_lr', type=float, default=None)
    group.add_argument('--grouped_context_lr_schedule', type=str, default=None)
    group.add_argument('--grouped_context_model_lr_warmup_steps', type=int, default=0)
    group.add_argument('--grouped_context_context_lr_warmup_steps', type=int, default=0)
    group.add_argument('--grouped_context_model_phase_warmup_steps', type=int, default=0)
    group.add_argument('--grouped_context_reset_model_optimizer_state_on_phase_start', action='store_true', default=False)
    group.add_argument('--grouped_context_post_curriculum_lr', type=float, default=None)
    group.add_argument('--grouped_context_post_curriculum_start_step', type=int, default=0)
    group.add_argument('--grouped_context_phase_max_displacement', type=float, default=0.0)
    group.add_argument('--grouped_context_phase_max_displacement_start_step', type=int, default=0)
    group.add_argument('--grouped_context_validation_interval', type=int, default=0)
    group.add_argument('--grouped_context_validation_seed', type=int, default=20260810)
    group.add_argument('--grouped_context_protected_checkpoint_steps', type=str, default=None)
    group.add_argument('--grouped_context_alternating_interval', type=int, default=0)
    group.add_argument('--grouped_context_alternating_start', type=str, default='model')
    group.add_argument('--grouped_context_alternating_warmup_steps', type=int, default=0)
    group.add_argument('--grouped_context_weight_decay', type=float, default=0.0)
    group.add_argument('--grouped_context_structured_updates', type=int, default=0)
    group.add_argument('--grouped_context_friction_groups_per_update', type=int, default=4)
    group.add_argument('--grouped_context_actions_per_update', type=int, default=4)
    group.add_argument('--grouped_context_microbatches_per_update', type=int, default=0)
    group.add_argument('--grouped_context_tensor_batch_size', type=int, default=1)
    group.add_argument('--grouped_context_episode_key', type=str, default='episode_index')
    group.add_argument('--grouped_context_curriculum_initial_groups', type=int, default=0)
    group.add_argument('--grouped_context_curriculum_warmup_groups', type=int, default=0)
    group.add_argument('--grouped_context_curriculum_add_groups', type=int, default=0)
    group.add_argument('--grouped_context_curriculum_activation_interval', type=int, default=2)
    group.add_argument('--grouped_context_curriculum_group_order', type=str, default=None)
    group.add_argument('--grouped_context_curriculum_total_groups', type=int, default=0)
    group.add_argument('--grouped_context_curriculum_strata', type=int, default=0)
    group.add_argument('--grouped_context_curriculum_initial_model_steps', type=int, default=300)
    group.add_argument('--grouped_context_curriculum_initial_sample_mode', type=str, default='all')
    group.add_argument('--grouped_context_curriculum_initial_sample_groups', type=str, default=None)
    group.add_argument('--grouped_context_curriculum_initial_sample_rank_field', type=str, default='physical_friction_mu')
    group.add_argument('--grouped_context_curriculum_random_context_warmup', action='store_true', default=False)
    group.add_argument('--grouped_context_curriculum_shared_initial_friction', action='store_true', default=False)
    group.add_argument('--grouped_context_curriculum_new_context_steps', type=int, default=200)
    group.add_argument('--grouped_context_curriculum_all_context_steps', type=int, default=200)
    group.add_argument('--grouped_context_curriculum_model_steps', type=int, default=200)
    group.add_argument('--grouped_context_post_activation_steps', type=int, default=0)
    group.add_argument('--grouped_context_curriculum_rest_init_min', type=float, default=0.4)
    group.add_argument('--grouped_context_curriculum_rest_init_max', type=float, default=0.6)
    group.add_argument('--grouped_context_curriculum_initial_jitter', type=float, default=0.0)
    group.add_argument('--grouped_context_resume_context_table', type=str, default=None)
    group.add_argument('--grouped_context_resume_step', type=int, default=0)
    group.add_argument('--grouped_context_initialize_optimizer_state', type=str, default=None)
    group.add_argument('--grouped_context_initial_context_pool_path', type=str, default=None)
    group.add_argument('--grouped_context_stage_checkpoints_locally', action='store_true', default=False)
    group.add_argument('--grouped_context_local_checkpoint_root', type=str, default=None)
    group.add_argument('--grouped_context_model_phase_checkpoint_policy', type=str, default='all')
    group.add_argument('--frame_stride', type=int, default=1)
    return parser

def _metadata_index(metadata_path: str) -> tuple[dict[float, dict[int, list[int]]], list[dict]]:
    rows = _read_jsonl(metadata_path)
    grouped: dict[float, dict[int, list[int]]] = {}
    for (index, row) in enumerate(rows):
        mu = float(row['friction_mu'])
        action_id = int(row.get('action_id', 0))
        grouped.setdefault(mu, {}).setdefault(action_id, []).append(index)
    return (grouped, rows)
_ACTION_SAMPLING_WEIGHT_CACHE: dict[tuple[str, str, float, float, float], dict[int, dict[int, float]]] = {}








def _nested_uniform_group_order(num_groups: int, initial_groups: int, total_groups: int) -> list[int]:
    if num_groups <= 0:
        raise ValueError(f'num_groups must be positive, got {num_groups}.')
    total = min(int(total_groups) if total_groups > 0 else num_groups, num_groups)
    initial = min(max(int(initial_groups), 1), total)
    selected: list[int] = []

    def add(index: int) -> None:
        index = max(0, min(num_groups - 1, int(index)))
        if index not in selected:
            selected.append(index)
    if initial == 1:
        add((num_groups - 1) // 2)
    else:
        for i in range(initial):
            add(round(i * (num_groups - 1) / max(initial - 1, 1)))
    while len(selected) < initial:
        candidates = [index for index in range(num_groups) if index not in selected]
        best = max(candidates, key=lambda index: min((abs(index - old) for old in selected)))
        add(best)
    while len(selected) < total:
        candidates = [index for index in range(num_groups) if index not in selected]
        center = (num_groups - 1) / 2.0
        best = max(candidates, key=lambda index: (min((abs(index - old) for old in selected)), -abs(float(index) - center), -index))
        add(best)
    return selected

def _random_stratified_group_order(*, num_groups: int, initial_groups: int, add_groups: int, total_groups: int, num_strata: int, shared_initial_across_strata: bool=False) -> list[int]:
    if num_strata <= 1:
        raise ValueError(f'num_strata must exceed one, got {num_strata}.')
    if num_groups <= 0 or num_groups % num_strata != 0:
        raise ValueError(f'num_groups={num_groups} must be positive and divisible by num_strata={num_strata}.')
    total = min(int(total_groups) if total_groups > 0 else num_groups, num_groups)
    for (name, value) in (('initial_groups', initial_groups), ('add_groups', add_groups), ('total_groups', total)):
        if int(value) <= 0 or int(value) % num_strata != 0:
            raise ValueError(f'{name}={value} must be positive and divisible by {num_strata}.')
    groups_per_stratum = num_groups // num_strata
    initial_per_stratum = int(initial_groups) // num_strata
    add_per_stratum = int(add_groups) // num_strata
    total_per_stratum = total // num_strata
    if initial_per_stratum > groups_per_stratum or total_per_stratum > groups_per_stratum:
        raise ValueError('Requested more groups per stratum than the dataset contains.')
    system_rng = random.SystemRandom()
    initial_by_stratum: list[list[int]] = []
    remaining_by_stratum: list[list[int]] = []
    shared_anchor_offsets: list[int] | None = None
    if shared_initial_across_strata:
        shared_anchor_offsets = []
        for bin_index in range(initial_per_stratum):
            bin_start = bin_index * groups_per_stratum // initial_per_stratum
            bin_end = (bin_index + 1) * groups_per_stratum // initial_per_stratum
            shared_anchor_offsets.append(system_rng.choice(range(bin_start, bin_end)))
    for stratum in range(num_strata):
        block_start = stratum * groups_per_stratum
        block = list(range(block_start, block_start + groups_per_stratum))
        if shared_anchor_offsets is not None:
            anchors = [block_start + offset for offset in shared_anchor_offsets]
        else:
            anchors = []
            for bin_index in range(initial_per_stratum):
                bin_start = bin_index * groups_per_stratum // initial_per_stratum
                bin_end = (bin_index + 1) * groups_per_stratum // initial_per_stratum
                anchors.append(system_rng.choice(block[bin_start:bin_end]))
        remaining = [index for index in block if index not in set(anchors)]
        system_rng.shuffle(remaining)
        initial_by_stratum.append(anchors)
        remaining_by_stratum.append(remaining)
    order = [index for stratum_indices in initial_by_stratum for index in stratum_indices]
    selected_per_stratum = [initial_per_stratum] * num_strata
    offsets = [0] * num_strata
    while len(order) < total:
        for stratum in range(num_strata):
            remaining_needed = total_per_stratum - selected_per_stratum[stratum]
            take = min(add_per_stratum, remaining_needed)
            start = offsets[stratum]
            end = start + take
            order.extend(remaining_by_stratum[stratum][start:end])
            offsets[stratum] = end
            selected_per_stratum[stratum] += take
    if len(order) != total or len(set(order)) != total:
        raise RuntimeError(f'Invalid stratified curriculum order: length={len(order)} unique={len(set(order))}.')
    return order

def _load_or_create_curriculum_group_order(*, accelerator, output_path: str, num_groups: int, initial_groups: int, add_groups: int, total_groups: int, num_strata: int, shared_initial_across_strata: bool=False, explicit_order: str | None=None) -> list[int]:
    if explicit_order:
        order = [int(value.strip()) for value in explicit_order.split(',')]
        expected_total = min(int(total_groups) if total_groups > 0 else num_groups, num_groups)
        if len(order) != expected_total or len(set(order)) != expected_total or any(index < 0 or index >= num_groups for index in order):
            raise ValueError('Configured curriculum group order must contain each active group exactly once.')
        destination = Path(output_path) / 'curriculum_group_order.json'
        if accelerator.is_main_process and not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f'.{destination.name}.tmp-{os.getpid()}')
            with temporary.open('w', encoding='utf-8') as handle:
                json.dump({'group_order': order, 'num_groups': num_groups, 'total_groups': expected_total}, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        accelerator.wait_for_everyone()
        with destination.open('r', encoding='utf-8') as handle:
            saved = json.load(handle)
        if saved['group_order'] != order:
            raise ValueError(f'Configured curriculum order differs from {destination}.')
        return order
    if num_strata <= 1:
        return _nested_uniform_group_order(num_groups, initial_groups, total_groups)
    destination = Path(output_path) / 'curriculum_group_order.json'
    if accelerator.is_main_process:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            order = _random_stratified_group_order(num_groups=num_groups, initial_groups=initial_groups, add_groups=add_groups, total_groups=total_groups, num_strata=num_strata, shared_initial_across_strata=shared_initial_across_strata)
            payload = {'sampling': 'os_random_without_replacement', 'num_groups': int(num_groups), 'initial_groups': int(initial_groups), 'add_groups': int(add_groups), 'total_groups': int(total_groups), 'num_strata': int(num_strata), 'shared_initial_across_strata': bool(shared_initial_across_strata), 'group_order': order}
            temporary = destination.with_name(f'.{destination.name}.tmp-{os.getpid()}')
            with temporary.open('w', encoding='utf-8') as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            print(f'[curriculum] saved immutable random group order to {destination}', flush=True)
    accelerator.wait_for_everyone()
    with destination.open('r', encoding='utf-8') as handle:
        payload = json.load(handle)
    expected = {'num_groups': int(num_groups), 'initial_groups': int(initial_groups), 'add_groups': int(add_groups), 'total_groups': int(total_groups), 'num_strata': int(num_strata)}
    for (key, value) in expected.items():
        if int(payload.get(key, -1)) != value:
            raise ValueError(f'Persisted curriculum order has {key}={payload.get(key)!r}, expected {value}.')
    persisted_shared_initial = bool(payload.get('shared_initial_across_strata', False))
    if persisted_shared_initial != bool(shared_initial_across_strata):
        raise ValueError(f'Persisted curriculum order has shared_initial_across_strata={persisted_shared_initial}, expected {bool(shared_initial_across_strata)}.')
    order = [int(index) for index in payload['group_order']]
    expected_total = min(int(total_groups), int(num_groups))
    if len(order) != expected_total or len(set(order)) != expected_total or any((index < 0 or index >= num_groups for index in order)):
        raise ValueError(f'Persisted curriculum order in {destination} is invalid.')
    return order

@torch.no_grad()
def _share_initial_contexts_across_strata(table, group_order: list[int], initial_groups: int, num_strata: int) -> dict[int, int]:
    if num_strata <= 1 or initial_groups <= 0 or initial_groups % num_strata != 0:
        raise ValueError(f'Cannot share initial contexts with initial_groups={initial_groups}, strata={num_strata}.')
    groups_per_stratum = initial_groups // num_strata
    aliases: dict[int, int] = {}
    for local_index in range(groups_per_stratum):
        canonical_index = int(group_order[local_index])
        for stratum in range(1, num_strata):
            alias_index = int(group_order[stratum * groups_per_stratum + local_index])
            table.contexts[alias_index].copy_(table.contexts[canonical_index])
            aliases[alias_index] = canonical_index
    return aliases

@torch.no_grad()
def _initialize_curriculum_contexts(table, args, group_order: list[int], total_groups: int) -> None:
    mode = str(getattr(args, 'grouped_context_init_mode', '')).strip().lower()
    ordered_mode = mode in ('ordered_initial_random_rest', 'curriculum_ordered_initial_random_rest')
    initial_jitter = float(getattr(args, 'grouped_context_curriculum_initial_jitter', 0.0) or 0.0)
    if not ordered_mode and initial_jitter <= 0:
        return
    initial_groups = min(int(getattr(args, 'grouped_context_curriculum_initial_groups', 0) or 0), int(total_groups), len(group_order))
    if initial_groups <= 0:
        raise ValueError('ordered_initial_random_rest requires positive grouped_context_curriculum_initial_groups.')
    generator = torch.Generator(device=table.contexts.device)
    generator.manual_seed(int(getattr(args, 'seed', 0)) + 7919)
    if initial_jitter > 0 and (not ordered_mode):
        base_context = table.contexts[int(group_order[0])].detach().clone()
        clamp_min = getattr(args, 'grouped_context_clamp_min', None)
        clamp_max = getattr(args, 'grouped_context_clamp_max', None)
        for group_index in group_order[:initial_groups]:
            noise = torch.empty_like(base_context)
            noise.uniform_(-initial_jitter, initial_jitter, generator=generator)
            initialized = base_context + noise
            initialized.clamp_(min=clamp_min, max=clamp_max)
            table.contexts[int(group_index)].copy_(initialized)
        return
    rest_min = float(getattr(args, 'grouped_context_curriculum_rest_init_min', 0.4))
    rest_max = float(getattr(args, 'grouped_context_curriculum_rest_init_max', 0.6))
    table.contexts.uniform_(rest_min, rest_max, generator=generator)
    ordered = torch.linspace(float(getattr(args, 'grouped_context_init_min', 0.2)), float(getattr(args, 'grouped_context_init_max', 0.8)), steps=initial_groups, dtype=table.contexts.dtype, device=table.contexts.device)
    for (value, group_index) in zip(ordered, group_order[:initial_groups]):
        table.contexts[int(group_index)].fill_(float(value))

def _build_optimizer(model, args):
    context_lr = getattr(args, 'grouped_context_lr', None)
    if context_lr is None:
        context_lr = args.learning_rate
    context_params = []
    background_context_params = []
    bridge_global_params = []
    other_params = []
    for (name, param) in model.named_parameters():
        if not param.requires_grad:
            continue
        if name == 'friction_context_table.contexts':
            context_params.append(param)
        elif name == 'background_context_table.contexts':
            background_context_params.append(param)
        elif name == 'friction_context_table.global_context':
            bridge_global_params.append(param)
        else:
            other_params.append(param)
    param_groups = []
    if other_params:
        param_groups.append({'name': 'model', 'params': other_params, 'lr': args.learning_rate, 'weight_decay': args.weight_decay})
    if context_params:
        param_groups.append({'name': 'context', 'params': context_params, 'lr': float(context_lr), 'weight_decay': float(getattr(args, 'grouped_context_weight_decay', 0.0))})
    if background_context_params:
        background_context_lr = getattr(args, 'grouped_background_context_lr', None)
        if background_context_lr is None:
            background_context_lr = context_lr
        param_groups.append({'name': 'background_context', 'params': background_context_params, 'lr': float(background_context_lr), 'weight_decay': float(getattr(args, 'grouped_context_weight_decay', 0.0))})
    if bridge_global_params:
        param_groups.append({'name': 'bridge_global', 'params': bridge_global_params, 'lr': float(getattr(args, 'grouped_context_bridge_global_warmup_lr', context_lr)), 'weight_decay': 0.0})
    return torch.optim.AdamW(param_groups)

def _build_curriculum_optimizer(model, args, accelerator):
    state_path = getattr(args, 'grouped_context_initialize_optimizer_state', None)
    if not state_path:
        return _build_optimizer(model, args)
    context_parameters = [model.friction_context_table.contexts]
    if model.background_context_table is not None:
        context_parameters.append(model.background_context_table.contexts)
    original_requires_grad = [parameter.requires_grad for parameter in context_parameters]
    for parameter in context_parameters:
        parameter.requires_grad_(False)
    optimizer = _build_optimizer(model, args)
    source = Path(state_path)
    if source.is_dir():
        source = source / 'optimizer_scheduler.pt'
    payload = torch.load(source, map_location='cpu', weights_only=False)
    source_optimizer = payload['optimizer']
    source_groups = [group.get('name') for group in source_optimizer['param_groups']]
    if source_groups != ['model']:
        raise ValueError(f'Expected one model optimizer group in {source}, got {source_groups}.')
    optimizer.load_state_dict(source_optimizer)
    for (parameter, requires_grad) in zip(context_parameters, original_requires_grad):
        parameter.requires_grad_(requires_grad)
    optimizer.add_param_group({'name': 'context', 'params': [model.friction_context_table.contexts], 'lr': float(args.grouped_context_lr), 'weight_decay': float(getattr(args, 'grouped_context_weight_decay', 0.0))})
    if model.background_context_table is not None:
        optimizer.add_param_group({'name': 'background_context', 'params': [model.background_context_table.contexts], 'lr': float(getattr(args, 'grouped_background_context_lr', None) or args.grouped_context_lr), 'weight_decay': float(getattr(args, 'grouped_context_weight_decay', 0.0))})
    if accelerator.is_main_process:
        print(f'[optimizer_init] restored model AdamW state from {source}; new environment-C parameter group starts with fresh moments', flush=True)
    return optimizer

def _segment_scheduled_lr(schedule: str | None, step: int, default_lr: float) -> float:
    if not schedule:
        return float(default_lr)
    current = max(0, int(step))
    for raw_segment in str(schedule).split(','):
        segment = raw_segment.strip()
        if not segment:
            continue
        parts = [part.strip() for part in segment.split(':')]
        mode = parts[0].lower()
        if mode == 'warmup' and len(parts) == 3:
            end = int(float(parts[1]))
            target_lr = float(parts[2])
            if current <= end:
                return target_lr * (float(current) / max(float(end), 1.0))
        elif mode == 'hold' and len(parts) == 4:
            start = int(float(parts[1]))
            end = int(float(parts[2]))
            lr = float(parts[3])
            if start <= current < end:
                return lr
        elif mode == 'linear' and len(parts) == 5:
            start = int(float(parts[1]))
            end = int(float(parts[2]))
            start_lr = float(parts[3])
            end_lr = float(parts[4])
            if start <= current < end:
                alpha = (float(current) - float(start)) / max(float(end - start), 1.0)
                return start_lr + alpha * (end_lr - start_lr)
        elif mode == 'after' and len(parts) == 3:
            start = int(float(parts[1]))
            lr = float(parts[2])
            if current >= start:
                return lr
        else:
            raise ValueError(f'Invalid grouped_context_lr_schedule segment. Expected one of warmup:end:lr, hold:start:end:lr, linear:start:end:lr0:lr1, after:start:lr; got {segment!r}.')
    return float(default_lr)

def _apply_scheduled_lrs(optimizer, args, step: int):
    context_lr = getattr(args, 'grouped_context_lr', None)
    if context_lr is None:
        context_lr = args.learning_rate
    context_lr = _segment_scheduled_lr(getattr(args, 'grouped_context_lr_schedule', None), step, float(context_lr))
    model_lr = float(args.learning_rate)
    warmup_steps = int(getattr(args, 'grouped_context_model_lr_warmup_steps', 0) or 0)
    if warmup_steps > 0:
        model_lr = model_lr * min(float(step) / float(warmup_steps), 1.0)
    phase = _alternating_phase(args, step)
    if phase == 'model':
        context_lr = 0.0
    elif phase == 'context':
        model_lr = 0.0
    for group in optimizer.param_groups:
        if group.get('name') == 'context':
            group['lr'] = context_lr
        else:
            group['lr'] = model_lr
    return (model_lr, context_lr)

def _alternating_phase(args, step: int) -> str | None:
    interval = int(getattr(args, 'grouped_context_alternating_interval', 0) or 0)
    if interval <= 0:
        return None
    warmup_steps = int(getattr(args, 'grouped_context_alternating_warmup_steps', 0) or 0)
    if int(step) <= warmup_steps:
        return 'model'
    phase_index = (max(1, int(step) - warmup_steps) - 1) // interval
    start = str(getattr(args, 'grouped_context_alternating_start', 'model')).strip().lower()
    if start not in ('model', 'context'):
        raise ValueError(f"grouped_context_alternating_start must be 'model' or 'context', got {start!r}.")
    model_phase = phase_index % 2 == 0 if start == 'model' else phase_index % 2 == 1
    return 'model' if model_phase else 'context'

def _set_alternating_requires_grad(model, args, step: int) -> str | None:
    phase = _alternating_phase(args, step)
    if phase is None:
        for param in model.parameters():
            param.requires_grad_(True)
        return None
    for (name, param) in model.named_parameters():
        is_context = name.endswith('friction_context_table.contexts') or name == 'friction_context_table.contexts'
        param.requires_grad_(is_context if phase == 'context' else not is_context)
    return phase

def _set_curriculum_requires_grad(model, phase: str, freeze_background: bool=False) -> None:
    context_phase = phase in ('new_context', 'all_context')
    for (name, param) in model.named_parameters():
        is_context = name.endswith('friction_context_table.contexts') or name == 'friction_context_table.contexts'
        is_background_context = name.endswith('background_context_table.contexts') or name == 'background_context_table.contexts'
        is_background_encoder = 'background_context_encoder' in name
        if freeze_background and (is_background_context or is_background_encoder):
            param.requires_grad_(False)
        elif is_context:
            param.requires_grad_(context_phase)
        elif is_background_context:
            param.requires_grad_(phase in ('all_context',))
        else:
            param.requires_grad_(not context_phase)

def _apply_curriculum_lrs(optimizer, args, step: int, phase: str, phase_start: int | None=None, freeze_background: bool=False):
    model_lr = float(args.learning_rate)
    warmup_steps = int(getattr(args, 'grouped_context_model_lr_warmup_steps', 0) or 0)
    if warmup_steps > 0:
        model_lr = model_lr * min(float(step) / float(warmup_steps), 1.0)
    context_lr = getattr(args, 'grouped_context_lr', None)
    if context_lr is None:
        context_lr = args.learning_rate
    context_lr = float(context_lr)
    background_context_lr = getattr(args, 'grouped_background_context_lr', None)
    if background_context_lr is None:
        background_context_lr = context_lr
    background_context_lr = float(background_context_lr)
    if phase == 'new_context' and getattr(args, 'grouped_context_new_context_lr', None) is not None:
        context_lr = float(args.grouped_context_new_context_lr)
    post_context_lr = getattr(args, 'grouped_context_post_curriculum_lr', None)
    post_context_start = int(getattr(args, 'grouped_context_post_curriculum_start_step', 0) or 0)
    if phase in ('all_context',) and post_context_lr is not None and (post_context_start > 0) and (int(step) >= post_context_start):
        context_lr = float(post_context_lr)
    context_warmup_steps = int(getattr(args, 'grouped_context_context_lr_warmup_steps', 0) or 0)
    if context_warmup_steps > 0:
        context_warmup = min(float(step) / float(context_warmup_steps), 1.0)
        context_lr *= context_warmup
        background_context_lr *= context_warmup
    if phase in ('new_context', 'all_context'):
        model_lr = 0.0
    else:
        context_lr = 0.0
        phase_warmup_steps = int(getattr(args, 'grouped_context_model_phase_warmup_steps', 0) or 0)
        if phase_warmup_steps > 0 and phase_start is not None and (int(phase_start) > 1):
            local_step = max(1, int(step) - int(phase_start) + 1)
            model_lr *= min(float(local_step) / float(phase_warmup_steps), 1.0)
    if phase not in ('all_context',):
        background_context_lr = 0.0
    if freeze_background:
        background_context_lr = 0.0
    for group in optimizer.param_groups:
        if group.get('name') == 'context':
            group['lr'] = context_lr
        elif group.get('name') == 'background_context':
            group['lr'] = background_context_lr
        else:
            group['lr'] = model_lr
    return (model_lr, context_lr)

def _clear_optimizer_group_state(optimizer, group_name: str) -> int:
    base_optimizer = optimizer
    visited = set()
    while hasattr(base_optimizer, 'optimizer') and id(base_optimizer) not in visited:
        visited.add(id(base_optimizer))
        base_optimizer = base_optimizer.optimizer
    cleared = 0
    for group in base_optimizer.param_groups:
        if group.get('name') != group_name:
            continue
        for parameter in group['params']:
            if parameter in base_optimizer.state:
                del base_optimizer.state[parameter]
                cleared += 1
    return cleared

def _mask_context_grad(model, train_context_indices: list[int]) -> None:
    table = getattr(model, 'friction_context_table', None)
    if table is None or table.contexts.grad is None:
        return
    mask = torch.zeros(table.contexts.shape[0], device=table.contexts.grad.device, dtype=table.contexts.grad.dtype)
    for index in train_context_indices:
        if 0 <= int(index) < mask.numel():
            mask[int(index)] = 1
    table.contexts.grad.mul_(mask[:, None, None])

def _context_row_mask(table, train_context_indices: list[int]) -> torch.Tensor:
    mask = torch.zeros(table.contexts.shape[0], device=table.contexts.device, dtype=torch.bool)
    for index in train_context_indices:
        if 0 <= int(index) < mask.numel():
            mask[int(index)] = True
    return mask

@torch.no_grad()
def _zero_masked_context_optimizer_state(optimizer, table, train_mask: torch.Tensor) -> None:
    raw_optimizer = getattr(optimizer, 'optimizer', optimizer)
    state = getattr(raw_optimizer, 'state', {}).get(table.contexts, {})
    broadcast_mask = train_mask[:, None, None]
    for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
        value = state.get(key)
        if torch.is_tensor(value) and value.shape == table.contexts.shape:
            value.mul_(broadcast_mask.to(device=value.device, dtype=value.dtype))

@torch.no_grad()
def _project_context_displacement(table, anchor: torch.Tensor, max_displacement: float) -> torch.Tensor:
    delta = table.contexts - anchor
    flat = delta.flatten(start_dim=1)
    norms = torch.linalg.vector_norm(flat.float(), dim=1)
    scale = torch.clamp(float(max_displacement) / norms.clamp_min(1e-12), max=1.0)
    view_shape = (scale.shape[0],) + (1,) * (delta.ndim - 1)
    table.contexts.copy_(anchor + delta * scale.to(delta.dtype).view(view_shape))
    return torch.linalg.vector_norm((table.contexts - anchor).flatten(start_dim=1).float(), dim=1)

def _build_fixed_validation_indices(grouped_indices, accelerator, actions_per_group: int, seed: int, args=None) -> tuple[list[int], list]:
    validation_args = argparse.Namespace(**vars(args))
    validation_args.seed = int(seed)
    rows = _read_jsonl(args.dataset_metadata_path)
    groups = int(args.grouped_context_friction_groups_per_update)
    indices = _sample_update_indices(
        grouped_indices=grouped_indices, rows=rows, accelerator=accelerator,
        args=validation_args, update_idx=0, friction_groups=groups,
        actions_per_update=int(actions_per_group),
        microbatches_per_update=groups * int(actions_per_group),
    )
    episode_key = str(getattr(args, 'grouped_context_episode_key', 'episode_index'))
    return indices, [rows[index][episode_key] for index in indices]

@torch.no_grad()
def _run_fixed_validation(accelerator, model, dataset, sample_indices: list[int], seed: int) -> tuple[float, int]:
    python_state = random.getstate()
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = model.training
    model.eval()
    losses = []
    try:
        for (position, sample_index) in enumerate(sample_indices):
            sample_seed = int(seed) + int(sample_index) * 1009 + int(position)
            random.seed(sample_seed)
            torch.manual_seed(sample_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sample_seed)
            losses.append(model(dataset[sample_index]).detach().float())
    finally:
        random.setstate(python_state)
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
        model.train(was_training)
    local_sum = torch.stack(losses).sum() if losses else torch.zeros((), device=accelerator.device)
    local_count = torch.tensor(float(len(losses)), device=accelerator.device)
    global_sum = accelerator.reduce(local_sum, reduction='sum')
    global_count = accelerator.reduce(local_count, reduction='sum')
    return (float((global_sum / global_count.clamp_min(1.0)).cpu()), int(global_count.item()))

def _parse_step_set(raw_steps: str | None) -> set[int]:
    if not raw_steps:
        return set()
    return {int(part.strip()) for part in str(raw_steps).split(',') if part.strip()}

def _protect_paired_checkpoint(accelerator, model_logger, step: int) -> None:
    import os
    import shutil
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output_root = Path(model_logger.output_path)
        protected_root = output_root / 'protected_checkpoints' / f'step-{int(step)}'
        protected_root.mkdir(parents=True, exist_ok=True)
        sources = [output_root / f'step-{int(step)}.safetensors', output_root / f'step-{int(step)}.context_table.json']
        for source in sources:
            if not source.is_file():
                raise FileNotFoundError(f'Cannot protect missing checkpoint artifact: {source}')
            destination = protected_root / source.name
            temporary = protected_root / f'.{source.name}.tmp-{os.getpid()}'
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            try:
                os.link(source, temporary)
            except OSError:
                shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        marker = protected_root / '.complete'
        marker.write_text(f'protected_step={int(step)}\n', encoding='utf-8')
        print(f'[checkpoint] permanently protected step={step} at {protected_root}', flush=True)
    accelerator.wait_for_everyone()

def _save_wallclock_paired_checkpoint(accelerator, model, model_logger, step: int, phase: str, freeze_background: bool) -> None:
    """Save a complete model/current-C pair at a safe optimizer boundary."""
    original_requires_grad = [parameter.requires_grad for parameter in model.parameters()]
    try:
        _set_curriculum_requires_grad(model, 'model', freeze_background=freeze_background)
        model_logger.save_model(accelerator, model, f'step-{int(step)}.safetensors')
    finally:
        for (parameter, requires_grad) in zip(model.parameters(), original_requires_grad):
            parameter.requires_grad_(requires_grad)
    _log_context_table(accelerator, model, step, phase, 'wallclock_checkpoint')
    _save_phase_context_table(accelerator, model, model_logger, step, phase, 'wallclock_checkpoint')
    _protect_paired_checkpoint(accelerator, model_logger, step)

def _log_context_table(accelerator, model, step: int, phase: str | None, reason: str) -> None:
    if not accelerator.is_main_process:
        return
    table = getattr(accelerator.unwrap_model(model), 'friction_context_table', None)
    if table is None:
        return
    values = table.friction_values.detach().float().cpu().tolist()
    contexts = table.contexts.detach().float().cpu()
    print(f"[context_table] step={step} phase={phase or 'none'} reason={reason}", flush=True)
    for (mu, context) in zip(values, contexts):
        flat = context.flatten()
        if flat.numel() == 1:
            summary = f'c={float(flat[0]):.6f}'
        else:
            head = ','.join((f'{float(x):.4f}' for x in flat[:min(6, flat.numel())]))
            summary = f'mean={float(flat.mean()):.6f} std={float(flat.std(unbiased=False)):.6f} norm={float(torch.linalg.vector_norm(flat)):.6f} head=[{head}]'
        print(f'[context_table] mu={float(mu):.6f} {summary}', flush=True)
    global_context = getattr(table, 'global_context', None)
    if global_context is not None:
        flat = global_context.detach().float().cpu().flatten()
        head = ','.join((f'{float(x):.4f}' for x in flat[:min(6, flat.numel())]))
        print(f'[context_table] global mean={float(flat.mean()):.6f} std={float(flat.std(unbiased=False)):.6f} norm={float(torch.linalg.vector_norm(flat)):.6f} head=[{head}]', flush=True)
    background_table = getattr(accelerator.unwrap_model(model), 'background_context_table', None)
    if background_table is not None:
        background_values = background_table.friction_values.detach().float().cpu().tolist()
        background_contexts = background_table.contexts.detach().float().cpu()
        for (background_index, context) in zip(background_values, background_contexts):
            flat = context.flatten()
            head = ','.join((f'{float(x):.4f}' for x in flat[:min(6, flat.numel())]))
            print(f'[background_context_table] index={int(background_index)} mean={float(flat.mean()):.6f} std={float(flat.std(unbiased=False)):.6f} norm={float(torch.linalg.vector_norm(flat)):.6f} head=[{head}]', flush=True)

def _save_phase_context_table(accelerator, model, model_logger, step: int, phase: str | None, reason: str) -> None:
    if not accelerator.is_main_process:
        return
    safe_phase = str(phase or 'none').replace('/', '_')
    safe_reason = str(reason or 'phase').replace('/', '_')
    timestamp = datetime.now().strftime('%m%d-%H%M')
    path = os.path.join(model_logger.output_path, f'phase-{timestamp}-step-{int(step):06d}-{safe_phase}-{safe_reason}.context_table.json')
    model_logger.save_context_table(accelerator, model, path)


def launch_grouped_stage1(accelerator, dataset, model, model_logger, args):
    optimizer = _build_optimizer(model, args)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=1, collate_fn=lambda items: items[0], num_workers=args.dataset_num_workers)
    model.to(device=accelerator.device)
    (model, optimizer, dataloader) = accelerator.prepare(model, optimizer, dataloader)
    initialize_deepspeed_gradient_checkpointing(accelerator)
    for epoch_id in range(args.num_epochs):
        iterator = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        for data in iterator:
            with accelerator.accumulate(model):
                step = int(getattr(model_logger, 'step', 0)) + 1
                _apply_scheduled_lrs(optimizer, args, step)
                _set_alternating_requires_grad(model, args, step)
                optimizer.zero_grad(set_to_none=True)
                loss = model(data)
                accelerator.backward(loss)
                if args.max_grad_norm is not None and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                accelerator.unwrap_model(model).friction_context_table.clamp_(args.grouped_context_clamp_min, args.grouped_context_clamp_max)
                optimizer.zero_grad()
                model_logger.on_step_end(accelerator, model, args.save_steps, loss=loss)
        if args.save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, args.save_steps)

def launch_structured_grouped_stage1(accelerator, dataset, model, model_logger, args):
    (grouped_indices, metadata_rows) = _metadata_index(args.dataset_metadata_path)
    optimizer = _build_optimizer(model, args)
    model.to(device=accelerator.device)
    (model, optimizer) = accelerator.prepare(model, optimizer)
    initialize_deepspeed_gradient_checkpointing(accelerator)
    updates = int(args.grouped_context_structured_updates)
    friction_groups = int(args.grouped_context_friction_groups_per_update)
    actions_per_update = int(args.grouped_context_actions_per_update)
    microbatches_per_update = int(args.grouped_context_microbatches_per_update)
    if microbatches_per_update <= 0:
        microbatches_per_update = friction_groups * actions_per_update
    if updates <= 0:
        raise ValueError('grouped_context_structured_updates must be positive in structured mode.')
    if accelerator.is_main_process:
        context_lr = args.grouped_context_lr if args.grouped_context_lr is not None else args.learning_rate
        print(f'[structured_grouped_stage1] updates={updates} friction_groups={friction_groups} actions_per_update={actions_per_update} microbatches_per_update={microbatches_per_update} model_lr={args.learning_rate:g} context_lr={float(context_lr):g}', flush=True)
    resume_step = int(getattr(args, 'grouped_context_resume_step', 0) or 0)
    iterator = tqdm(range(resume_step, updates), disable=not accelerator.is_local_main_process)
    for update_idx in iterator:
        step = update_idx + 1
        _apply_scheduled_lrs(optimizer, args, step)
        _set_alternating_requires_grad(model, args, step)
        sample_indices = _sample_update_indices(grouped_indices=grouped_indices, rows=metadata_rows, accelerator=accelerator, args=args, update_idx=update_idx, friction_groups=friction_groups, actions_per_update=actions_per_update, microbatches_per_update=microbatches_per_update)
        optimizer.zero_grad(set_to_none=True)
        detached_losses = []
        for (micro_idx, sample_index) in enumerate(sample_indices):
            sync_context = accelerator.no_sync(model) if micro_idx < len(sample_indices) - 1 else contextlib.nullcontext()
            with sync_context:
                loss = model(dataset[sample_index])
                detached_losses.append(loss.detach())
                accelerator.backward(loss / len(sample_indices))
        if args.max_grad_norm is not None and args.max_grad_norm > 0:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        accelerator.unwrap_model(model).friction_context_table.clamp_(args.grouped_context_clamp_min, args.grouped_context_clamp_max)
        background_context_table = getattr(accelerator.unwrap_model(model), 'background_context_table', None)
        if background_context_table is not None:
            background_context_table.clamp_(args.grouped_context_clamp_min, args.grouped_context_clamp_max)
        phase = _alternating_phase(args, step)
        interval = int(getattr(args, 'grouped_context_alternating_interval', 0) or 0)
        warmup_steps = int(getattr(args, 'grouped_context_alternating_warmup_steps', 0) or 0)
        phase_end = step == warmup_steps or (step > warmup_steps and interval > 0 and ((step - warmup_steps) % interval == 0))
        if interval > 0 and phase_end:
            _log_context_table(accelerator, model, step, phase, 'phase_end')
            _save_phase_context_table(accelerator, model, model_logger, step, phase, 'phase_end')
        mean_loss = torch.stack([loss.float() for loss in detached_losses]).mean()
        model_logger.on_step_end(accelerator, model, args.save_steps, loss=mean_loss)
    model_logger.on_training_end(accelerator, model, args.save_steps)

def _assign_initial_contexts_from_pool(accelerator, table, group_order: list[int], initial_groups: int, pool_path: str, output_path: str) -> None:
    pool = torch.load(pool_path, map_location='cpu', weights_only=False).float()
    if pool.ndim != 3 or tuple(pool.shape[1:]) != tuple(table.contexts.shape[1:]):
        raise ValueError(f'Context pool shape {tuple(pool.shape)} is incompatible with table shape {tuple(table.contexts.shape)}.')
    count = min(int(initial_groups), len(group_order))
    if count > int(pool.shape[0]):
        raise ValueError(f'Cannot assign {count} initial groups from pool_size={int(pool.shape[0])}.')
    assignment_path = Path(output_path) / 'initial_context_pool_assignment.json'
    if accelerator.is_main_process and (not assignment_path.exists()):
        selected_pool_indices = random.SystemRandom().sample(range(int(pool.shape[0])), count)
        records = []
        table_values = table.friction_values.detach().float().cpu().tolist()
        for (group_index, pool_index) in zip(group_order[:count], selected_pool_indices):
            records.append({'group_index': int(group_index), 'group_value': float(table_values[int(group_index)]), 'pool_index': int(pool_index)})
        temporary = assignment_path.with_name(f'.{assignment_path.name}.tmp-{os.getpid()}')
        with temporary.open('w', encoding='utf-8') as handle:
            json.dump({'source_pool': str(pool_path), 'pool_size': int(pool.shape[0]), 'records': records}, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, assignment_path)
    accelerator.wait_for_everyone()
    with assignment_path.open('r', encoding='utf-8') as handle:
        assignment = json.load(handle)
    with torch.no_grad():
        for record in assignment['records']:
            table.contexts[int(record['group_index'])].copy_(pool[int(record['pool_index'])].to(device=table.contexts.device, dtype=table.contexts.dtype))
    if accelerator.is_main_process:
        print(f"[context_pool] assigned {len(assignment['records'])} initial groups from fixed pool {pool_path}; mapping={assignment_path}", flush=True)

def launch_curriculum_grouped_stage1(accelerator, dataset, model, model_logger, args):
    (grouped_indices, metadata_rows) = _metadata_index(args.dataset_metadata_path)
    unwrapped = model
    friction_values = unwrapped.friction_context_table.friction_values.detach().float().cpu().tolist()
    total_groups = int(args.grouped_context_curriculum_total_groups) or len(friction_values)
    group_order = _load_or_create_curriculum_group_order(accelerator=accelerator, output_path=model_logger.output_path, num_groups=len(friction_values), initial_groups=int(args.grouped_context_curriculum_initial_groups), add_groups=int(args.grouped_context_curriculum_add_groups), total_groups=total_groups, num_strata=int(getattr(args, 'grouped_context_curriculum_strata', 0) or 0), shared_initial_across_strata=bool(getattr(args, 'grouped_context_curriculum_shared_initial_friction', False)), explicit_order=getattr(args, 'grouped_context_curriculum_group_order', None))
    _initialize_curriculum_contexts(unwrapped.friction_context_table, args, group_order, total_groups)
    initial_context_pool_path = getattr(args, 'grouped_context_initial_context_pool_path', None)
    if initial_context_pool_path and (not getattr(args, 'grouped_context_resume_context_table', None)):
        _assign_initial_contexts_from_pool(accelerator, unwrapped.friction_context_table, group_order, int(args.grouped_context_curriculum_initial_groups), initial_context_pool_path, model_logger.output_path)
    shared_initial = bool(getattr(args, 'grouped_context_curriculum_shared_initial_friction', False))
    random_context_warmup = bool(getattr(args, 'grouped_context_curriculum_random_context_warmup', False))
    if shared_initial and random_context_warmup:
        raise ValueError('shared_initial_friction and random_context_warmup are mutually exclusive.')
    initial_sample_group_indices = None
    initial_sample_mode = str(getattr(args, 'grouped_context_curriculum_initial_sample_mode', 'all') or 'all').strip().lower()
    if initial_sample_mode not in ('all', 'explicit', 'stratum_median'):
        raise ValueError(f'grouped_context_curriculum_initial_sample_mode must be all, explicit, or stratum_median, got {initial_sample_mode!r}.')
    if initial_sample_mode == 'explicit':
        raw_sample_groups = str(getattr(args, 'grouped_context_curriculum_initial_sample_groups', None) or '')
        requested_group_values = [float(value.strip()) for value in raw_sample_groups.split(',') if value.strip()]
        if not requested_group_values:
            raise ValueError('explicit initial sampling requires grouped_context_curriculum_initial_sample_groups.')
        initial_group_count = min(int(args.grouped_context_curriculum_initial_groups), len(group_order))
        active_initial_indices = set(group_order[:initial_group_count])
        initial_sample_group_indices = []
        for requested_value in requested_group_values:
            distances = [abs(float(group_value) - requested_value) for group_value in friction_values]
            group_index = min(range(len(distances)), key=distances.__getitem__)
            if distances[group_index] > 1e-05:
                raise ValueError(f'Explicit initial sample group {requested_value:g} is absent from the context table.')
            if group_index not in active_initial_indices:
                raise ValueError(f'Explicit initial sample group {requested_value:g} is not among the first {initial_group_count} active curriculum groups.')
            initial_sample_group_indices.append(group_index)
        if len(set(initial_sample_group_indices)) != len(initial_sample_group_indices):
            raise ValueError('Explicit initial sample groups must be unique.')
        expected_strata = int(getattr(args, 'grouped_context_curriculum_strata', 0) or 0)
        if expected_strata > 0:
            stratify_field = str(getattr(args, 'grouped_context_stratify_field', None) or 'environment_index')
            row_by_group = {}
            for row in metadata_rows:
                row_by_group.setdefault(float(row['friction_mu']), row)
            selected_strata = {row_by_group[float(friction_values[group_index])][stratify_field] for group_index in initial_sample_group_indices}
            if len(selected_strata) != expected_strata:
                raise ValueError(f'Explicit initial sampling covers {len(selected_strata)} strata, expected {expected_strata}.')
    elif initial_sample_mode == 'stratum_median':
        initial_group_count = min(int(args.grouped_context_curriculum_initial_groups), len(group_order))
        stratify_field = str(getattr(args, 'grouped_context_stratify_field', None) or 'environment_index')
        rank_field = str(getattr(args, 'grouped_context_curriculum_initial_sample_rank_field', 'physical_friction_mu'))
        row_by_group = {}
        for row in metadata_rows:
            row_by_group.setdefault(float(row['friction_mu']), row)
        groups_by_stratum = defaultdict(list)
        for group_index in group_order[:initial_group_count]:
            group_value = float(friction_values[int(group_index)])
            row = row_by_group[group_value]
            groups_by_stratum[row[stratify_field]].append((float(row[rank_field]), int(group_index)))
        initial_sample_group_indices = []
        for stratum in sorted(groups_by_stratum, key=str):
            ranked = sorted(groups_by_stratum[stratum])
            initial_sample_group_indices.append(ranked[len(ranked) // 2][1])
        expected_strata = int(getattr(args, 'grouped_context_curriculum_strata', 0) or 0)
        if expected_strata > 0 and len(initial_sample_group_indices) != expected_strata:
            raise ValueError(f'Initial median sampling selected {len(initial_sample_group_indices)} strata, expected {expected_strata}.')
    shared_context_indices: dict[int, int] = {}
    if shared_initial:
        shared_context_indices = _share_initial_contexts_across_strata(unwrapped.friction_context_table, group_order, int(args.grouped_context_curriculum_initial_groups), int(getattr(args, 'grouped_context_curriculum_strata', 0) or 0))
    metadata_group_values = sorted((float(value) for value in grouped_indices))
    if len(metadata_group_values) != len(friction_values):
        raise ValueError(f'Metadata has {len(metadata_group_values)} groups but the context table has {len(friction_values)}.')
    shared_context_values = {metadata_group_values[alias_index]: metadata_group_values[canonical_index] for (alias_index, canonical_index) in shared_context_indices.items()}
    optimizer = _build_curriculum_optimizer(model, args, accelerator)
    model.to(device=accelerator.device)
    (model, optimizer) = accelerator.prepare(model, optimizer)
    initialize_deepspeed_gradient_checkpointing(accelerator)
    initial_steps = int(args.grouped_context_curriculum_initial_model_steps)
    assignment_model_steps = 0
    add_groups = int(args.grouped_context_curriculum_add_groups)
    groups_after_initial = max(0, min(total_groups, len(friction_values)) - int(args.grouped_context_curriculum_initial_groups))
    rounds = 1 + (groups_after_initial + max(add_groups, 1) - 1) // max(add_groups, 1)
    activation_steps = _curriculum_activation_period_steps(args)
    default_updates = initial_steps + assignment_model_steps + rounds * activation_steps
    updates = int(args.grouped_context_structured_updates) or default_updates
    actions_per_update = int(args.grouped_context_actions_per_update)
    friction_groups_per_update = int(args.grouped_context_friction_groups_per_update)
    microbatches_per_update = int(args.grouped_context_microbatches_per_update)
    if microbatches_per_update <= 0:
        microbatches_per_update = friction_groups_per_update * actions_per_update
    validation_interval = int(getattr(args, 'grouped_context_validation_interval', 0) or 0)
    validation_seed = int(getattr(args, 'grouped_context_validation_seed', 20260810))
    validation_indices: list[int] = []
    validation_actions: list = []
    if validation_interval > 0:
        (validation_indices, validation_actions) = _build_fixed_validation_indices(grouped_indices, accelerator, actions_per_update, validation_seed, args=args)
    protected_checkpoint_steps = _parse_step_set(getattr(args, 'grouped_context_protected_checkpoint_steps', None))
    wallclock_request_file_raw = os.environ.get('BWM_WALLCLOCK_CHECKPOINT_REQUEST_FILE', '').strip()
    wallclock_request_file = Path(wallclock_request_file_raw) if wallclock_request_file_raw else None
    wallclock_checkpoint_done = False
    wallclock_deadline = float(os.environ.get('BWM_WALLCLOCK_CHECKPOINT_AT', '0') or 0)
    wallclock_armed = wallclock_deadline > 0 or wallclock_request_file is not None
    wallclock_flag = torch.zeros((), dtype=torch.int32, device=accelerator.device)
    if accelerator.is_main_process and wallclock_deadline > 0:
        print(f'[wallclock_checkpoint] allocation deadline epoch={wallclock_deadline:.0f}; save at the first safe optimizer boundary', flush=True)
    if accelerator.is_main_process and wallclock_request_file is not None:
        print(f'[wallclock_checkpoint] armed request_file={wallclock_request_file} trigger=slurm_time_limit_minus_00:30:00', flush=True)
    if accelerator.is_main_process:
        print(f"[curriculum_grouped_stage1] updates={updates} initial_groups={args.grouped_context_curriculum_initial_groups} add_groups={add_groups} total_groups={total_groups} rounds={rounds} actions_per_update={actions_per_update} microbatches_per_update={microbatches_per_update}", flush=True)
        selected = [friction_values[index] for index in group_order[:min(total_groups, len(group_order))]]
        print('[curriculum_grouped_stage1] group_order_mu=' + ','.join((f'{float(value):.6g}' for value in selected)), flush=True)
        if validation_interval > 0:
            print(f"[validation_setup] interval={validation_interval} seed={validation_seed} actions={','.join((str(value) for value in validation_actions))}", flush=True)
        if initial_sample_group_indices is not None:
            initial_sample_values = [friction_values[index] for index in initial_sample_group_indices]
            print('[curriculum_grouped_stage1] initial_model_sample_mu=' + ','.join((f'{float(value):.6g}' for value in initial_sample_values)), flush=True)
    resume_step = int(getattr(args, 'grouped_context_resume_step', 0) or 0)
    iterator = tqdm(range(resume_step, updates), disable=not accelerator.is_local_main_process)
    last_phase_key = None
    last_forced_model_step = None
    context_phase_anchor = None
    context_phase_limit = float(getattr(args, 'grouped_context_phase_max_displacement', 0.0) or 0.0)
    context_phase_limit_start = int(getattr(args, 'grouped_context_phase_max_displacement_start_step', 0) or 0)
    for update_idx in iterator:
        step = update_idx + 1
        phase_info = _curriculum_phase_for_step(args, group_order, step)
        phase = str(phase_info['phase'])
        phase_sample_group_indices = phase_info['sample_group_indices']
        phase_local_step = step - int(phase_info['phase_start']) + 1
        if initial_sample_group_indices is not None and step <= initial_steps:
            phase_sample_group_indices = initial_sample_group_indices
        phase_key = (phase_info['round'], phase, phase_info['phase_start'], phase_info['phase_end'])
        phase_changed = phase_key != last_phase_key
        if phase_changed:
            context_phase = phase in ('new_context', 'all_context')
            if context_phase and context_phase_limit > 0.0 and (int(phase_info['phase_start']) >= context_phase_limit_start):
                context_phase_anchor = accelerator.unwrap_model(model).friction_context_table.contexts.detach().clone()
            else:
                context_phase_anchor = None
        if accelerator.is_main_process and phase_changed:
            sample_mu = [friction_values[index] for index in phase_sample_group_indices]
            print(f"[curriculum_phase] step={step} round={phase_info['round']} phase={phase} range={phase_info['phase_start']}-{phase_info['phase_end']} active_count={phase_info['active_count']} new_count={phase_info['new_count']} sample_mu={','.join((f'{float(value):.6g}' for value in sample_mu))}", flush=True)
        if phase_changed and phase == 'model' and bool(getattr(args, 'grouped_context_reset_model_optimizer_state_on_phase_start', False)):
            cleared_states = _clear_optimizer_group_state(optimizer, 'model')
            if accelerator.is_main_process:
                print(f'[optimizer_reset] step={step} group=model cleared_parameter_states={cleared_states}', flush=True)
        last_phase_key = phase_key
        background_freeze_after = int(getattr(args, 'grouped_background_freeze_after_steps', 0) or 0)
        freeze_background = background_freeze_after > 0 and step > background_freeze_after
        if accelerator.is_main_process and background_freeze_after > 0 and (step == background_freeze_after + 1):
            print(f'[background_context] permanently froze Background Z and background_context_encoder after step={background_freeze_after}', flush=True)
        _set_curriculum_requires_grad(model, phase, freeze_background=freeze_background)
        (current_model_lr, current_context_lr) = _apply_curriculum_lrs(optimizer, args, step, phase, phase_start=int(phase_info['phase_start']), freeze_background=freeze_background)
        if accelerator.is_main_process and phase_changed:
            print(f'[phase_lr] step={step} model_lr={current_model_lr:.8g} context_lr={current_context_lr:.8g}', flush=True)
        sample_values = [friction_values[index] for index in phase_sample_group_indices]
        sample_friction_groups = friction_groups_per_update
        sample_indices = _sample_update_indices(grouped_indices=grouped_indices, rows=metadata_rows, accelerator=accelerator, args=args, update_idx=update_idx, friction_groups=sample_friction_groups, actions_per_update=actions_per_update, microbatches_per_update=microbatches_per_update, allowed_friction_values=sample_values)
        optimizer.zero_grad(set_to_none=True)
        detached_losses = []
        tensor_batch_size = int(args.grouped_context_tensor_batch_size)
        if tensor_batch_size < 1:
            raise ValueError('grouped_context_tensor_batch_size must be positive.')
        update_started = time.perf_counter()
        for batch_start in range(0, len(sample_indices), tensor_batch_size):
            batch_indices = sample_indices[batch_start:batch_start + tensor_batch_size]
            sync_context = accelerator.no_sync(model) if batch_start + len(batch_indices) < len(sample_indices) else contextlib.nullcontext()
            with sync_context:
                samples = []
                for (offset, sample_index) in enumerate(batch_indices):
                    micro_idx = batch_start + offset
                    sample_data = dataset[sample_index]
                    if random_context_warmup and step <= initial_steps:
                        random_context = random.Random(int(args.seed) + int(step) * 104729 + int(accelerator.process_index) * 1009 + int(micro_idx))
                        pool_size = min(int(args.grouped_context_curriculum_initial_groups), len(group_order))
                        borrowed_index = group_order[random_context.randrange(pool_size)]
                        sample_data = sample_data.copy()
                        sample_data['friction_mu'] = float(friction_values[borrowed_index])
                    elif shared_context_values:
                        sample_group_value = float(metadata_rows[sample_index]['friction_mu'])
                        canonical_value = shared_context_values.get(sample_group_value)
                        if canonical_value is not None:
                            sample_data = sample_data.copy()
                            sample_data['friction_mu'] = canonical_value
                    samples.append(sample_data)
                loss = model(samples if tensor_batch_size > 1 else samples[0])
                detached_losses.extend([loss.detach()] * len(samples))
                accelerator.backward(loss * (len(samples) / len(sample_indices)))
        frozen_context_snapshot = None
        frozen_context_mask = None
        if phase in ('new_context', 'all_context'):
            unwrapped_model = accelerator.unwrap_model(model)
            _mask_context_grad(unwrapped_model, phase_info['train_context_indices'])
            context_table = unwrapped_model.friction_context_table
            train_mask = _context_row_mask(context_table, phase_info['train_context_indices'])
            if not bool(train_mask.all()):
                frozen_context_snapshot = context_table.contexts.detach().clone()
                frozen_context_mask = ~train_mask
                _zero_masked_context_optimizer_state(optimizer, context_table, train_mask)
        if args.max_grad_norm is not None and args.max_grad_norm > 0:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        if frozen_context_snapshot is not None:
            context_table = accelerator.unwrap_model(model).friction_context_table
            with torch.no_grad():
                context_table.contexts[frozen_context_mask].copy_(frozen_context_snapshot[frozen_context_mask])
        if shared_context_indices:
            context_table = accelerator.unwrap_model(model).friction_context_table
            with torch.no_grad():
                for (alias_index, canonical_index) in shared_context_indices.items():
                    context_table.contexts[alias_index].copy_(context_table.contexts[canonical_index])
        accelerator.unwrap_model(model).friction_context_table.clamp_(args.grouped_context_clamp_min, args.grouped_context_clamp_max)
        if context_phase_anchor is not None:
            _project_context_displacement(accelerator.unwrap_model(model).friction_context_table, context_phase_anchor, context_phase_limit)
        if random_context_warmup and step == initial_steps and accelerator.is_main_process:
            print('[curriculum] random C borrowing ended; the initial active groups now permanently use their distinct table entries', flush=True)
        if step == int(phase_info['phase_end']):
            checkpoint_policy = str(getattr(args, 'grouped_context_model_phase_checkpoint_policy', 'all') or 'all').strip().lower()
            cycle_offset = initial_steps + assignment_model_steps
            is_cycle_end = int(activation_steps) > 0 and step > cycle_offset and ((step - cycle_offset) % int(activation_steps) == 0)
            should_checkpoint_model_phase = phase == 'model' and (checkpoint_policy == 'all' or (checkpoint_policy == 'cycle_end' and is_cycle_end))
            if checkpoint_policy not in ('all', 'cycle_end', 'none'):
                raise ValueError(f'grouped_context_model_phase_checkpoint_policy must be all, cycle_end, or none; got {checkpoint_policy!r}.')
            if should_checkpoint_model_phase:
                regular_save_due = int(args.save_steps) > 0 and step % int(args.save_steps) == 0
                if not regular_save_due:
                    if accelerator.is_main_process:
                        print(f'[checkpoint] forcing paired model/context save at model phase end step={step}', flush=True)
                    model_logger.save_model(accelerator, model, f'step-{step}.safetensors')
                last_forced_model_step = step
            _log_context_table(accelerator, model, step, phase, 'curriculum_phase_end')
            _save_phase_context_table(accelerator, model, model_logger, step, phase, 'curriculum_phase_end')
            if context_phase_anchor is not None and accelerator.is_main_process:
                displacement = torch.linalg.vector_norm((accelerator.unwrap_model(model).friction_context_table.contexts - context_phase_anchor).flatten(start_dim=1).float(), dim=1)
                print(f'[context_displacement] step={step} limit={context_phase_limit:.6f} mean={float(displacement.mean().cpu()):.6f} max={float(displacement.max().cpu()):.6f}', flush=True)
        mean_loss = torch.stack([loss.float() for loss in detached_losses]).mean()
        model_logger.on_step_end(accelerator, model, args.save_steps, loss=mean_loss)
        wallclock_due = False
        if wallclock_armed and (not wallclock_checkpoint_done):
            if accelerator.is_main_process:
                requested = time.time() >= wallclock_deadline if wallclock_deadline > 0 else wallclock_request_file.is_file()
                wallclock_flag.fill_(int(requested))
            if accelerator.num_processes > 1:
                torch.distributed.broadcast(wallclock_flag, src=0)
            wallclock_due = bool(wallclock_flag.item())
        if wallclock_due:
            if accelerator.is_main_process:
                print(f'[wallclock_checkpoint] request observed; forcing complete model/C save after optimizer step={step} phase={phase}', flush=True)
            if last_forced_model_step != step:
                _save_wallclock_paired_checkpoint(accelerator, model, model_logger, step, phase, freeze_background)
            else:
                _log_context_table(accelerator, model, step, phase, 'wallclock_checkpoint_existing_pair')
                _save_phase_context_table(accelerator, model, model_logger, step, phase, 'wallclock_checkpoint_existing_pair')
                _protect_paired_checkpoint(accelerator, model_logger, step)
            last_forced_model_step = step
            wallclock_checkpoint_done = True
            accelerator.wait_for_everyone()
            if accelerator.is_main_process and wallclock_request_file is not None:
                consumed = wallclock_request_file.with_name(f'{wallclock_request_file.name}.consumed-step-{int(step)}')
                try:
                    os.replace(wallclock_request_file, consumed)
                except FileNotFoundError:
                    pass
                print(f'[wallclock_checkpoint] protected paired checkpoint step={step}', flush=True)
            accelerator.wait_for_everyone()
        if step == int(phase_info['phase_end']) and phase == 'model' and (step in protected_checkpoint_steps):
            _protect_paired_checkpoint(accelerator, model_logger, step)
        if validation_interval > 0 and step % validation_interval == 0:
            (validation_loss, validation_count) = _run_fixed_validation(accelerator, model, dataset, validation_indices, validation_seed)
            if accelerator.is_main_process:
                print(f'[validation] step={step} loss={validation_loss:.6f} count={validation_count}', flush=True)
    if last_forced_model_step != model_logger.num_steps:
        model_logger.on_training_end(accelerator, model, args.save_steps)

def main() -> None:
    parser = add_grouped_context_config(wan_parser())
    args = parser.parse_args()
    if args.config is not None:
        args = merge_yaml_and_args(args.config, parser, args)
    set_global_seed(args.seed)
    runtime_config = prepare_runtime_config(args)
    friction_values = _unique_friction_values(args.dataset_metadata_path)
    background_values = _unique_background_values(args.dataset_metadata_path) if bool(getattr(args, 'background_context_enabled', False)) else []
    loggers = [name for name in ('wandb', 'swanlab') if getattr(args, f'use_{name}', False)]
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision=args.mixed_precision, log_with=loggers or None, kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters), InitProcessGroupKwargs(timeout=timedelta(hours=1))])
    dataset = build_dataset(args, runtime_config)
    model = GroupedContextStage1Module(model_paths=json.dumps(runtime_config['model_paths_list']), model_id_with_origin_paths=args.model_id_with_origin_paths, tokenizer_path=runtime_config['tokenizer_path'], enable_text=args.enable_text, trainable_models=args.trainable_models, lora_base_model=args.lora_base_model, lora_target_modules=args.lora_target_modules, lora_rank=args.lora_rank, lora_checkpoint=args.lora_checkpoint, preset_lora_path=args.preset_lora_path, preset_lora_model=args.preset_lora_model, use_gradient_checkpointing=args.use_gradient_checkpointing, use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload, extra_inputs=args.extra_inputs, modules=runtime_config['modules'], fp8_models=args.fp8_models, offload_models=args.offload_models, ckpt_path=args.ckpt_path, task=args.task, device='cpu' if args.initialize_model_on_cpu else accelerator.device, max_timestep_boundary=args.max_timestep_boundary, min_timestep_boundary=args.min_timestep_boundary, num_history_frames=args.num_history_frames, args=args, grouped_args=args, friction_values=friction_values, background_values=background_values)
    restored_global_context = False
    if getattr(args, 'grouped_context_resume_context_table', None):
        restored_global_context = _load_grouped_context_table(model, args.grouped_context_resume_context_table)
        _load_background_context_table(model, args.grouped_context_resume_context_table)
        if accelerator.is_main_process:
            print(f'[resume] restored context table from {args.grouped_context_resume_context_table} at step={int(args.grouped_context_resume_step)}', flush=True)
    if accelerator.is_main_process:
        print(f'Grouped-C Stage1: groups={len(friction_values)} context_dim={args.physical_context_dim} tokens={args.physical_context_tokens} background_groups={len(background_values)}', flush=True)
    if bool(getattr(args, 'grouped_context_stage_checkpoints_locally', False)):
        local_checkpoint_root = getattr(args, 'grouped_context_local_checkpoint_root', None) or os.environ.get('BWM_LOCAL_CHECKPOINT_ROOT')
        if not local_checkpoint_root:
            raise ValueError('Local checkpoint staging is enabled but no local root was configured.')
        model_logger = StagedGroupedContextModelLogger(args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt, save_minutes=args.checkpoint_save_minutes, keep_last=args.checkpoint_keep_last, log_steps=args.log_steps, local_checkpoint_root=local_checkpoint_root)
    else:
        model_logger = GroupedContextModelLogger(args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt, save_minutes=args.checkpoint_save_minutes, keep_last=args.checkpoint_keep_last, log_steps=args.log_steps)
    model_logger.num_steps = int(getattr(args, 'grouped_context_resume_step', 0) or 0)
    if int(getattr(args, 'grouped_context_curriculum_initial_groups', 0) or 0) > 0:
        launch_curriculum_grouped_stage1(accelerator, dataset, model, model_logger, args)
    elif int(getattr(args, 'grouped_context_structured_updates', 0) or 0) > 0:
        launch_structured_grouped_stage1(accelerator, dataset, model, model_logger, args)
    else:
        launch_grouped_stage1(accelerator, dataset, model, model_logger, args)
if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main()
