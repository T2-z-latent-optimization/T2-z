"""Frozen-DINOv2 amortized environment-code components."""
from __future__ import annotations
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
import yaml
from transformers import AutoModel
from .environment_sampling import sample_environment_slots, sample_slots
from .window_sampling import WindowIndexSelector

@dataclass(frozen=True)
class SupportQueryEpisodeIndices:
    environment_id: Any
    support_index: int
    query_indices: tuple[int, ...]
    additional_support_indices: tuple[int, ...] = ()

    @property
    def support_indices(self) -> tuple[int, ...]:
        """Support indices."""
        return (self.support_index, *self.additional_support_indices)

class EnvironmentEpisodeSampler:
    """Sample supports from each environment batch; use the entire batch as queries."""

    def __init__(self, *, metadata_path: str | Path, active_environment_manifest: str | Path, seed: int, support_k: int=1, support_k_choices: tuple[int, ...] | None=None, trajectories_per_environment: int=0, environment_key: str='mu_index', action_key: str='action_id', distinct_actions: bool=False, window_sampling_mode: str='uniform_episode_then_window', window_kind_field: str='sampling_kind', preferred_window_kind: str='precise', preferred_window_probability: float=0.5) -> None:
        self.metadata_path = Path(metadata_path)
        self.environment_key = str(environment_key)
        self.action_key = str(action_key)
        self.seed = int(seed)
        self.support_k = int(support_k)
        self.support_k_choices = (self.support_k,) if support_k_choices is None else tuple((int(value) for value in support_k_choices))
        self.trajectories_per_environment = int(trajectories_per_environment)
        self.distinct_actions = bool(distinct_actions)
        if self.support_k < 1:
            raise ValueError('support_k must be positive.')
        if not self.support_k_choices or min(self.support_k_choices) < 1:
            raise ValueError('support_k_choices must contain positive integers.')
        if self.trajectories_per_environment < 0:
            raise ValueError('trajectories_per_environment must be non-negative.')
        if self.trajectories_per_environment:
            if self.trajectories_per_environment < max(self.support_k_choices):
                raise ValueError('trajectories_per_environment must be at least every support K.')
        with self.metadata_path.open('r', encoding='utf-8') as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]
        with Path(active_environment_manifest).open('r', encoding='utf-8') as handle:
            manifest = yaml.safe_load(handle)
        selection = manifest['selection']
        self.active_environment_ids = tuple(selection['active_environment_ids'])
        raw_weights = selection.get('environment_sampling_weights', {})
        self.environment_weights = {environment_id: float(raw_weights.get(environment_id, raw_weights.get(str(environment_id), 1.0))) for environment_id in self.active_environment_ids}
        grouped: dict[Any, list[int]] = {value: [] for value in self.active_environment_ids}
        for index, row in enumerate(self.rows):
            environment_id = row[self.environment_key]
            if environment_id in grouped:
                grouped[environment_id].append(index)
        grouped_by_action: dict[Any, dict[int, list[int]]] = {}
        for environment_id, indices in grouped.items():
            indices.sort(key=lambda index: (int(self.rows[index][self.action_key]), index))
            actions = [int(self.rows[index][self.action_key]) for index in indices]
            by_action: dict[int, list[int]] = {}
            for index, action_id in zip(indices, actions):
                by_action.setdefault(action_id, []).append(index)
            grouped_by_action[environment_id] = by_action
        self.grouped_indices = grouped
        self.grouped_indices_by_action = grouped_by_action
        self.window_selector = WindowIndexSelector(self.rows, mode=window_sampling_mode, kind_field=window_kind_field, preferred_kind=preferred_window_kind, preferred_probability=preferred_window_probability)
        if self.window_selector.enabled and (not self.distinct_actions):
            raise ValueError('Episode/window balancing requires distinct_actions=True and an episode identifier as action_key (not an action level).')

    def sample(self, *, step: int, process_index: int, num_processes: int, environments_per_rank: int) -> tuple[SupportQueryEpisodeIndices, ...]:
        global_count = int(num_processes) * int(environments_per_rank)
        rng = random.Random(self.seed + int(step) * 104729)
        selected = sample_environment_slots(rng, self.active_environment_ids, global_count, self.environment_weights)
        episodes = []
        for environment_id in selected:
            indices = self.grouped_indices[environment_id]
            episode_support_k = self.support_k_choices[0] if len(self.support_k_choices) == 1 else rng.choice(self.support_k_choices)
            if self.distinct_actions:
                by_action = self.grouped_indices_by_action[environment_id]
                action_ids = sorted(by_action)
                if self.trajectories_per_environment:
                    candidate_actions = sample_slots(rng, action_ids, self.trajectories_per_environment)
                    support_positions = rng.sample(range(len(candidate_actions)), episode_support_k)
                    support_actions = [candidate_actions[index] for index in support_positions]
                    remaining_actions = [action_id for index, action_id in enumerate(candidate_actions) if index not in support_positions]
                else:
                    if episode_support_k == 1:
                        support_actions = [rng.choice(action_ids)]
                    else:
                        support_actions = rng.sample(action_ids, episode_support_k)
                    remaining_actions = [action_id for action_id in action_ids if action_id not in support_actions]
                support_indices = [self.window_selector.choose(rng, by_action[action_id]) for action_id in support_actions]
                support_index = support_indices[0]
                remaining_actions.sort()
                remaining_queries = [self.window_selector.choose(rng, by_action[action_id]) for action_id in remaining_actions]
            else:
                if self.trajectories_per_environment:
                    candidate_indices = sample_slots(rng, indices, self.trajectories_per_environment)
                    support_positions = rng.sample(range(len(candidate_indices)), episode_support_k)
                    support_indices = [candidate_indices[index] for index in support_positions]
                    remaining_queries = [index for position, index in enumerate(candidate_indices) if position not in support_positions]
                else:
                    if episode_support_k == 1:
                        support_indices = [indices[rng.randrange(len(indices))]]
                    else:
                        support_indices = rng.sample(indices, episode_support_k)
                    support_index_set = set(support_indices)
                    remaining_queries = [index for index in indices if index not in support_index_set]
                support_index = support_indices[0]
            query_indices = tuple((*remaining_queries, *support_indices))
            episodes.append(SupportQueryEpisodeIndices(environment_id=environment_id, support_index=support_index, query_indices=query_indices, additional_support_indices=tuple(support_indices[1:])))
        start = int(process_index) * int(environments_per_rank)
        end = start + int(environments_per_rank)
        return tuple(episodes[start:end])

class ActionChunkEncoder(nn.Module):

    def __init__(self, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.step_projection = nn.Sequential(nn.Linear(self.action_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.temporal_encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, chunks: list[torch.Tensor]) -> torch.Tensor:
        outputs = []
        for chunk in chunks:
            if chunk.ndim != 2 or chunk.shape[-1] != self.action_dim:
                raise ValueError(f'Expected action chunk [T,{self.action_dim}], got {tuple(chunk.shape)}.')
            encoded = self.step_projection(chunk).unsqueeze(0)
            _, hidden = self.temporal_encoder(encoded)
            outputs.append(hidden[-1, 0])
        return torch.stack(outputs, dim=0)

class DINOv2AmortizedContextEncoder(nn.Module):
    """Maps one observed support video/action trajectory to a bounded 32-D Z."""

    def __init__(self, *, model_path: str | Path, sampled_frames: int=11, temporal_stride: int=4, action_dim: int=14, hidden_dim: int=256, action_hidden_dim: int=128, output_dim: int=32, aggregation_mode: str='concat_mlp', mlp_hidden_dim: int=1024) -> None:
        super().__init__()
        self.model_path = str(Path(model_path).expanduser())
        self.sampled_frames = int(sampled_frames)
        self.temporal_stride = int(temporal_stride)
        self.action_dim = int(action_dim)
        self.output_dim = int(output_dim)
        self.aggregation_mode = str(aggregation_mode)
        if self.sampled_frames < 2:
            raise ValueError('sampled_frames must be at least two.')
        if self.temporal_stride < 1:
            raise ValueError('temporal_stride must be positive.')
        if self.aggregation_mode != 'concat_mlp':
            raise ValueError('Only concat_mlp aggregation is supported')
        if int(mlp_hidden_dim) < 1:
            raise ValueError('mlp_hidden_dim must be positive.')
        self.dino = AutoModel.from_pretrained(self.model_path, local_files_only=True)
        self.dino.requires_grad_(False)
        self.dino.eval()
        dino_dim = int(self.dino.config.hidden_size)
        processor_path = Path(self.model_path) / 'preprocessor_config.json'
        processor = {}
        if processor_path.is_file():
            with processor_path.open('r', encoding='utf-8') as handle:
                processor = json.load(handle)
        image_mean = processor.get('image_mean', [0.485, 0.456, 0.406])
        image_std = processor.get('image_std', [0.229, 0.224, 0.225])
        size = processor.get('size', {'shortest_edge': 224})
        if isinstance(size, dict):
            image_size = int(size.get('height', size.get('shortest_edge', 224)))
        else:
            image_size = int(size)
        self.image_size = image_size
        self.register_buffer('image_mean', torch.tensor(image_mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer('image_std', torch.tensor(image_std, dtype=torch.float32).view(1, 3, 1, 1))
        self.visual_projection = nn.Sequential(nn.LayerNorm(2 * dino_dim), nn.Linear(2 * dino_dim, hidden_dim), nn.GELU())
        self.action_encoder = ActionChunkEncoder(self.action_dim, action_hidden_dim)
        segment_dim = hidden_dim * 2 + action_hidden_dim
        concatenated_dim = (self.sampled_frames - 1) * segment_dim
        self.concat_mlp = nn.Sequential(nn.LayerNorm(concatenated_dim), nn.Linear(concatenated_dim, int(mlp_hidden_dim)), nn.GELU(), nn.LayerNorm(int(mlp_hidden_dim)), nn.Linear(int(mlp_hidden_dim), int(mlp_hidden_dim)), nn.GELU())
        summary_dim = int(mlp_hidden_dim)
        self.output_head = nn.Sequential(nn.LayerNorm(summary_dim), nn.Linear(summary_dim, output_dim))
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

    def train(self, mode: bool=True):
        super().train(mode)
        self.dino.eval()
        return self

    @staticmethod
    def _canonical_video(video: Any) -> torch.Tensor:
        tensor = torch.as_tensor(video, dtype=torch.float32)
        if tensor.ndim == 5:
            tensor = tensor[0]
        if tensor.ndim != 4:
            raise ValueError(f'Support video must have shape [V,C,T,H,W] or [C,T,H,W], got {tuple(tensor.shape)}.')
        if tensor.shape[0] != 3:
            raise ValueError(f'Support video must be RGB, got shape={tuple(tensor.shape)}.')
        return tensor

    def extract_visual_features(self, video: Any) -> tuple[torch.Tensor, tuple[int, ...], int]:
        video_tensor = self._canonical_video(video)
        frame_count = int(video_tensor.shape[1])
        indices_tensor = torch.arange(0, frame_count, self.temporal_stride, dtype=torch.long)
        if int(indices_tensor[-1]) != frame_count - 1:
            indices_tensor = torch.cat((indices_tensor, torch.tensor([frame_count - 1])))
        aligned_anchor_count = int(torch.unique(indices_tensor).numel())
        if aligned_anchor_count < self.sampled_frames:
            raise ValueError(f'Need at least {self.sampled_frames} Wan-aligned anchors for T={frame_count} and stride={self.temporal_stride}, got {indices_tensor.tolist()}.')
        if aligned_anchor_count > self.sampled_frames:
            sample_positions = torch.linspace(0, aligned_anchor_count - 1, steps=self.sampled_frames, dtype=torch.float64).round().to(dtype=torch.long)
            indices_tensor = indices_tensor.index_select(0, sample_positions)
        indices = tuple((int(value) for value in indices_tensor.tolist()))
        device = next(self.dino.parameters()).device
        frames = video_tensor[:, indices].permute(1, 0, 2, 3).to(device=device)
        frames = (frames + 1.0) * 0.5
        frames = F.interpolate(frames, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        frames = (frames - self.image_mean.to(device=device)) / self.image_std.to(device=device)
        with torch.no_grad():
            hidden = self.dino(pixel_values=frames).last_hidden_state
            cls = hidden[:, 0]
            mean_patch = hidden[:, 1:].mean(dim=1)
            features = torch.cat((cls, mean_patch), dim=-1).detach()
        return (features, indices, frame_count)

    def _canonical_actions(self, action: Any, device: torch.device) -> torch.Tensor:
        tensor = torch.as_tensor(action, dtype=torch.float32, device=device)
        while tensor.ndim > 2 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim == 1:
            if tensor.numel() % self.action_dim != 0:
                raise ValueError(f'Cannot reshape action vector of length {tensor.numel()}.')
            tensor = tensor.view(-1, self.action_dim)
        if tensor.ndim != 2:
            tensor = tensor.reshape(-1, tensor.shape[-1])
        if tensor.shape[-1] != self.action_dim and tensor.shape[0] == self.action_dim:
            tensor = tensor.transpose(0, 1)
        if tensor.shape[-1] != self.action_dim:
            raise ValueError(f'Expected action dim {self.action_dim}, got shape={tuple(tensor.shape)}.')
        return tensor

    def _summarize_support(self, *, visual_features: torch.Tensor, action: Any, frame_indices: tuple[int, ...], frame_count: int) -> torch.Tensor:
        device = visual_features.device
        actions = self._canonical_actions(action, device)
        action_count = int(actions.shape[0])
        chunks = []
        for start_frame, end_frame in zip(frame_indices[:-1], frame_indices[1:]):
            denominator = max(int(frame_count) - 1, 1)
            start = int(round(start_frame / denominator * max(action_count - 1, 0)))
            end = int(round(end_frame / denominator * max(action_count - 1, 0)))
            end = max(start + 1, min(action_count, end))
            chunks.append(actions[start:end])
        frame_features = self.visual_projection(visual_features)
        action_features = self.action_encoder(chunks)
        segment_features = torch.cat((frame_features[:-1], frame_features[1:], action_features), dim=-1)
        summary = self.concat_mlp(segment_features.reshape(1, -1))
        return summary

    def project_support(self, *, visual_features: torch.Tensor, action: Any, frame_indices: tuple[int, ...], frame_count: int) -> torch.Tensor:
        summary = self._summarize_support(visual_features=visual_features, action=action, frame_indices=frame_indices, frame_count=frame_count)
        return torch.sigmoid(self.output_head(summary))

    def project_supports(self, *, visual_features: tuple[torch.Tensor, ...], actions: tuple[Any, ...], frame_indices: tuple[tuple[int, ...], ...], frame_counts: tuple[int, ...]) -> torch.Tensor:
        support_count = len(visual_features)
        if support_count < 1:
            raise ValueError('At least one support trajectory is required.')
        if not (len(actions) == support_count and len(frame_indices) == support_count and (len(frame_counts) == support_count)):
            raise ValueError('Support feature/action/frame metadata lengths must match.')
        if support_count == 1:
            return self.project_support(visual_features=visual_features[0], action=actions[0], frame_indices=frame_indices[0], frame_count=frame_counts[0])
        summaries = [self._summarize_support(visual_features=features, action=action, frame_indices=indices, frame_count=count) for features, action, indices, count in zip(visual_features, actions, frame_indices, frame_counts)]
        pooled_summary = torch.cat(summaries, dim=0).mean(dim=0, keepdim=True)
        return torch.sigmoid(self.output_head(pooled_summary))

    def forward(self, video: Any, action: Any) -> torch.Tensor:
        features, indices, frame_count = self.extract_visual_features(video)
        return self.project_support(visual_features=features, action=action, frame_indices=indices, frame_count=frame_count)
