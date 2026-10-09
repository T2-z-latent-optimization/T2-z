"""Environment and episode stream sampling for TTT."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random

import yaml

from ..environment_sampling import sample_environment_slots, sample_slots
from ..window_sampling import WindowIndexSelector


@dataclass(frozen=True)
class StreamEpisode:
    """One randomly ordered stream from a single physical environment."""

    environment_id: int
    indices: tuple[int, ...]
    action_ids: tuple[int, ...]


class EnvironmentStreamSampler:
    """Sample disjoint action chunks and assign disjoint environments to ranks."""

    def __init__(
        self,
        *,
        metadata_path: str | Path,
        active_environment_manifest: str | Path,
        seed: int,
        sequence_length: int = 6,
        environment_key: str = "mu_index",
        action_key: str = "action_id",
        distinct_actions: bool = False,
        window_sampling_mode: str = "uniform_episode_then_window",
        window_kind_field: str = "sampling_kind",
        preferred_window_kind: str = "precise",
        preferred_window_probability: float = 0.5,
    ) -> None:
        self.seed = int(seed)
        self.sequence_length = int(sequence_length)
        self.environment_key = str(environment_key)
        self.action_key = str(action_key)
        self.distinct_actions = bool(distinct_actions)
        if self.sequence_length < 1:
            raise ValueError("A stream must contain at least one chunk.")

        with Path(metadata_path).open("r", encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]
        with Path(active_environment_manifest).open("r", encoding="utf-8") as handle:
            manifest = yaml.safe_load(handle)
        selection = manifest["selection"]
        self.active_environment_ids = tuple(
            int(value) for value in selection["active_environment_ids"]
        )
        raw_weights = selection.get("environment_sampling_weights", {})
        self.environment_weights = {
            environment_id: float(
                raw_weights.get(
                    environment_id,
                    raw_weights.get(str(environment_id), 1.0),
                )
            )
            for environment_id in self.active_environment_ids
        }

        grouped: dict[int, list[int]] = {
            environment_id: [] for environment_id in self.active_environment_ids
        }
        for index, row in enumerate(self.rows):
            environment_id = int(row[self.environment_key])
            if environment_id in grouped:
                grouped[environment_id].append(index)
        grouped_by_action: dict[int, dict[int, list[int]]] = {}
        for environment_id, indices in grouped.items():
            indices.sort(key=lambda index: (int(self.rows[index][self.action_key]), index))
            action_ids = [int(self.rows[index][self.action_key]) for index in indices]
            by_action: dict[int, list[int]] = {}
            for index, action_id in zip(indices, action_ids):
                by_action.setdefault(action_id, []).append(index)
            grouped_by_action[environment_id] = by_action
        self.grouped_indices = grouped
        self.grouped_indices_by_action = grouped_by_action
        self.window_selector = WindowIndexSelector(
            self.rows,
            mode=window_sampling_mode,
            kind_field=window_kind_field,
            preferred_kind=preferred_window_kind,
            preferred_probability=preferred_window_probability,
        )
        if self.window_selector.enabled and not self.distinct_actions:
            raise ValueError(
                "Episode/window balancing requires distinct_actions=True and an "
                "episode identifier as action_key (not an action level)."
            )

    def sample(
        self,
        *,
        step: int,
        process_index: int,
        num_processes: int,
        environments_per_rank: int = 1,
    ) -> tuple[StreamEpisode, ...]:
        global_environment_count = int(num_processes) * int(environments_per_rank)
        rng = random.Random(self.seed + int(step) * 104729)
        selected_environments = sample_environment_slots(
            rng,
            self.active_environment_ids,
            global_environment_count,
            self.environment_weights,
        )
        episodes = []
        for environment_id in selected_environments:
            if self.distinct_actions:
                by_action = self.grouped_indices_by_action[environment_id]
                selected_actions = sample_slots(rng, sorted(by_action), self.sequence_length)
                indices = [
                    self.window_selector.choose(rng, by_action[action_id])
                    for action_id in selected_actions
                ]
            else:
                indices = sample_slots(
                    rng, self.grouped_indices[environment_id], self.sequence_length
                )
            episodes.append(
                StreamEpisode(
                    environment_id=environment_id,
                    indices=tuple(indices),
                    action_ids=tuple(
                        int(self.rows[index][self.action_key]) for index in indices
                    ),
                )
            )
        start = int(process_index) * int(environments_per_rank)
        end = start + int(environments_per_rank)
        return tuple(episodes[start:end])
