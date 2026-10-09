from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class Chunk:
    start: int
    end: int
    stride: int = 1
    observation_timing: str = "pre_step"

    def __post_init__(self):
        if self.start < 0 or self.end < self.start or self.stride < 1:
            raise ValueError(f"Invalid native-frame window: {self}")
        if (self.end - self.start) % self.stride:
            raise ValueError("Chunk end must lie on the sampling grid")

    @property
    def indices(self) -> np.ndarray:
        return np.arange(self.start, self.end + 1, self.stride, dtype=np.int64)

    @property
    def length(self) -> int:
        return len(self.indices)


@dataclass
class Candidate:
    index: int
    order: float
    parameters: dict[str, Any]
    model_actions: np.ndarray
    conditioning: str = "exact_open_loop"

    def to_wire(self) -> dict:
        return {
            "index": self.index, "order": self.order,
            "parameters": self.parameters, "model_actions": self.model_actions,
            "conditioning": self.conditioning,
        }


class Simulator(Protocol):
    def prepare(self) -> dict: ...
    def candidates(self) -> list[dict]: ...
    def fingerprint(self) -> str: ...
    def execute(self, action_index: int) -> dict: ...
    def close(self) -> None: ...


class Predictor(Protocol):
    def freeze_context(self, request: dict) -> dict: ...
    def predict(self, request: dict) -> np.ndarray: ...
    def close(self) -> None: ...
