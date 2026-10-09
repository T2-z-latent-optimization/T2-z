"""Explicit lifecycle management for per-environment TTT fast states."""

from __future__ import annotations

from contextlib import contextmanager
from enum import Enum
from typing import Iterator

import torch

from .fast_weight import TTTKVBState, TTTMLPMemory


class TTTKVBMode(str, Enum):
    DISABLED = "disabled"
    SUPPORT_WRITE = "support_write"
    QUERY_READ = "query_read"
    CAUSAL_SCAN = "causal_scan"


class TTTKVBController:
    """Owns ephemeral fast states without registering them in checkpoints."""

    def __init__(self) -> None:
        self.mode = TTTKVBMode.DISABLED
        self.batch_size: int | None = None
        self.differentiable = False
        self.query_state_indices: torch.Tensor | None = None
        self._states: dict[str, TTTKVBState] = {}
        self._write_statistics: dict[str, list[dict[str, float]]] = {}

    def reset(self, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = int(batch_size)
        self._states.clear()
        self._write_statistics.clear()
        self.query_state_indices = None

    def detach_fast_state(self) -> None:
        """Keep learned state values while truncating the cross-chunk graph.

        Call only after the preceding segment's backward has completed and
        outside an active read/write context. Each tensor becomes a gradient-
        enabled leaf so subsequent inner updates can differentiate locally.
        Initial fast weights are not reloaded and write statistics are retained.
        """
        if self.mode != TTTKVBMode.DISABLED or self.batch_size is None:
            raise RuntimeError("Detach fast state between segments, after leaving the model context")
        self._states = {
            layer_id: state.detached(requires_grad=True)
            for layer_id, state in self._states.items()
        }

    def clear(self) -> None:
        self.mode = TTTKVBMode.DISABLED
        self.batch_size = None
        self.differentiable = False
        self.query_state_indices = None
        self._states.clear()
        self._write_statistics.clear()

    @contextmanager
    def support_write(self, differentiable: bool) -> Iterator[None]:
        if self.batch_size is None:
            raise RuntimeError("reset(batch_size) must be called before support_write")
        previous = (self.mode, self.differentiable, self.query_state_indices)
        self.mode = TTTKVBMode.SUPPORT_WRITE
        self.differentiable = bool(differentiable)
        self.query_state_indices = None
        try:
            yield
        finally:
            self.mode, self.differentiable, self.query_state_indices = previous

    @contextmanager
    def query_read(self, state_indices: torch.Tensor | None = None) -> Iterator[None]:
        if self.batch_size is None:
            raise RuntimeError("reset(batch_size) must be called before query_read")
        previous = (self.mode, self.differentiable, self.query_state_indices)
        self.mode = TTTKVBMode.QUERY_READ
        self.query_state_indices = state_indices
        try:
            yield
        finally:
            self.mode, self.differentiable, self.query_state_indices = previous

    @contextmanager
    def causal_scan(self, differentiable: bool = True) -> Iterator[None]:
        if self.batch_size is None:
            raise RuntimeError("reset(batch_size) must be called before causal_scan")
        previous = (self.mode, self.differentiable, self.query_state_indices)
        self.mode = TTTKVBMode.CAUSAL_SCAN
        self.differentiable = bool(differentiable)
        self.query_state_indices = None
        try:
            yield
        finally:
            self.mode, self.differentiable, self.query_state_indices = previous

    def scan(self, layer_id: str, memory: TTTMLPMemory, tokens: torch.Tensor) -> torch.Tensor:
        if self.mode != TTTKVBMode.CAUSAL_SCAN:
            raise RuntimeError(f"Cannot scan fast state in mode={self.mode}")
        state, output, stats = memory.write_then_read(
            tokens,
            self._states.get(layer_id),
            differentiable=self.differentiable,
        )
        self._states[layer_id] = state
        self._write_statistics.setdefault(layer_id, []).extend(stats)
        return output

    def write(self, layer_id: str, memory: TTTMLPMemory, tokens: torch.Tensor) -> None:
        if self.mode != TTTKVBMode.SUPPORT_WRITE:
            raise RuntimeError(f"Cannot write fast state in mode={self.mode}")
        state, stats = memory.write(
            tokens,
            self._states.get(layer_id),
            differentiable=self.differentiable,
        )
        self._states[layer_id] = state
        self._write_statistics.setdefault(layer_id, []).extend(stats)

    def read(self, layer_id: str, memory: TTTMLPMemory, tokens: torch.Tensor) -> torch.Tensor:
        if self.mode != TTTKVBMode.QUERY_READ:
            raise RuntimeError(f"Cannot read fast state in mode={self.mode}")
        state = self._states.get(layer_id)
        if state is None:
            if self.batch_size is None:
                raise RuntimeError("Fast state has not been initialized")
            state = memory.initial_state(
                batch_size=self.batch_size,
                device=tokens.device,
                differentiable=torch.is_grad_enabled(),
            )
        if self.query_state_indices is not None:
            indices = self.query_state_indices.to(device=state.w1.device, dtype=torch.long)
            state = state.index_select(indices)
        return memory.read(tokens, state)

    def write_statistics(self) -> dict[str, list[dict[str, float]]]:
        return {key: list(values) for key, values in self._write_statistics.items()}

    def state_norms(self) -> dict[str, float]:
        result = {}
        for layer_id, state in self._states.items():
            total = sum(tensor.detach().float().square().sum() for tensor in state.tensors)
            result[layer_id] = float(torch.sqrt(total).cpu())
        return result
