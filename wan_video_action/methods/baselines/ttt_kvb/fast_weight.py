"""Differentiable TTT-MLP fast weights with a key-value binding objective.

This module implements the learner, not the support/query policy.  The fast
state is explicit and never stored as an ``nn.Parameter``.  Shared initial
weights, Q/K/V views, the inner LayerNorm, and the token-wise learning-rate
gate are outer-loop parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class TTTKVBState:
    """Per-environment, per-head fast state for a two-layer TTT MLP."""

    w1: torch.Tensor
    b1: torch.Tensor
    w2: torch.Tensor
    b2: torch.Tensor

    @property
    def tensors(self) -> tuple[torch.Tensor, ...]:
        return self.w1, self.b1, self.w2, self.b2

    @property
    def batch_size(self) -> int:
        return int(self.w1.shape[0])

    def index_select(self, indices: torch.Tensor) -> "TTTKVBState":
        return TTTKVBState(*(tensor.index_select(0, indices) for tensor in self.tensors))

    def detached(self, requires_grad: bool = False) -> "TTTKVBState":
        tensors = []
        for tensor in self.tensors:
            value = tensor.detach()
            if requires_grad:
                value = value.requires_grad_(True)
            tensors.append(value)
        return TTTKVBState(*tensors)


def _head_layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    variance = (x - mean).square().mean(dim=-1, keepdim=True)
    normalized = (x - mean) * torch.rsqrt(variance + eps)
    return normalized * weight[None, :, None, :] + bias[None, :, None, :]


class TTTMLPMemory(nn.Module):
    """Original-style TTT-MLP learner adapted to explicit support/query use.

    The inner model is ``q + LN(MLP(q))``.  Support tokens train the residual
    MLP to map K to ``V - K`` using MSE.  Query tokens only evaluate the final
    fast state.  Fast-state arithmetic is FP32 even when the Wan backbone uses
    BF16.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        expansion: int = 4,
        base_inner_lr: float = 0.1,
        inner_batch_size: int = 64,
        write_token_budget: int = 512,
        norm_eps: float = 1e-6,
        scan_checkpoint_updates: int = 0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        if expansion <= 0:
            raise ValueError("expansion must be positive")

        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.hidden_dim = int(expansion) * self.head_dim
        self.base_inner_lr = float(base_inner_lr)
        self.inner_batch_size = int(inner_batch_size)
        self.write_token_budget = int(write_token_budget)
        self.norm_eps = float(norm_eps)
        self.scan_checkpoint_updates = int(scan_checkpoint_updates)
        if self.scan_checkpoint_updates < 0:
            raise ValueError("scan_checkpoint_updates must be nonnegative")




        self.q_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)

        self.initial_w1 = nn.Parameter(
            torch.empty(self.num_heads, self.head_dim, self.hidden_dim)
        )
        self.initial_b1 = nn.Parameter(torch.zeros(self.num_heads, 1, self.hidden_dim))
        self.initial_w2 = nn.Parameter(
            torch.empty(self.num_heads, self.hidden_dim, self.head_dim)
        )
        self.initial_b2 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))
        nn.init.normal_(self.initial_w1, mean=0.0, std=0.02)
        nn.init.normal_(self.initial_w2, mean=0.0, std=0.02)

        self.fast_norm_weight = nn.Parameter(torch.ones(self.num_heads, self.head_dim))
        self.fast_norm_bias = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))


        self.lr_weight = nn.Parameter(torch.empty(self.num_heads, self.dim))
        self.lr_bias = nn.Parameter(torch.zeros(self.num_heads))
        nn.init.normal_(self.lr_weight, mean=0.0, std=0.02)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
        differentiable: bool,
    ) -> TTTKVBState:
        with torch.enable_grad():
            values = []
            for parameter in (
                self.initial_w1,
                self.initial_b1,
                self.initial_w2,
                self.initial_b2,
            ):
                value = parameter.to(device=device, dtype=torch.float32)
                value = value.unsqueeze(0).expand(int(batch_size), *value.shape)
                if not differentiable:
                    value = value.detach().clone().requires_grad_(True)
                values.append(value)
        return TTTKVBState(*values)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.reshape(batch, length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, length, _ = x.shape
        return x.permute(0, 2, 1, 3).reshape(batch, length, self.dim)

    def _fast_residual(self, x: torch.Tensor, state: TTTKVBState) -> torch.Tensor:
        hidden = torch.einsum("bhld,bhdf->bhlf", x, state.w1) + state.b1
        hidden = F.gelu(hidden, approximate="tanh")
        output = torch.einsum("bhlf,bhfd->bhld", hidden, state.w2) + state.b2
        return _head_layer_norm(
            output,
            self.fast_norm_weight.float(),
            self.fast_norm_bias.float(),
            self.norm_eps,
        )

    def _token_eta(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.einsum("bld,hd->bhl", x.float(), self.lr_weight.float())
        logits = logits + self.lr_bias.float()[None, :, None]
        return self.base_inner_lr * torch.sigmoid(logits) / float(self.head_dim)

    def _uniform_token_indices(self, length: int, device: torch.device) -> torch.Tensor:
        budget = self.write_token_budget
        if budget <= 0 or length <= budget:
            return torch.arange(length, device=device)
        indices = torch.linspace(0, length - 1, steps=budget, device=device)
        return indices.round().to(dtype=torch.long).unique(sorted=True)

    def _update_once(
        self,
        x: torch.Tensor,
        state: TTTKVBState,
        differentiable: bool,
    ) -> tuple[TTTKVBState, dict[str, float]]:
        k = self._split_heads(self.k_proj(x).float())
        v = self._split_heads(self.v_proj(x).float())
        eta = self._token_eta(x)

        prediction = k + self._fast_residual(k, state)
        token_loss = 0.5 * (prediction - v).square().mean(dim=-1)
        weighted_loss = (eta * token_loss).sum()
        gradients = torch.autograd.grad(
            weighted_loss,
            state.tensors,
            create_graph=differentiable,
            retain_graph=differentiable,
            allow_unused=False,
        )
        updated = TTTKVBState(*(value - grad for value, grad in zip(state.tensors, gradients)))

        displacement_sq = sum(
            (after.detach() - before.detach()).float().square().sum()
            for before, after in zip(state.tensors, updated.tensors)
        )
        stats = {
            "binding_loss": float(token_loss.detach().mean().cpu()),
            "state_delta_l2": float(torch.sqrt(displacement_sq).cpu()),
            "eta_mean": float(eta.detach().mean().cpu()),
        }
        if not differentiable:
            updated = updated.detached(requires_grad=True)
        return updated, stats

    def write(
        self,
        x: torch.Tensor,
        state: TTTKVBState | None,
        differentiable: bool,
    ) -> tuple[TTTKVBState, list[dict[str, float]]]:
        """Write a token sequence into the fast state.

        Uniform subsampling is deterministic and spans the flattened Wan video
        token sequence, so all temporal regions remain represented.
        """

        if x.ndim != 3:
            raise ValueError(f"Expected support tokens [B,L,D], got {tuple(x.shape)}")
        with torch.enable_grad():
            selected = self._uniform_token_indices(int(x.shape[1]), x.device)
            support = x.detach().index_select(1, selected)
            if state is None:
                state = self.initial_state(
                    batch_size=int(support.shape[0]),
                    device=support.device,
                    differentiable=differentiable,
                )
            if state.batch_size != int(support.shape[0]):
                raise ValueError(
                    f"Fast-state batch={state.batch_size} does not match support batch={support.shape[0]}"
                )

            chunk = self.inner_batch_size if self.inner_batch_size > 0 else int(support.shape[1])
            statistics = []
            for start in range(0, int(support.shape[1]), chunk):
                state, stats = self._update_once(
                    support[:, start : start + chunk],
                    state,
                    differentiable=differentiable,
                )
                statistics.append(stats)
        return state, statistics

    def _checkpointed_scan_segment(self, tokens: torch.Tensor, *state_tensors):
        """Pure segment: explicit fast state in/out, no controller mutation.

        Non-reentrant checkpointing preserves the higher-order gradients from
        the inner autograd.grad calls. Replaying the entire Wan block would not
        be safe because the controller owns mutable cross-chunk state.
        """
        state = TTTKVBState(*state_tensors)
        chunk_size = self.inner_batch_size if self.inner_batch_size > 0 else int(tokens.shape[1])
        outputs, statistics = [], []
        for start in range(0, int(tokens.shape[1]), chunk_size):
            token_batch = tokens[:, start : start + chunk_size]
            state, stats = self._update_once(token_batch, state, differentiable=True)
            outputs.append(self.read(token_batch, state))
            statistics.append(stats)
        return (*state.tensors, torch.cat(outputs, dim=1), statistics)

    def write_then_read(
        self,
        x: torch.Tensor,
        state: TTTKVBState | None,
        differentiable: bool,
    ) -> tuple[TTTKVBState, torch.Tensor, list[dict[str, float]]]:
        """Paper-style causal TTT scan over every token: update, then emit."""

        if x.ndim != 3:
            raise ValueError(f"Expected tokens [B,L,D], got {tuple(x.shape)}")
        with torch.enable_grad():
            if state is None:
                state = self.initial_state(
                    batch_size=int(x.shape[0]),
                    device=x.device,
                    differentiable=differentiable,
                )
            chunk_size = self.inner_batch_size if self.inner_batch_size > 0 else int(x.shape[1])
            outputs = []
            statistics = []
            if differentiable and self.scan_checkpoint_updates:
                segment_size = chunk_size * self.scan_checkpoint_updates
                for start in range(0, int(x.shape[1]), segment_size):
                    result = checkpoint(
                        self._checkpointed_scan_segment,
                        x[:, start : start + segment_size],
                        *state.tensors,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                    state = TTTKVBState(*result[:4])
                    outputs.append(result[4])
                    statistics.extend(result[5])
                return state, torch.cat(outputs, dim=1), statistics
            for start in range(0, int(x.shape[1]), chunk_size):
                token_batch = x[:, start : start + chunk_size]
                state, stats = self._update_once(
                    token_batch,
                    state,
                    differentiable=differentiable,
                )
                outputs.append(self.read(token_batch, state))
                statistics.append(stats)
        return state, torch.cat(outputs, dim=1), statistics

    def read(self, x: torch.Tensor, state: TTTKVBState) -> torch.Tensor:
        """Read a query without changing the fast state."""

        if x.ndim != 3:
            raise ValueError(f"Expected query tokens [B,L,D], got {tuple(x.shape)}")
        if state.batch_size != int(x.shape[0]):
            raise ValueError(
                f"Fast-state batch={state.batch_size} does not match query batch={x.shape[0]}"
            )
        q = self._split_heads(self.q_proj(x).float())
        output = q + self._fast_residual(q, state)
        output = self._merge_heads(output)
        projected = self.out_proj(output.to(dtype=self.out_proj.weight.dtype))
        return projected.to(dtype=x.dtype)

    def outer_parameters(self) -> Iterable[nn.Parameter]:
        return self.parameters()
