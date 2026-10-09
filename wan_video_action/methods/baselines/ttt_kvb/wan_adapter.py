"""Non-destructive TTT-KVB attachment for DiffSynth Wan self-attention."""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType

import torch
from torch import nn

from .controller import TTTKVBController, TTTKVBMode
from .fast_weight import TTTMLPMemory


def _parse_layer_spec(spec: str, total_layers: int) -> tuple[int, ...]:
    normalized = str(spec).strip().lower()
    if normalized == "all":
        return tuple(range(total_layers))
    if normalized.startswith("last:"):
        count = int(normalized.split(":", 1)[1])
        return tuple(range(max(0, total_layers - count), total_layers))
    if normalized.startswith("uniform:"):
        count = int(normalized.split(":", 1)[1])
        if count <= 0:
            return tuple()
        if count >= total_layers:
            return tuple(range(total_layers))
        points = torch.linspace(0, total_layers - 1, steps=count).round().to(torch.long)
        return tuple(int(value) for value in points.unique(sorted=True).tolist())
    indices = tuple(int(value.strip()) for value in normalized.split(",") if value.strip())
    if any(index < 0 or index >= total_layers for index in indices):
        raise ValueError(f"Layer spec {spec!r} is outside [0, {total_layers})")
    return indices


def _ttt_kvb_attention_forward(self: nn.Module, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:


    base_output = self._ttt_kvb_original_forward(self, x, freqs)
    controller: TTTKVBController = self._ttt_kvb_controller
    if controller.mode == TTTKVBMode.DISABLED:
        return base_output
    memory_input = base_output if self._ttt_kvb_serial_after_attention else x
    if controller.mode == TTTKVBMode.CAUSAL_SCAN:
        memory_output = controller.scan(
            self._ttt_kvb_layer_id, self.ttt_kvb_memory, memory_input
        )
        gate = torch.tanh(self.ttt_kvb_gate).to(base_output.dtype)
        if gate.ndim == 1:
            gate = gate.view(1, 1, -1)
        return base_output + gate * memory_output
    if controller.mode == TTTKVBMode.SUPPORT_WRITE:


        controller.write(self._ttt_kvb_layer_id, self.ttt_kvb_memory, memory_input)
        return base_output
    if controller.mode == TTTKVBMode.QUERY_READ:
        memory_output = controller.read(self._ttt_kvb_layer_id, self.ttt_kvb_memory, memory_input)
        gate = torch.tanh(self.ttt_kvb_gate).to(base_output.dtype)
        if gate.ndim == 1:
            gate = gate.view(1, 1, -1)
        return base_output + gate * memory_output
    raise RuntimeError(f"Unsupported TTT-KVB mode: {controller.mode}")


@dataclass(frozen=True)
class TTTKVBInstallation:
    controller: TTTKVBController
    layer_indices: tuple[int, ...]
    attention_modules: tuple[nn.Module, ...]

    def ttt_parameters(self):
        for module in self.attention_modules:
            yield module.ttt_kvb_gate
            yield from module.ttt_kvb_memory.parameters()


def install_ttt_kvb(
    dit: nn.Module,
    layer_spec: str = "uniform:8",
    expansion: int = 4,
    base_inner_lr: float = 0.1,
    inner_batch_size: int = 64,
    write_token_budget: int = 512,
    gate_init: float = 0.1,
    gate_vector: bool = False,
    serial_after_attention: bool = False,
    scan_checkpoint_updates: int = 0,
) -> TTTKVBInstallation:
    """Attach TTT branches after the Wan checkpoint has been loaded.

    Existing ``self_attn.q/k/v/o`` paths are unchanged.  Only new
    ``ttt_kvb_*`` state-dict entries are introduced.
    """

    blocks = getattr(dit, "blocks", None)
    if blocks is None:
        raise TypeError("Expected a Wan DiT with a .blocks ModuleList")
    indices = _parse_layer_spec(layer_spec, len(blocks))
    controller = TTTKVBController()
    installed = []

    for index in indices:
        attention = blocks[index].self_attn
        if hasattr(attention, "ttt_kvb_memory"):
            raise RuntimeError(f"Wan block {index} already has a TTT-KVB branch")
        dim = int(attention.dim)
        num_heads = int(attention.num_heads)
        memory = TTTMLPMemory(
            dim=dim,
            num_heads=num_heads,
            expansion=expansion,
            base_inner_lr=base_inner_lr,
            inner_batch_size=inner_batch_size,
            write_token_budget=write_token_budget,
            scan_checkpoint_updates=scan_checkpoint_updates,
        )
        reference = attention.q.weight
        memory.to(device=reference.device, dtype=reference.dtype)

        attention.add_module("ttt_kvb_memory", memory)
        gate_value = (
            torch.full((dim,), float(gate_init), device=reference.device, dtype=torch.float32)
            if gate_vector
            else torch.tensor(float(gate_init), device=reference.device, dtype=torch.float32)
        )
        attention.register_parameter(
            "ttt_kvb_gate",
            nn.Parameter(gate_value),
        )
        attention._ttt_kvb_original_forward = attention.__class__.forward
        attention._ttt_kvb_controller = controller
        attention._ttt_kvb_layer_id = f"block_{index:02d}"
        attention._ttt_kvb_serial_after_attention = bool(serial_after_attention)
        attention.forward = MethodType(_ttt_kvb_attention_forward, attention)
        installed.append(attention)

    return TTTKVBInstallation(
        controller=controller,
        layer_indices=indices,
        attention_modules=tuple(installed),
    )
