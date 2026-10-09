from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
PUSH_BOX_FRICTION_CONTEXT_MU_MIN = 0.0
PUSH_BOX_FRICTION_CONTEXT_MU_MAX = 0.25

def normalize_push_box_friction_mu(mu: float) -> float:
    """
    Fixed oracle-C normalization for push-box friction experiments.

    The physical context C is one scalar. We map the expected coefficient range
    [0, 0.25] to [0, 1].
    """
    mu_value = float(mu)
    if mu_value < 0:
        raise ValueError(f'friction_mu must be non-negative, got {mu_value}.')
    return (mu_value - PUSH_BOX_FRICTION_CONTEXT_MU_MIN) / (PUSH_BOX_FRICTION_CONTEXT_MU_MAX - PUSH_BOX_FRICTION_CONTEXT_MU_MIN)

@dataclass(frozen=True)
class PhysicalContextConfig:
    context_dim: int = 128
    model_dim: int = 1536
    num_tokens: int = 1
    hidden_dim: Optional[int] = None
    init_std: float = 0.0
    init_value: float = 0.0
    input_norm: str = 'layernorm'
    projection: str = 'mlp'
    temporal_position: str = 'none'

class BiasFreeMLP(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, input_norm: str='layernorm'):
        super().__init__()
        use_input_norm = str(input_norm).lower() != 'none' and int(in_dim) != 1
        self.net = nn.Sequential(nn.LayerNorm(in_dim) if use_input_norm else nn.Identity(), nn.Linear(in_dim, hidden_dim, bias=False), nn.GELU(approximate='tanh'), nn.Linear(hidden_dim, out_dim, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class PhysicalContextEncoder(nn.Module):
    """
    Project a latent physical-property code C into BWM conditioning channels.

    The default context starts at zero, so enabling the module is initially close
    to the base action-conditioned model. The projection weights are non-zero,
    which keeps gradients flowing into both the default C and the projectors.
    """

    def __init__(self, context_dim: int=128, model_dim: int=1536, num_tokens: int=1, hidden_dim: Optional[int]=None, init_std: float=0.0, init_value: float=0.0, input_norm: str='layernorm', projection: str='mlp', temporal_position: str='none'):
        super().__init__()
        if context_dim <= 0:
            raise ValueError(f'context_dim must be positive, got {context_dim}.')
        if model_dim <= 0:
            raise ValueError(f'model_dim must be positive, got {model_dim}.')
        if num_tokens <= 0:
            raise ValueError(f'num_tokens must be positive, got {num_tokens}.')
        input_norm = str(input_norm).lower()
        if input_norm not in {'layernorm', 'none'}:
            raise ValueError(f'input_norm must be layernorm or none, got {input_norm!r}.')
        projection = str(projection).lower()
        if projection != 'mlp':
            raise ValueError(f'Unsupported context projection: {projection!r}')
        temporal_position = str(temporal_position).lower()
        if temporal_position not in {'none', 'learned'}:
            raise ValueError(f'temporal_position must be none or learned, got {temporal_position!r}.')
        hidden_dim = int(hidden_dim or min(model_dim, max(context_dim * 4, 128)))
        self.config = PhysicalContextConfig(context_dim=int(context_dim), model_dim=int(model_dim), num_tokens=int(num_tokens), hidden_dim=hidden_dim, init_std=float(init_std), init_value=float(init_value), input_norm=input_norm, projection=projection, temporal_position=temporal_position)
        self.default_context = nn.Parameter(torch.full((num_tokens, context_dim), float(init_value)))
        if init_std > 0:
            nn.init.normal_(self.default_context, mean=float(init_value), std=init_std)
        self.token_projector = BiasFreeMLP(context_dim, hidden_dim, model_dim, input_norm=input_norm)
        self.mod_projector = BiasFreeMLP(context_dim, hidden_dim, model_dim, input_norm=input_norm)
        if temporal_position == 'learned' and num_tokens > 1:
            self.token_position_embedding = nn.Parameter(torch.zeros(num_tokens, model_dim))
            self.mod_position_embedding = nn.Parameter(torch.zeros(num_tokens, model_dim))
            nn.init.normal_(self.token_position_embedding, mean=0.0, std=0.02)
            nn.init.normal_(self.mod_position_embedding, mean=0.0, std=0.02)
        else:
            self.register_parameter('token_position_embedding', None)
            self.register_parameter('mod_position_embedding', None)

    @property
    def context_dim(self) -> int:
        return self.config.context_dim

    @property
    def model_dim(self) -> int:
        return self.config.model_dim

    @property
    def num_tokens(self) -> int:
        return self.config.num_tokens

    def _normalize_context(self, physical_context: Optional[torch.Tensor], *, batch_size: int, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        if physical_context is None:
            context = self.default_context.unsqueeze(0).expand(batch_size, -1, -1)
            return context.to(device=device, dtype=dtype)
        context = torch.as_tensor(physical_context, device=device, dtype=dtype)
        if context.ndim == 1:
            context = context.view(1, 1, -1)
        elif context.ndim == 2:
            if context.shape[-1] != self.context_dim:
                raise ValueError(f'Expected physical_context last dim {self.context_dim}, got shape={tuple(context.shape)}.')
            if context.shape[0] == batch_size:
                context = context.unsqueeze(1)
            elif context.shape[0] == self.num_tokens or batch_size == 1:
                context = context.unsqueeze(0)
            else:
                raise ValueError(f'Ambiguous 2D physical_context. Use shape (B, C), (tokens, C), or (B, tokens, C); got shape={tuple(context.shape)} with batch_size={batch_size}.')
        elif context.ndim != 3:
            raise ValueError(f'physical_context must have shape (C,), (tokens, C), or (B, tokens, C); got shape={tuple(context.shape)}.')
        if context.shape[-1] != self.context_dim:
            raise ValueError(f'Expected physical_context last dim {self.context_dim}, got shape={tuple(context.shape)}.')
        if context.shape[1] != self.num_tokens:
            if context.shape[1] == 1:
                context = context.expand(-1, self.num_tokens, -1)
            else:
                raise ValueError(f'Expected {self.num_tokens} physical context tokens, got {context.shape[1]}.')
        if context.shape[0] == 1 and batch_size != 1:
            context = context.expand(batch_size, -1, -1)
        elif context.shape[0] != batch_size:
            raise ValueError(f'physical_context batch size {context.shape[0]} does not match expected batch_size={batch_size}.')
        return context

    def forward(self, physical_context: Optional[torch.Tensor]=None, *, batch_size: int=1, target_groups: int=1, dtype: Optional[torch.dtype]=None, device: Optional[torch.device | str]=None) -> tuple[torch.Tensor, torch.Tensor]:
        if target_groups <= 0:
            raise ValueError(f'target_groups must be positive, got {target_groups}.')
        dtype = dtype or self.default_context.dtype
        device = device or self.default_context.device
        context = self._normalize_context(physical_context, batch_size=int(batch_size), dtype=dtype, device=device)
        token_emb = self.token_projector(context)
        if self.token_position_embedding is not None and token_emb.shape[1] == self.token_position_embedding.shape[0]:
            token_emb = token_emb + self.token_position_embedding.to(dtype=token_emb.dtype, device=token_emb.device).unsqueeze(0)
        if context.shape[1] == int(target_groups):
            mod_emb = self.mod_projector(context)
            if self.mod_position_embedding is not None and mod_emb.shape[1] == self.mod_position_embedding.shape[0]:
                mod_emb = mod_emb + self.mod_position_embedding.to(dtype=mod_emb.dtype, device=mod_emb.device).unsqueeze(0)
        else:
            pooled_context = context.mean(dim=1)
            mod_emb = self.mod_projector(pooled_context).unsqueeze(1).expand(-1, int(target_groups), -1)
        return (token_emb, mod_emb)

class LowRankResidualAdapter(nn.Module):

    def __init__(self, dim: int, rank: int=16, gate_init: float=0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f'rank must be positive, got {rank}.')
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.cond_norm = nn.LayerNorm(dim)
        self.cond_gate = nn.Linear(dim, 1, bias=False)
        nn.init.zeros_(self.cond_gate.weight)

    def forward(self, x: torch.Tensor, conditioning: Optional[torch.Tensor]=None) -> torch.Tensor:
        gate = self.gate
        if conditioning is not None:
            if conditioning.ndim == 3:
                conditioning = conditioning.mean(dim=1)
            if conditioning.ndim != 2 or conditioning.shape[-1] != x.shape[-1]:
                raise ValueError(f'Adapter conditioning must have shape (B, D) or (B, T, D); got conditioning={tuple(conditioning.shape)}, x={tuple(x.shape)}.')
            cond_gate = self.cond_gate(self.cond_norm(conditioning)).to(dtype=x.dtype)
            gate = gate.to(dtype=x.dtype) + cond_gate.view(cond_gate.shape[0], 1, 1)
        return x + gate * self.up(self.down(self.norm(x)))

class PhysicalResidualAdapterBank(nn.Module):
    """
    Lightweight trainable residual adapters for selected DiT blocks.

    This is deliberately independent from the token-injection C path: it supports
    the alternative experiment where test-time adaptation directly updates a
    small parameter subset while the Wan backbone stays frozen. It is a
    conservative parameter-adaptation baseline, not a direct implementation of
    TTT-E2E's updated-MLP design.
    """

    def __init__(self, dim: int, num_layers: int, rank: int=16, layers: Optional[list[int]]=None, gate_init: float=0.0):
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f'num_layers must be positive, got {num_layers}.')
        selected = list(range(num_layers)) if layers is None else sorted(set((int(layer) for layer in layers)))
        invalid = [layer for layer in selected if layer < 0 or layer >= num_layers]
        if invalid:
            raise ValueError(f'Adapter layer indices out of range for num_layers={num_layers}: {invalid}.')
        self.num_layers = int(num_layers)
        self.selected_layers = tuple(selected)
        self.adapters = nn.ModuleDict({str(layer): LowRankResidualAdapter(dim=dim, rank=rank, gate_init=gate_init) for layer in selected})

    def forward(self, layer_index: int, x: torch.Tensor, conditioning: Optional[torch.Tensor]=None) -> torch.Tensor:
        key = str(int(layer_index))
        if key not in self.adapters:
            return x
        return self.adapters[key](x, conditioning=conditioning)
