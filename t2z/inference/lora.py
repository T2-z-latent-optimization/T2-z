"""LoRA adaptation from a frozen Standard checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from t2z.inference.standard import (
    _parse_sample_indices,
    _run_autoregressive,
    build_infer_dataset,
    build_pipeline,
    prepare_sample_for_rollout,
)
from t2z.inference.ours import _flow_match_loss, _freeze_pipe, _prepare_loss_inputs
from wan_video_action.parsers import add_general_config, merge_yaml_and_args


class LoRALinear(nn.Module):
    """Frozen Linear plus an FP32 low-rank update with zero initial output."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, generator: torch.Generator):
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.lora_a = nn.Parameter(torch.empty(
            self.rank, base.in_features, device=base.weight.device, dtype=torch.float32
        ))
        self.lora_b = nn.Parameter(torch.zeros(
            base.out_features, self.rank, device=base.weight.device, dtype=torch.float32
        ))
        with torch.no_grad():
            self.lora_a.normal_(mean=0.0, std=1.0 / math.sqrt(base.in_features), generator=generator)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("initial_a", self.lora_a.detach().clone(), persistent=False)
        self.register_buffer("initial_b", self.lora_b.detach().clone(), persistent=False)

    def reset_adapter(self) -> None:
        with torch.no_grad():
            self.lora_a.copy_(self.initial_a)
            self.lora_b.copy_(self.initial_b)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        adapter_input = F.linear(inputs, self.lora_a.to(dtype=inputs.dtype))
        adapter_output = F.linear(adapter_input, self.lora_b.to(dtype=inputs.dtype))
        return base_output + adapter_output.to(dtype=base_output.dtype) * self.scale


def parse_args() -> argparse.Namespace:
    parser = add_general_config(argparse.ArgumentParser())
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--support-indices", default=None)
    parser.add_argument("--support-environment-key", default="environment_index")
    parser.add_argument("--sample-indices", default=None)
    parser.add_argument("--lora-steps", type=int, default=50)
    parser.add_argument("--lora-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lora-gradient-clip", type=float, default=1.0)
    parser.add_argument("--adapter-seed", type=int, default=20260708)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if args.config is not None:
        args = merge_yaml_and_args(args.config, parser, args)
    for key in ("support_indices", "sample_indices"):
        if not getattr(args, key):
            parser.error(f"{key} is required in the config or CLI")
    if int(args.frame_stride) < 1:
        raise ValueError("frame_stride must be positive")
    if int(args.lora_rank) < 1 or not args.lora_target_modules:
        raise ValueError("LoRA requires a positive rank and nonempty target modules")
    args.stage2_fixed_timestep_index = None
    return args


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)




def install_lora(
    dit: nn.Module,
    *,
    targets: set[str],
    rank: int,
    alpha: float,
    seed: int,
) -> list[tuple[str, LoRALinear]]:
    generator = torch.Generator(device=next(dit.parameters()).device)
    generator.manual_seed(int(seed))
    selected: list[tuple[str, nn.Linear]] = []
    for name, module in list(dit.named_modules()):
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in targets:
            selected.append((name, module))
    adapters: list[tuple[str, LoRALinear]] = []
    for name, module in selected:
        parent_name, child_name = name.rsplit(".", 1)
        parent = dit.get_submodule(parent_name)
        wrapped = LoRALinear(module, rank=rank, alpha=alpha, generator=generator)
        setattr(parent, child_name, wrapped)
        adapters.append((name, wrapped))
    if not adapters:
        raise RuntimeError(f"No DiT Linear modules matched LoRA targets {sorted(targets)}.")
    return adapters


def reset_lora(adapters: list[tuple[str, LoRALinear]]) -> list[nn.Parameter]:
    parameters: list[nn.Parameter] = []
    for _, adapter in adapters:
        adapter.reset_adapter()
        parameters.extend((adapter.lora_a, adapter.lora_b))
    return parameters


def adapt_lora(
    pipe,
    support_items: list[dict],
    adapters: list[tuple[str, LoRALinear]],
    args,
) -> list[float]:
    parameters = reset_lora(adapters)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(args.lora_learning_rate),
        weight_decay=0.0,
    )
    pipe.scheduler.set_timesteps(1000, training=True)
    empty_context = torch.empty(0, device=pipe.device, dtype=pipe.torch_dtype)
    losses: list[float] = []
    for step in range(int(args.lora_steps)):
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for support in support_items:
            inputs = _prepare_loss_inputs(pipe, support, empty_context, args)
            loss = _flow_match_loss(pipe, inputs, args) / float(len(support_items))
            loss.backward()
            loss_value += float(loss.detach().float().cpu())
        if float(args.lora_gradient_clip) > 0:
            torch.nn.utils.clip_grad_norm_(parameters, float(args.lora_gradient_clip))
        optimizer.step()
        losses.append(loss_value)
        if step == 0 or (step + 1) % 10 == 0:
            print(
                f"[lora_inner] step={step + 1}/{args.lora_steps} "
                f"support_count={len(support_items)} loss={loss_value:.6f}",
                flush=True,
            )
    return losses


def main() -> None:
    args = parse_args()
    query_indices = list(_parse_sample_indices(args.sample_indices) or [])
    from t2z.inference.selection import group_samples, parse_indices
    groups = group_samples(read_jsonl(args.dataset_metadata_path), parse_indices(args.support_indices), query_indices, args.support_environment_key)

    dataset = build_infer_dataset(args)
    pipe = build_pipeline(args)
    _freeze_pipe(pipe)
    targets = {
        item.strip() for item in str(args.lora_target_modules).split(",") if item.strip()
    }
    adapters = install_lora(
        pipe.dit,
        targets=targets,
        rank=int(args.lora_rank),
        alpha=float(args.lora_alpha),
        seed=int(args.adapter_seed),
    )
    adapter_count = sum(
        adapter.lora_a.numel() + adapter.lora_b.numel() for _, adapter in adapters
    )
    print(
        f"[lora] modules={len(adapters)} params={adapter_count} rank={args.lora_rank} "
        f"targets={sorted(targets)}",
        flush=True,
    )

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    result_path = output_path.parent / "results.jsonl"
    rows: list[dict] = []
    for group in groups:
        supports = group["support_indices"]
        pending_queries = []
        for query_index in group["query_indices"]:
            query_row = dataset.data[int(query_index)]
            prediction = output_path / (
                f"sample{query_index:04d}_episode{int(query_row['episode_index']):06d}_"
                f"frames{int(query_row['start_frame']):04d}-{int(query_row['end_frame']):04d}.mp4"
            )
            if not (prediction.exists() and args.skip_existing):
                pending_queries.append((query_index, query_row, prediction))
        if not pending_queries:
            print(f"[skip_group] environment={group['environment_id']}", flush=True)
            continue
        print(
            f"[adapt] environment={group['environment_id']} support={supports} "
            f"queries={group['query_indices']}",
            flush=True,
        )
        support_items = [dataset[index] for index in supports]
        losses = adapt_lora(pipe, support_items, adapters, args)
        pipe.eval()
        for query_index, query_row, prediction in pending_queries:
            query = prepare_sample_for_rollout(dataset[int(query_index)], int(query_index), pipe, args)
            query["output_path"] = str(prediction)
            with torch.no_grad():
                _run_autoregressive(pipe=pipe, sample=query, args=args)
            rows.append({
                "sample_index": int(query_index),
                "sample_id": query_row.get("sample_id"),
                "episode_index": int(query_row["episode_index"]),
                "environment_id": group["environment_id"],
                "friction_mu": (
                    float(query_row["friction_mu"])
                    if "friction_mu" in query_row else None
                ),
                "support_indices": supports,
                "adaptation_reused_across_queries": True,
                "lora_steps": int(args.lora_steps),
                "lora_rank": int(args.lora_rank),
                "lora_learning_rate": float(args.lora_learning_rate),
                "lora_param_count": int(adapter_count),
                "inner_losses": losses,
                "prediction_path": str(prediction),
            })
            write_jsonl(result_path, rows)
            torch.cuda.empty_cache()
    print(f"[done] predictions={output_path} count={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
