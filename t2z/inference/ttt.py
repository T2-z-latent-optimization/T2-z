"""Training-faithful TTT inference with grouped support trajectories."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from t2z.inference.standard import (
    _parse_sample_indices,
    _run_autoregressive,
    build_infer_dataset,
    prepare_sample_for_rollout,
)
from t2z.inference.ttt_model import build_training_faithful_model
from t2z.inference.ttt_stream import (
    _mean_state_norm,
    _prepare_support_model_inputs,
    _update_counts,
)
from t2z.training.ttt import add_ttt_kvb_config
from wan_video_action.parsers import add_general_config, merge_yaml_and_args
from wan_video_action.utils import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = add_ttt_kvb_config(
        add_general_config(
            argparse.ArgumentParser(
                "Training-faithful grouped-support TTT inference"
            )
        )
    )
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--skip_existing", action="store_true", default=False)
    parser.add_argument("--sample_indices", type=str, default=None)
    parser.add_argument("--support_indices", type=str, default=None)
    parser.add_argument("--ttt_checkpoint_path", type=str, default=None)
    parser.add_argument("--expected_supports_per_environment", type=int, default=0)
    parser.add_argument("--ttt_support_noise_seed_offset", type=int, default=1_000_003)
    args = parser.parse_args()
    if args.config is not None:
        args = merge_yaml_and_args(args.config, parser, args)
    for key in ("sample_indices", "support_indices", "ttt_checkpoint_path"):
        if not getattr(args, key):
            parser.error(f"{key} is required in the config or CLI")
    if args.ttt_protocol != "write_then_predict":
        raise ValueError(
            "Grouped inference requires ttt_protocol=write_then_predict, "
            f"got {args.ttt_protocol!r}."
        )
    return args


def _environment_key(value: Any) -> str:
    if isinstance(value, float):
        return format(value, ".12g")
    return str(value)


def _environment_seed(value: Any) -> int:
    digest = hashlib.sha256(_environment_key(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


class GroupedSameTimestepSupportQueryModelFn:
    """Replay all supports, then run one independent query at each denoising step."""

    def __init__(
        self,
        *,
        pipe,
        controller,
        original_model_fn,
        support_replays: list[dict[str, Any]],
    ) -> None:
        self.pipe = pipe
        self.controller = controller
        self.original_model_fn = original_model_fn
        self.support_replays = support_replays
        self.trace: list[dict[str, Any]] = []

    def _support_call_inputs(
        self, replay: dict[str, Any], timestep: torch.Tensor
    ) -> dict[str, Any]:
        support_inputs = dict(replay["model_inputs"])
        support_latents = self.pipe.scheduler.add_noise(
            replay["input_latents"], replay["noise"], timestep
        )
        first_frame_latents = support_inputs.get("first_frame_latents")
        if isinstance(first_frame_latents, torch.Tensor):
            support_latents = support_latents.clone()
            support_latents[:, :, 0:1] = first_frame_latents[:, :, 0:1]
        support_inputs["latents"] = support_latents
        support_inputs["noise"] = replay["noise"]
        support_inputs["timestep"] = timestep
        support_inputs["use_gradient_checkpointing"] = False
        support_inputs["use_gradient_checkpointing_offload"] = False
        for model_name in self.pipe.in_iteration_models:
            support_inputs[model_name] = getattr(self.pipe, model_name)
        return support_inputs

    def __call__(self, *args, **query_inputs):
        timestep = query_inputs.get("timestep")
        if not isinstance(timestep, torch.Tensor):
            raise RuntimeError("Wan model_fn call did not provide a tensor timestep.")

        self.controller.reset(batch_size=1)
        try:
            with self.controller.causal_scan(differentiable=False):
                for replay in self.support_replays:
                    self.original_model_fn(
                        **self._support_call_inputs(replay, timestep)
                    )
            support_statistics = self.controller.write_statistics()
            support_counts = _update_counts(support_statistics)
            support_norms = self.controller.state_norms()

            with self.controller.causal_scan(differentiable=False):
                prediction = self.original_model_fn(*args, **query_inputs)
            complete_statistics = self.controller.write_statistics()
            complete_counts = _update_counts(complete_statistics)
            query_counts = {
                layer_id: complete_counts.get(layer_id, 0)
                - support_counts.get(layer_id, 0)
                for layer_id in complete_counts
            }
            query_norms = self.controller.state_norms()
            self.trace.append(
                {
                    "denoising_call": len(self.trace),
                    "timestep": float(
                        timestep.detach().float().reshape(-1)[0].cpu()
                    ),
                    "support_count": len(self.support_replays),
                    "support_updates_per_layer": support_counts,
                    "query_updates_per_layer": query_counts,
                    "support_state_norm_mean": _mean_state_norm(support_norms),
                    "query_state_norm_mean": _mean_state_norm(query_norms),
                }
            )
            return prediction
        finally:
            self.controller.clear()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    os.makedirs(args.output_path, exist_ok=True)
    dataset = build_infer_dataset(args)
    model, installation = build_training_faithful_model(args)
    pipe = model.pipe
    controller = installation.controller
    original_model_fn = pipe.model_fn

    support_by_environment: dict[str, list[int]] = defaultdict(list)
    environment_values: dict[str, Any] = {}
    for support_index in _parse_sample_indices(args.support_indices):
        value = dataset[support_index][args.ttt_environment_key]
        key = _environment_key(value)
        support_by_environment[key].append(int(support_index))
        environment_values[key] = value

    queries_by_environment: dict[str, list[int]] = defaultdict(list)
    for sample_index in _parse_sample_indices(args.sample_indices):
        value = dataset[sample_index][args.ttt_environment_key]
        key = _environment_key(value)
        queries_by_environment[key].append(int(sample_index))
        environment_values[key] = value

    expected = int(args.expected_supports_per_environment)
    for key in queries_by_environment:
        count = len(support_by_environment.get(key, ()))
        if count == 0:
            raise KeyError(f"No TTT support for environment {key}.")

    records: list[dict[str, Any]] = []
    for key, query_indices in queries_by_environment.items():
        value = environment_values[key]
        support_indices = support_by_environment[key]
        support_replays: list[dict[str, Any]] = []
        for ordinal, support_index in enumerate(support_indices):
            noise_seed = (
                int(args.seed)
                + int(args.ttt_support_noise_seed_offset)
                + _environment_seed(value)
                + ordinal * 104_729
            ) % (2**63 - 1)
            model_inputs, input_latents, support_noise = (
                _prepare_support_model_inputs(
                    model,
                    dataset[support_index],
                    noise_seed=noise_seed,
                )
            )
            support_replays.append(
                {
                    "support_index": support_index,
                    "noise_seed": noise_seed,
                    "model_inputs": model_inputs,
                    "input_latents": input_latents,
                    "noise": support_noise,
                }
            )
        print(
            f"[supports_ready] environment={key} supports={support_indices}",
            flush=True,
        )

        for sample_index in query_indices:
            sample = prepare_sample_for_rollout(
                dataset[sample_index], sample_index, pipe, args
            )
            if args.skip_existing and Path(sample["output_path"]).is_file():
                print(f"[skip] existing prediction {sample['output_path']}")
                continue

            wrapped_model_fn = GroupedSameTimestepSupportQueryModelFn(
                pipe=pipe,
                controller=controller,
                original_model_fn=original_model_fn,
                support_replays=support_replays,
            )
            pipe.model_fn = wrapped_model_fn
            try:
                predicted_path = _run_autoregressive(
                    pipe=pipe,
                    sample=sample,
                    args=args,
                )
            finally:
                pipe.model_fn = original_model_fn
                controller.clear()

            records.append(
                {
                    "environment_id": key,
                    "support_indices": support_indices,
                    "query_index": sample_index,
                    "support_noise_seeds": [
                        replay["noise_seed"] for replay in support_replays
                    ],
                    "prediction": str(predicted_path),
                    "denoising_trace": wrapped_model_fn.trace,
                }
            )
            print(
                f"[done] sample_index={sample_index} environment={key} "
                f"supports={len(support_indices)} "
                f"denoising_calls={len(wrapped_model_fn.trace)} "
                f"output={predicted_path}",
                flush=True,
            )

        del support_replays
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    protocol = {
        "method": "ttt",
        "implementation": "same_timestep_grouped_support_query_branch",
        "support_query_disjoint": True,
        "expected_supports_per_environment": expected,
        "diffusion_timestep_policy": (
            "all_supports_and_query_share_each_denoising_timestep"
        ),
        "fast_state_lifecycle": (
            "reset_to_learned_initial_then_ordered_support_scan_then_query_scan_"
            "per_timestep"
        ),
        "query_policy": "update_then_read",
        "cross_query_state": "independent_branch_from_same_support_set",
        "cross_denoising_timestep_state": "reset_and_rebuild",
        "outer_loss_at_inference": False,
        "checkpoint": args.ttt_checkpoint_path,
        "ttt_layers": list(installation.layer_indices),
        "records": records,
    }
    protocol_path = Path(args.output_path) / "protocol.json"
    temporary_path = protocol_path.with_suffix(".json.partial")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2)
    temporary_path.replace(protocol_path)
    print(f"[protocol] {protocol_path}", flush=True)


if __name__ == "__main__":
    main()
