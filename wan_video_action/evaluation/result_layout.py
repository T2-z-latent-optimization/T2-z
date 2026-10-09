"""Canonical, collision-resistant layout for new evaluation artifacts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_ROOT = Path(
    "${T2Z_REPO_ROOT}/results"
)
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _component(value: str, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_COMPONENT.fullmatch(value):
        raise ValueError(
            f"{label} must match {_SAFE_COMPONENT.pattern!r}, got {value!r}."
        )
    return value


@dataclass(frozen=True)
class EvaluationResultLayout:
    root: Path
    benchmark: str
    evaluation_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", self.root.expanduser().resolve())
        _component(self.benchmark, "benchmark")
        _component(self.evaluation_id, "evaluation_id")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "EvaluationResultLayout":
        results = config.get("results", {})
        return cls(
            root=Path(results.get("root") or DEFAULT_RESULTS_ROOT),
            benchmark=results.get("benchmark"),
            evaluation_id=results.get("evaluation_id"),
        )

    @property
    def evaluation_root(self) -> Path:
        return self.root / self.benchmark / self.evaluation_id

    @property
    def protocol_dir(self) -> Path:
        return self.evaluation_root / "protocol"

    @property
    def methods_dir(self) -> Path:
        return self.evaluation_root / "methods"

    @property
    def comparisons_dir(self) -> Path:
        return self.evaluation_root / "comparisons"

    def comparison_metric_dir(self, metric_family: str) -> Path:
        return self.comparisons_dir / "metrics" / _component(
            metric_family, "metric_family"
        )

    def method_run_dir(
        self,
        *,
        method: str,
        checkpoint_tag: str,
        seed: int,
    ) -> Path:
        if seed < 0:
            raise ValueError("seed cannot be negative.")
        return (
            self.methods_dir
            / _component(method, "method")
            / _component(checkpoint_tag, "checkpoint_tag")
            / f"seed_{seed}"
        )

    def create_shared_directories(self) -> None:
        self.protocol_dir.mkdir(parents=True, exist_ok=True)
        self.comparisons_dir.joinpath("metrics").mkdir(
            parents=True, exist_ok=True
        )
        self.comparisons_dir.joinpath("tables").mkdir(
            parents=True, exist_ok=True
        )
        self.comparisons_dir.joinpath("plots").mkdir(
            parents=True, exist_ok=True
        )
        self.comparisons_dir.joinpath("videos").mkdir(
            parents=True, exist_ok=True
        )

    def create_method_run(
        self,
        *,
        method: str,
        checkpoint_tag: str,
        seed: int,
    ) -> Path:
        run_dir = self.method_run_dir(
            method=method,
            checkpoint_tag=checkpoint_tag,
            seed=seed,
        )
        for relative in (
            "predictions",
            "masks",
            "metrics/global",
            "metrics/object_centric",
            "visualizations",
        ):
            run_dir.joinpath(relative).mkdir(parents=True, exist_ok=True)
        return run_dir
