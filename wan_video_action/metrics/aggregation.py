"""Hierarchical query -> environment -> seed macro aggregation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np


def _mean_rows(
    rows: Iterable[dict[str, Any]],
    group_keys: Sequence[str],
    metric_names: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)

    result: list[dict[str, Any]] = []
    for key, members in groups.items():
        output = dict(zip(group_keys, key))
        output["count"] = len(members)
        for metric in metric_names:
            values = [
                float(member[metric])
                for member in members
                if metric in member and member[metric] is not None
            ]
            output[metric] = float(np.mean(values)) if values else None
        result.append(output)
    result.sort(key=lambda row: tuple(str(row[key]) for key in group_keys))
    return result


def aggregate_query_metrics(
    query_rows: list[dict[str, Any]],
    metric_names: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    environment_keys = (
        "method",
        "split",
        "support_size",
        "seed",
        "environment_id",
    )
    environment_rows = _mean_rows(query_rows, environment_keys, metric_names)
    seed_keys = ("method", "split", "support_size", "seed")
    seed_rows = _mean_rows(environment_rows, seed_keys, metric_names)
    summary_keys = ("method", "split", "support_size")
    summary_rows = _mean_rows(seed_rows, summary_keys, metric_names)
    return {
        "environment": environment_rows,
        "seed": seed_rows,
        "summary": summary_rows,
    }
