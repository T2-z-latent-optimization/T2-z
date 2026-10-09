"""Shared deterministic environment sampling helpers for method baselines."""

from __future__ import annotations

import random
from typing import Mapping, Sequence, TypeVar


T = TypeVar("T")


def weighted_sample_without_replacement(
    rng: random.Random,
    population: Sequence[T],
    count: int,
    weights: Mapping[T, float] | None = None,
) -> list[T]:
    """Sample unique environments with optional sampling weights."""

    values = list(population)
    if count < 0 or count > len(values):
        raise ValueError(
            f"Cannot sample {count} unique values from a population of {len(values)}."
        )
    resolved = [float(weights.get(value, 1.0)) if weights else 1.0 for value in values]
    if any(weight <= 0.0 for weight in resolved):
        raise ValueError(f"Environment sampling weights must be positive: {resolved}.")
    if len(set(resolved)) == 1:
        return rng.sample(values, count)
    ranked = sorted(
        (
            (rng.random() ** (1.0 / weight), index, value)
            for index, (value, weight) in enumerate(zip(values, resolved))
        ),
        reverse=True,
    )
    return [value for _, _, value in ranked[:count]]


def sample_slots(rng, population, count):
    values = list(population)
    if count <= len(values):
        return rng.sample(values, count)
    return rng.choices(values, k=count)


def sample_environment_slots(rng, population, count, weights=None):
    values = list(population)
    if count <= len(values):
        return weighted_sample_without_replacement(rng, values, count, weights)
    resolved = [float(weights.get(value, 1.0)) if weights else 1.0 for value in values]
    return rng.choices(values, weights=resolved, k=count)
