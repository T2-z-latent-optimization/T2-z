"""Registry for method protocol definitions."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from .protocol import MethodFamily, MethodSpec


class MethodRegistry:
    def __init__(self, specs: Iterable[MethodSpec] = ()) -> None:
        self._specs: dict[str, MethodSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: MethodSpec) -> None:
        if spec.slug in self._specs:
            raise ValueError(f"Method already registered: {spec.slug}")
        self._specs[spec.slug] = spec

    def get(self, slug: str) -> MethodSpec:
        try:
            return self._specs[slug]
        except KeyError as exc:
            available = ", ".join(self._specs)
            raise KeyError(f"Unknown method '{slug}'. Available: {available}") from exc

    def all(self, family: MethodFamily | None = None) -> tuple[MethodSpec, ...]:
        specs = tuple(self._specs.values())
        if family is None:
            return specs
        return tuple(spec for spec in specs if spec.family is family)

    def __contains__(self, slug: object) -> bool:
        return slug in self._specs

    def __iter__(self) -> Iterator[MethodSpec]:
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)


from .baselines import SPECS as BASELINE_SPECS
from .ours import SPECS as OUR_SPECS


BUILTIN_METHODS = MethodRegistry((*BASELINE_SPECS, *OUR_SPECS))


def get_method_spec(slug: str) -> MethodSpec:
    return BUILTIN_METHODS.get(slug)


def list_method_specs(family: MethodFamily | None = None) -> tuple[MethodSpec, ...]:
    return BUILTIN_METHODS.all(family)
