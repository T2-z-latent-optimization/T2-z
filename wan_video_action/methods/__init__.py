"""Method contracts and protocol definitions for controlled comparisons."""

from .protocol import (
    AdaptationTarget,
    EnvironmentGroup,
    GroupedTrainingBatch,
    MethodFamily,
    MethodRunner,
    MethodSpec,
    QueryStatePolicy,
    SupportQueryEpisode,
)
from .registry import BUILTIN_METHODS, get_method_spec, list_method_specs

__all__ = [
    "AdaptationTarget",
    "BUILTIN_METHODS",
    "EnvironmentGroup",
    "GroupedTrainingBatch",
    "MethodFamily",
    "MethodRunner",
    "MethodSpec",
    "QueryStatePolicy",
    "SupportQueryEpisode",
    "get_method_spec",
    "list_method_specs",
]
