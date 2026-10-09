from ..protocol import AdaptationTarget, MethodFamily, MethodSpec, QueryStatePolicy


SPEC = MethodSpec(
    slug="ours",
    display_name="Ours",
    family=MethodFamily.OURS,
    summary="Persistent environment memory with alternating assimilation and consolidation.",
    training_protocol=(
        "Assign a persistent Z to each latent environment index and alternate "
        "environment-code optimization with shared Wan model updates over the "
        "progressive environment stream."
    ),
    inference_protocol=(
        "Initialize Z using the configured precision and table initialization, optimize only Z "
        "on K support trajectories, freeze it, and reuse it for every disjoint query."
    ),
    adaptation_target=AdaptationTarget.ENVIRONMENT_CODE,
    query_state_policy=QueryStatePolicy.READ_ONLY,
    requires_grouped_training=True,
    invariants=(
        "A physical environment may have multiple independently trained latent entries.",
        "Support and query trajectories are disjoint.",
        "Query prediction does not update Z.",
    ),
)
