from ..protocol import AdaptationTarget, MethodFamily, MethodSpec, QueryStatePolicy


SPEC = MethodSpec(
    slug="standard",
    display_name="Standard",
    family=MethodFamily.BASELINE,
    summary="Normally trained pooled world model with no environment code or adaptation.",
    training_protocol=(
        "Sample environment batches according to the task configuration and train "
        "the world model with flow-matching loss, without an environment-code "
        "branch or alternating latent/model schedule."
    ),
    inference_protocol=(
        "Freeze the trained model, ignore the support set, and generate every disjoint "
        "query directly without test-time adaptation."
    ),
    adaptation_target=AdaptationTarget.NONE,
    query_state_policy=QueryStatePolicy.NONE,
    requires_grouped_training=False,
    invariants=(
        "Held-out environments remain excluded from pooled training.",
        "Frozen refers only to evaluation; Wan is normally optimized during training.",
    ),
)
