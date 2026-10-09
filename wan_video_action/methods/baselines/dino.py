from ..protocol import AdaptationTarget, MethodFamily, MethodSpec, QueryStatePolicy


SPEC = MethodSpec(
    slug="dino",
    display_name="DINO",
    family=MethodFamily.BASELINE,
    summary="Frozen DINOv2 features and action features mapped to Z by a concat MLP.",
    training_protocol=(
        "Sample support K using the task configuration, keep the existing "
        "support selection, and optimize mean flow-matching loss over the entire "
        "sampled environment batch, including its support chunks."
    ),
    inference_protocol=(
        "Encode the configured support set, aggregate its codes, and reuse the "
        "resulting Z for held-out queries without query-time adaptation."
    ),
    adaptation_target=AdaptationTarget.AMORTIZED_CODE,
    query_state_policy=QueryStatePolicy.READ_ONLY,
    requires_grouped_training=True,
    invariants=(
        "DINOv2 stays frozen; the context head and world model are trained.",
        "Evaluation queries do not enter the support encoder.",
    ),
)
