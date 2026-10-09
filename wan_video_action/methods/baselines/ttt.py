from ..protocol import AdaptationTarget, MethodFamily, MethodSpec, QueryStatePolicy


SPEC = MethodSpec(
    slug="ttt",
    display_name="TTT",
    family=MethodFamily.BASELINE,
    summary="Causal attention fast weights with update-then-read memory.",
    training_protocol=(
        "Scan environment chunks at a shared diffusion timestep. Update fast "
        "weights from each causal token block before reading them, and "
        "backpropagate flow-matching loss through the differentiable stream."
    ),
    inference_protocol=(
        "For each query and denoising timestep, reset to learned initial fast "
        "weights, replay the ordered supports at that timestep, then scan the "
        "query with update-then-read. Discard this branch after prediction."
    ),
    adaptation_target=AdaptationTarget.FAST_WEIGHTS,
    query_state_policy=QueryStatePolicy.UPDATE_THEN_READ,
    requires_grouped_training=True,
    invariants=(
        "Query writes use its current model tokens, not its ground-truth future.",
        "Temporal positions reset at trajectory boundaries.",
        "Fast state is not shared across query branches or denoising timesteps.",
    ),
)
