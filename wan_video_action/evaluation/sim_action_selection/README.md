# Simulation action selection

```bash
python wan_video_action/evaluation/sim_action_selection/setup_simulator_dependencies.py
bash wan_video_action/evaluation/sim_action_selection/setup_model_environment.sh

python -m wan_video_action.evaluation.sim_action_selection \
  --task light_switch \
  --method ours \
  --checkpoint "$CHECKPOINT" \
  --context-table "$CONTEXT_TABLE" \
  --source-root "$SIM_SOURCES" \
  --model-python "$MODEL_PYTHON" \
  --simulator-python "$SIMULATOR_PYTHON" \
  --action-stats "$ACTION_STATS" \
  --output "$OUTPUT"
```

Methods: `ours`, `standard`, `dino`, `ttt`, `lora`.
Tasks: `friction`, `gravity`, `mass_collision`, `mass_friction`,
`mass_balance`, `multi_background`, `light_switch`.

Task configuration is in `tasks/<task>/config.json`. Model configuration is in
`configs/inference/`. The model and simulator run in separate worker processes.
Use `--dino-model-path` to locate DINO weights. Action normalization statistics
and model weights are required; prerecorded interaction or query datasets are not.

The runner collects independent simulator interactions, pauses at the configured
native decision boundary, predicts planned candidate actions, selects an action,
and executes it. Ours adapts Z, DINO aggregates observed supports, TTT replays
supports before each query update-then-read, and LoRA minimizes mean support loss.
Standard uses no adaptation. All methods receive native RGB and planned commands
through the same online interface. Selected-action observations are measured only
after selection. Observed support outcomes may also inform selection; disable
that behavior with `--no-use-observed-support-outcomes`.

Light Switch prepares each button with its original motion planner and pauses
during descent, with `press_lead_steps` planned commands remaining. This chooses
a template position before pressing, not an annotated event frame. Each candidate
retains its own paused native instance, which is used if selected. The exact event
offset is not prescribed; the template should remain within the training window
range. Physical case parameters can be provided with `--case` or `--cases`.

Mass x Friction encodes each candidate's first three native simulator frames
with JPEG quality 95 and H.264 CRF 18/GOP 2, then conditions the model on only the
encoded first frame. The preview must remain before collision and follow the
candidate's model actions. This is a bounded preimpact preview, not a strictly
current-observation-only acquisition protocol. The paused execution scene is not
advanced by this encoding preview.

Outputs contain resolved configuration, native observations, adaptation state,
candidate predictions, selection, executed rollout and measured results.

Dependency installation: [setup](docs/simulator_dependency_setup.md).
Observation encoding: [capture](../../../docs/simulation_observation_capture.md).
