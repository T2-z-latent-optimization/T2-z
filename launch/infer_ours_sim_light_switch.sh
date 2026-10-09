#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${T2Z_CHECKPOINT:?Set T2Z_CHECKPOINT}"
: "${T2Z_CONTEXT_TABLE:?Set T2Z_CONTEXT_TABLE}"
: "${T2Z_INFER_OUTPUT:?Set T2Z_INFER_OUTPUT}"
python t2z/infer.py --method ours --config configs/inference/sim_light_switch.yaml --checkpoint "$T2Z_CHECKPOINT" --context-table "$T2Z_CONTEXT_TABLE" --output "$T2Z_INFER_OUTPUT" "$@"
