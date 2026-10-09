set -euo pipefail
cd "$(dirname "$0")/.."
accelerate launch --num_processes "${T2Z_NUM_PROCESSES:-2}" t2z/train.py --config configs/ours/real_soft.yaml "$@"
