#!/usr/bin/env bash
set -euo pipefail

# Example usage (aggregates seeds once for evaluation):
#   bash scripts/evaluation.sh "1 2 3"

SEEDS="${1:-"1 2 3"}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Optional overrides: export CKPT_DIR or OUT_DIR before calling this script
CKPT_DIR="${CKPT_DIR:-${ROOT_DIR}/best_checkpoints}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/analysis_output}"

# convert space-separated to comma-separated
SEED_CSV="$(echo "$SEEDS" | tr ' ' ',')" 

python "${ROOT_DIR}/analysis/evaluate.py" \
  --ckpt_dir "${CKPT_DIR}" \
  --out_dir "${OUT_DIR}" \
  --mode checkpoint \
  --fullscratch_seeds "${SEED_CSV}" \
  --multitask_seeds "${SEED_CSV}"
