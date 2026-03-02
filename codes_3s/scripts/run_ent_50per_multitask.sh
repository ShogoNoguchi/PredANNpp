#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SEED="${1:-42}"
RUN_NAME="${2:-EntropyMultitask_Finetune_seed${SEED}}"

cd "$ROOT_DIR"

python main_3s.py \
  --mode EntropyMultitask \
  --seed "$SEED" \
  --training_date "$RUN_NAME"
