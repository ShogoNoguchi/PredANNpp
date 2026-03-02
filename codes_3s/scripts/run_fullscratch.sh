#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SEED="${1:-42}"
RUN_NAME="${2:-Fullscratch_seed${SEED}}"

cd "$ROOT_DIR"

python main_3s.py \
  --mode Finetune \
  --seed "$SEED" \
  --training_date "$RUN_NAME" \
  --pretrain_ckpt_path None
