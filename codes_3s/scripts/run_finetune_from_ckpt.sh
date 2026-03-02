#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <pretrain_ckpt_path> [seed] [run_name]"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CKPT_PATH="$1"
SEED="${2:-42}"
RUN_NAME="${3:-finetune_from_ckpt_seed${SEED}}"

cd "$ROOT_DIR"

python main_3s.py \
  --mode Finetune \
  --seed "$SEED" \
  --training_date "$RUN_NAME" \
  --pretrain_ckpt_path "$CKPT_PATH"
