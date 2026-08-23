#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/00_config.sh"

mkdir -p "$ROOT/records"

$PYTHON scripts/sample_pilot_records.py \
  --source "$SOURCE" \
  --n_train "$N_TRAIN" \
  --n_holdout "$N_HOLDOUT" \
  --seed 42 \
  --out_dir "$ROOT/records"
