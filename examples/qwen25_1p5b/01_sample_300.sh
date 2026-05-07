#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
ROOT=${ROOT:-outputs/qwen25_1p5b_default}
TRAIN_PARQUET=${TRAIN_PARQUET:-data/external/simpleRL/simplelr_abel_level3to5/train.parquet}

mkdir -p "$ROOT"

$PYTHON scripts/sample_simplerl_records.py   --train_parquet "$TRAIN_PARQUET"   --levels 3,4,5   --per_level 100   --shard_size 100   --merged_shard_size 100   --seed 42   --out_dir "$ROOT/sampled_records"
