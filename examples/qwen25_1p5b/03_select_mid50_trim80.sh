#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
ROOT=${ROOT:-outputs/qwen25_1p5b_default}

$PYTHON scripts/select_mid_pool.py   --input "$ROOT/score/*_entropy.json"   --output_dir "$ROOT/selection"   --max_target_problems 50   --require_both_signs   --trim_fraction 0.8
