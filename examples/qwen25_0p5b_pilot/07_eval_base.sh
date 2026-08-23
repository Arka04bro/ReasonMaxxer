#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/00_config.sh"

# The base model's own pass@k curve. Every claim about "policy selection rather
# than capability acquisition" is a statement about the gap between this curve
# and a tuned checkpoint's, so it has to be measured at identical temperature
# and sample count.

mkdir -p "$ROOT/eval/base_test_n${EVAL_N_GENS}"

$PYTHON scripts/generate_rollouts.py \
  --model_path "$MODEL_PATH" \
  --condition_name "base_test_n${EVAL_N_GENS}" \
  --dataset "$EVAL_DATASET" \
  --problem_set full \
  --max_problems "$EVAL_MAX_PROBLEMS" \
  --num_generations "$EVAL_N_GENS" \
  --batch_size "$GEN_BATCH" \
  --temperature "$TEMPERATURE" --top_p "$TOP_P" \
  --max_tokens "$MAX_TOKENS" --max_model_len "$MAX_MODEL_LEN" \
  --seed 42 \
  --output_dir "$ROOT/eval/base_test_n${EVAL_N_GENS}" \
  --output_name "base.json" \
  --force

$PYTHON - "$ROOT/eval/base_test_n${EVAL_N_GENS}/base.json" "$PASS_AT_KS" <<'PY'
import json, sys
sys.path.insert(0, ".")
from reasonmaxxer import metrics as mx
rows = json.load(open(sys.argv[1], encoding="utf-8")).get("results", [])
ks = tuple(int(x) for x in sys.argv[2].split(",") if x.strip())
summary = mx.summarize(rows, ks=ks)
print(mx.format_table(summary))
PY
