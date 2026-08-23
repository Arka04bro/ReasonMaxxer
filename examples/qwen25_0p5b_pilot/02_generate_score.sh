#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/00_config.sh"

# Generation and entropy scoring happen once and are reused by every condition
# in Phase 0, 1 and 2. This is the only step that needs the full rollout budget.

mkdir -p "$ROOT/gen" "$ROOT/score"

$PYTHON scripts/generate_rollouts.py \
  --model_path "$MODEL_PATH" \
  --condition_name "qwen25_0p5b_base_n${N_GENS}" \
  --dataset "$SOURCE" \
  --records_file "$ROOT/records/records_train.json" \
  --problem_set full \
  --num_generations "$N_GENS" \
  --batch_size "$GEN_BATCH" \
  --tensor_parallel_size 1 \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --max_tokens "$MAX_TOKENS" \
  --max_model_len "$MAX_MODEL_LEN" \
  --seed 42 \
  --output_dir "$ROOT/gen" \
  --output_name "base_n${N_GENS}.json" \
  --force

$PYTHON scripts/score_rollouts.py \
  --input_json "$ROOT/gen/base_n${N_GENS}.json" \
  --output_file "$ROOT/score/base_n${N_GENS}_entropy.json" \
  --model_path "$MODEL_PATH" \
  --dataset "$SOURCE" \
  --dtype "$DTYPE" \
  --hf_attn_implementation "$ATTN" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --n_gens "$N_GENS" \
  --force
