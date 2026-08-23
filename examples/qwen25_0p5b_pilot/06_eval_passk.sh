#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/00_config.sh"

# Two evaluations, deliberately separated:
#
#   holdout  - GSM8K train problems held out from training, n=1, used ONLY to
#              pick a checkpoint. Selecting on the benchmark would invalidate
#              every number that follows.
#   test     - GSM8K test with n=16, pass@k and maj@k. The policy-selection
#              claim predicts pass@1 up and pass@8 flat; if pass@8 rises as much
#              as pass@1, the claim is in trouble.

RUN=${RUN:?set RUN to a checkpoint directory name under $ROOT/checkpoints}
STAGE=${STAGE:-holdout}

if [ "$STAGE" = "holdout" ]; then
  $PYTHON scripts/eval_checkpoints.py \
    --run_dir "$ROOT/checkpoints/$RUN" \
    --base_model "$MODEL_PATH" \
    --max_lora_rank "$LORA_RANK" \
    --records_file "$ROOT/records/records_holdout.json" \
    --dataset "$EVAL_DATASET" \
    --problem_set full \
    --output_tag holdout \
    --output_dir "$ROOT/eval/${RUN}_holdout_n1" \
    --num_generations 1 \
    --pass_at_ks 1 \
    --device 0 --batch_size "$GEN_BATCH" \
    --temperature "$TEMPERATURE" --top_p "$TOP_P" --max_tokens "$MAX_TOKENS" \
    --seed 42 --prompt_style auto --stop_profile auto \
    --skip_existing --update_metrics_csv
else
  CKPT_TAG=${CKPT_TAG:?set CKPT_TAG to the checkpoint chosen on the holdout}
  $PYTHON scripts/eval_checkpoints.py \
    --run_dir "$ROOT/checkpoints/$RUN" \
    --base_model "$MODEL_PATH" \
    --max_lora_rank "$LORA_RANK" \
    --dataset "$EVAL_DATASET" \
    --problem_set full \
    --max_problems "$EVAL_MAX_PROBLEMS" \
    --output_tag "test_${CKPT_TAG}" \
    --output_dir "$ROOT/eval/${RUN}_test_n${EVAL_N_GENS}" \
    --num_generations "$EVAL_N_GENS" \
    --pass_at_ks "$PASS_AT_KS" \
    --device 0 --batch_size "$GEN_BATCH" \
    --temperature "$TEMPERATURE" --top_p "$TOP_P" --max_tokens "$MAX_TOKENS" \
    --seed 42 --prompt_style auto --stop_profile auto \
    --skip_existing
fi
