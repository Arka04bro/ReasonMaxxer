#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/00_config.sh"

# Phase 0: reproduce v1 at pilot scale across seeds. Until v1's own seed
# variance is known, no v2 comparison means anything.

TAU=$($PYTHON -c "import json,sys;print(json.load(open(sys.argv[1]))['tau_pos'])" \
      "$ROOT/selection/tau_calibration.json")
echo "[phase0] calibrated tau = $TAU"

for SEED in $SEEDS; do
  RUN="phase0_v1_seed${SEED}"
  echo "[phase0] === $RUN ==="
  mkdir -p "$ROOT/train/$RUN"

  $PYTHON scripts/prepare_training_data.py \
    --rollouts_file "$ROOT/selection/selected_rollouts_entropy.json" \
    --tau_pos "$TAU" --tau_neg "$TAU" \
    --gate entropy_tau \
    --min_pass 0.0 --max_pass 1.0 --max_target_problems 9999 \
    --selection_strategy closest_midpoint --seed "$SEED" \
    --target_ids_output "$ROOT/train/$RUN/target_ids.json" \
    --processed_output "$ROOT/train/$RUN/processed.json" \
    --training_examples_output "$ROOT/train/$RUN/examples.json" \
    --stats_output "$ROOT/train/$RUN/stats.json" \
    --selected_records_output "$ROOT/train/$RUN/records.json"

  $PYTHON scripts/train_reasonmaxxer.py \
    --training_data "$ROOT/train/$RUN/processed.json" \
    --base_model "$MODEL_PATH" \
    --output_dir "$ROOT/checkpoints/$RUN" \
    --variant reasonmaxxer \
    --target_modules "$TARGET_MODULES" \
    --lora_rank "$LORA_RANK" --lora_alpha "$LORA_ALPHA" --lora_dropout 0.0 \
    --epochs "$EPOCHS" --batch_size 1 --grad_accum_steps "$GRAD_ACCUM" \
    --learning_rate "$LR" --warmup_steps 20 --max_grad_norm 1.0 \
    --kl_weight "$KL_WEIGHT" --adv_clip 2.5 \
    --decision_objective adv_ce \
    --max_seq_len "$MAX_SEQ_LEN" --truncate_side right \
    --dtype "$DTYPE" --hf_attn_implementation "$ATTN" \
    --seed "$SEED" --logging_steps 10 \
    --save_every_fractional_epoch 0.25 --save_every_epoch \
    --max_optimizer_steps "$MAX_OPT_STEPS"
done

echo "[phase0] done: $(echo $SEEDS | wc -w) runs in $ROOT/checkpoints"
