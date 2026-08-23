#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/00_config.sh"

# Phase 1: only the gate changes. Same rollouts, same objective, same number of
# optimizer steps, and the same per-rollout token budget --- control gates are
# handed exactly as many tokens as the entropy gate selected on that same
# rollout (--budget_match_tau).
#
# Equal token counts do not imply equal gradient mass, because cross-entropy
# correlates with base entropy. stats.json records gate.signal_mass_total for
# every condition, and it belongs in the results table next to accuracy.

GATES=${GATES:-"entropy_tau random low_entropy first_k last_k all"}
RUN_SHUFFLE=${RUN_SHUFFLE:-1}

TAU=$($PYTHON -c "import json,sys;print(json.load(open(sys.argv[1]))['tau_pos'])" \
      "$ROOT/selection/tau_calibration.json")

prepare_and_train () {
  local RUN=$1; shift
  mkdir -p "$ROOT/train/$RUN"
  $PYTHON scripts/prepare_training_data.py \
    --rollouts_file "$ROOT/selection/selected_rollouts_entropy.json" \
    --tau_pos "$TAU" --tau_neg "$TAU" \
    --min_pass 0.0 --max_pass 1.0 --max_target_problems 9999 \
    --selection_strategy closest_midpoint \
    --target_ids_output "$ROOT/train/$RUN/target_ids.json" \
    --processed_output "$ROOT/train/$RUN/processed.json" \
    --training_examples_output "$ROOT/train/$RUN/examples.json" \
    --stats_output "$ROOT/train/$RUN/stats.json" \
    --selected_records_output "$ROOT/train/$RUN/records.json" \
    "$@"

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
}

for SEED in $SEEDS; do
  for GATE in $GATES; do
    RUN="phase1_${GATE}_seed${SEED}"
    echo "[phase1] === $RUN ==="
    EXTRA=(--gate "$GATE" --seed "$SEED" --gate_seed "$SEED")
    if [ "$GATE" != "entropy_tau" ] && [ "$GATE" != "all" ]; then
      EXTRA+=(--budget_match_tau)
    fi
    prepare_and_train "$RUN" "${EXTRA[@]}"
  done

  if [ "$RUN_SHUFFLE" = "1" ]; then
    RUN="phase1_shuffled_adv_seed${SEED}"
    echo "[phase1] === $RUN ==="
    prepare_and_train "$RUN" --gate entropy_tau --seed "$SEED" --gate_seed "$SEED" \
      --shuffle_advantages within_problem
  fi
done

echo "[phase1] gate budgets and signal mass:"
$PYTHON - "$ROOT/train" <<'PY'
import json, sys
from pathlib import Path
rows = []
for stats in sorted(Path(sys.argv[1]).glob("phase1_*/stats.json")):
    g = json.loads(stats.read_text(encoding="utf-8")).get("gate", {})
    if g.get("num_rollouts"):
        rows.append((stats.parent.name, g))
print(f"{'run':38s} {'tok/roll':>9} {'match':>7} {'H_sel':>7} {'mass':>10} {'relpos':>7}")
for name, g in rows:
    print(f"{name:38s} {g['tokens_per_rollout_mean']:9.2f} "
          f"{(g['budget_match_ratio'] or float('nan')):7.3f} {g['selected_entropy_mean']:7.3f} "
          f"{g['signal_mass_total']:10.1f} {g['relative_position_mean']:7.2f}")
PY
