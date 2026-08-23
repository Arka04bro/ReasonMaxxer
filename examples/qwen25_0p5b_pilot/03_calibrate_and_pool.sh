#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/00_config.sh"

mkdir -p "$ROOT/selection"

# Resolve tau from the token budget instead of hardcoding v1's 1.4 nats.
$PYTHON scripts/calibrate_tau.py \
  --input "$ROOT/score/base_n${N_GENS}_entropy.json" \
  --target_tokens_per_rollout "$TARGET_TOKENS_PER_ROLLOUT" \
  --output "$ROOT/selection/tau_calibration.json"

# Mid-difficulty pool. --require_both_signs keeps only problems with at least
# one correct and one incorrect rollout, which is what the contrastive update
# needs. No length trimming here: v1's --trim_fraction 0.8 drops the longest
# fifth of rollouts and biases the method toward shorter outputs, and the pilot
# is meant to measure that confound rather than bake it in.
$PYTHON scripts/select_mid_pool.py \
  --input "$ROOT/score/base_n${N_GENS}_entropy.json" \
  --output_dir "$ROOT/selection" \
  --max_target_problems 50 \
  --require_both_signs \
  --trim_fraction 1.0
