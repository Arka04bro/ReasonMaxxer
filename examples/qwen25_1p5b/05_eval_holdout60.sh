#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-1.5B}
ROOT=${ROOT:-outputs/qwen25_1p5b_default}
RUN_NAME=${RUN_NAME:-qwen25_1p5b_reasonmaxxer_tau1p4_r32_lr1e4_ep1p35_seed42}
HOLDOUT_IDS_FILE=${HOLDOUT_IDS_FILE:?set HOLDOUT_IDS_FILE to a JSON file with problem_ids}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} $PYTHON scripts/eval_checkpoints.py   --run_dir "$ROOT/checkpoints/$RUN_NAME"   --base_model "$MODEL_PATH"   --max_lora_rank 32   --holdout_ids_file "$HOLDOUT_IDS_FILE"   --holdout_ids_key problem_ids   --dataset math500   --problem_set full   --max_problems 60   --output_tag holdout60   --output_dir "$ROOT/eval/${RUN_NAME}_holdout60_n1"   --num_generations 1   --device 0   --batch_size 8   --temperature 0.6   --top_p 0.95   --max_tokens 8192   --seed 42   --match_timeout_s 0.2   --prompt_style auto   --stop_profile auto   --math500_extractor auto   --force   --update_metrics_csv
