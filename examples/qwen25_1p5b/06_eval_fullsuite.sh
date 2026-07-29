#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-1.5B}
ROOT=${ROOT:-outputs/qwen25_1p5b_default}
CHECKPOINT=${CHECKPOINT:?set CHECKPOINT to the chosen LoRA checkpoint directory}
MAX_LORA_RANK=${MAX_LORA_RANK:-32}

mkdir -p "$ROOT/final_eval"

# For math500 and gsm8k the repo can load the datasets directly.
for DATASET in math500 gsm8k; do
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} $PYTHON scripts/generate_rollouts.py \
    --model_path "$MODEL_PATH" \
    --lora_adapter "$CHECKPOINT" \
    --max_lora_rank "$MAX_LORA_RANK" \
    --condition_name "$(basename "$CHECKPOINT")_${DATASET}" \
    --dataset "$DATASET" \
    --problem_set full \
    --num_generations 1 \
    --batch_size 8 \
    --tensor_parallel_size 1 \
    --temperature 0.6 \
    --top_p 0.95 \
    --max_tokens 8192 \
    --seed 42 \
    --prompt_style auto \
    --stop_profile auto \
    --math500_extractor auto \
    --output_dir "$ROOT/final_eval" \
    --output_name "$(basename "$CHECKPOINT")_${DATASET}.json" \
    --force
 done

# For local benchmarks, pass records files explicitly.
for DATASET in aime24 amc23 minerva_math olympiadbench; do
  RECORDS_FILE=${DATASET^^}_RECORDS_FILE
  RECORDS_PATH=${!RECORDS_FILE:-}
  if [ -n "$RECORDS_PATH" ]; then
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} $PYTHON scripts/generate_rollouts.py \
      --model_path "$MODEL_PATH" \
      --lora_adapter "$CHECKPOINT" \
      --max_lora_rank "$MAX_LORA_RANK" \
      --condition_name "$(basename "$CHECKPOINT")_${DATASET}" \
      --dataset "$DATASET" \
      --records_file "$RECORDS_PATH" \
      --problem_set full \
      --num_generations 1 \
      --batch_size 8 \
      --tensor_parallel_size 1 \
      --temperature 0.6 \
      --top_p 0.95 \
      --max_tokens 8192 \
      --seed 42 \
      --prompt_style auto \
      --stop_profile auto \
      --math500_extractor auto \
      --output_dir "$ROOT/final_eval" \
      --output_name "$(basename "$CHECKPOINT")_${DATASET}.json" \
      --force
  fi
done
