#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-1.5B}
ROOT=${ROOT:-outputs/qwen25_1p5b_default}

mkdir -p "$ROOT/gen" "$ROOT/score"

for SHARD in 0 1 2; do
  CUDA_VISIBLE_DEVICES=${SHARD} $PYTHON scripts/generate_rollouts.py     --model_path "$MODEL_PATH"     --condition_name "qwen25_1p5b_shard${SHARD}_n20"     --dataset math500     --records_file "$ROOT/sampled_records/records_l345_300_shard${SHARD}.json"     --problem_set full     --num_generations 20     --batch_size 8     --tensor_parallel_size 1     --temperature 0.6     --top_p 0.95     --max_tokens 8192     --seed 42     --output_dir "$ROOT/gen"     --output_name "qwen25_1p5b_shard${SHARD}_n20.json"     --force &
done
wait

for SHARD in 0 1 2; do
  CUDA_VISIBLE_DEVICES=${SHARD} $PYTHON scripts/score_rollouts.py     --input_json "$ROOT/gen/qwen25_1p5b_shard${SHARD}_n20.json"     --output_file "$ROOT/score/qwen25_1p5b_shard${SHARD}_n20_entropy.json"     --model_path "$MODEL_PATH"     --dataset math500     --dtype bfloat16     --hf_attn_implementation sdpa     --max_seq_len 8192     --n_gens 20     --force &
done
wait
