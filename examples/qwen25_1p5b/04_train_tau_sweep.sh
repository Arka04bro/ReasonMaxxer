#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-1.5B}
ROOT=${ROOT:-outputs/qwen25_1p5b_default}
TAUS=${TAUS:-"1.0 1.4 1.8 2.2"}
GPUS=${GPUS:-"0 1 2 3"}

mkdir -p "$ROOT/train" "$ROOT/checkpoints"

gpu_arr=($GPUS)
i=0
for TAU in $TAUS; do
  TAG=${TAU/./p}
  RUN_NAME="qwen25_1p5b_reasonmaxxer_tau${TAG}_r32_lr1e4_ep1p35_seed42"

  $PYTHON scripts/prepare_training_data.py \
    --rollouts_file "$ROOT/selection/selected_rollouts_trim80_entropy.json" \
    --tau_pos "$TAU" \
    --tau_neg "$TAU" \
    --min_pass 0.0 \
    --max_pass 1.0 \
    --max_target_problems 9999 \
    --selection_strategy closest_midpoint \
    --seed 42 \
    --target_ids_output "$ROOT/train/target_ids_tau${TAG}.json" \
    --processed_output "$ROOT/train/processed_tau${TAG}.json" \
    --training_examples_output "$ROOT/train/training_examples_tau${TAG}.json" \
    --stats_output "$ROOT/train/training_stats_tau${TAG}.json" \
    --selected_records_output "$ROOT/train/selected_records_tau${TAG}.json"

  GPU=${gpu_arr[$((i % ${#gpu_arr[@]}))]}
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/train_reasonmaxxer.py \
    --training_data "$ROOT/train/processed_tau${TAG}.json" \
    --base_model "$MODEL_PATH" \
    --output_dir "$ROOT/checkpoints/$RUN_NAME" \
    --variant reasonmaxxer \
    --target_modules q_proj,k_proj,v_proj,o_proj \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.0 \
    --epochs 2 \
    --batch_size 1 \
    --grad_accum_steps 8 \
    --learning_rate 1e-4 \
    --warmup_steps 50 \
    --weight_decay 0.0 \
    --max_grad_norm 1.0 \
    --kl_weight 0.2 \
    --adv_clip 2.5 \
    --decision_objective adv_ce \
    --beta_neg 1.0 \
    --max_seq_len 8192 \
    --truncate_side right \
    --dtype bfloat16 \
    --hf_attn_implementation sdpa \
    --curriculum_ordering \
    --curriculum_key signal_clarity \
    --seed 42 \
    --logging_steps 20 \
    --save_every_fractional_epoch 0.15 \
    --save_every_epoch \
    --max_optimizer_steps 149 &

  i=$((i + 1))
done
wait
