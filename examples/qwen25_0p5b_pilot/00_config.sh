#!/usr/bin/env bash
# Shared configuration for the 0.5B pilot. Source this, or let the numbered
# scripts source it themselves.
#
# The pilot exists to run Phase 0 (reproduce v1 with error bars) and Phase 1
# (budget-matched gate controls) on free-tier hardware. Everything is sized for
# a single 16 GB T4: fp16 rather than bf16 (Turing has no bf16), short
# sequences, and GSM8K rather than MATH level 3-5 so that a 0.5B base model
# actually produces a mid-difficulty pool.

export PYTHON=${PYTHON:-python}
export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-0.5B}
export ROOT=${ROOT:-outputs/qwen25_0p5b_pilot}

# Data. GSM8K train, disjoint train/holdout split; evaluation uses GSM8K test.
export SOURCE=${SOURCE:-gsm8k}
export N_TRAIN=${N_TRAIN:-100}
export N_HOLDOUT=${N_HOLDOUT:-60}
export N_GENS=${N_GENS:-16}

# Generation. GSM8K solutions are short, so 1024 tokens is generous; this is the
# single biggest cost lever in the pilot.
export MAX_TOKENS=${MAX_TOKENS:-1024}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-1536}
export TEMPERATURE=${TEMPERATURE:-0.6}
export TOP_P=${TOP_P:-0.95}
export DTYPE=${DTYPE:-float16}
export ATTN=${ATTN:-sdpa}
export GEN_BATCH=${GEN_BATCH:-16}

# Decision-token budget. tau is calibrated to this per-rollout token count
# rather than fixed at v1's 1.4 nats, which is a property of Qwen2.5-1.5B's
# entropy distribution and does not transfer across model scales.
export TARGET_TOKENS_PER_ROLLOUT=${TARGET_TOKENS_PER_ROLLOUT:-10}

# Training.
export LORA_RANK=${LORA_RANK:-16}
export LORA_ALPHA=${LORA_ALPHA:-32}
export TARGET_MODULES=${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}
export EPOCHS=${EPOCHS:-2}
export LR=${LR:-1e-4}
export GRAD_ACCUM=${GRAD_ACCUM:-8}
export KL_WEIGHT=${KL_WEIGHT:-0.2}
export MAX_SEQ_LEN=${MAX_SEQ_LEN:-1536}
export MAX_OPT_STEPS=${MAX_OPT_STEPS:-150}

# Three seeds is the minimum for a difference between conditions to mean
# anything; every phase runs all of them.
export SEEDS=${SEEDS:-"42 43 44"}

# Evaluation. n=16 samples supports pass@k up to k=8 with room to spare.
export EVAL_DATASET=${EVAL_DATASET:-gsm8k}
export EVAL_MAX_PROBLEMS=${EVAL_MAX_PROBLEMS:-200}
export EVAL_N_GENS=${EVAL_N_GENS:-16}
export PASS_AT_KS=${PASS_AT_KS:-1,2,4,8,16}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
