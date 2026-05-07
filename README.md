# ReasonMaxxer

**Rethinking RL for LLM Reasoning: It's Sparse Policy Selection, Not Capability Learning**

This repository contains the **minimal public ReasonMaxxer pipeline** used for our rollout-generation, entropy scoring, contrastive training, and checkpoint evaluation runs.

Unlike the internal research repo, this release intentionally excludes unrelated analysis code, RL baselines, code-reasoning utilities, and experiment bookkeeping. The goal is to make the core **RL-free reasoning post-training pipeline** easy to read, reproduce, and extend.

![ReasonMaxxer main table](assets/reasonmaxxer_table.png)

## Overview

ReasonMaxxer is built around a simple empirical claim: for mathematical reasoning, the useful footprint of RL is **sparse** and concentrated at **high-entropy decision points**. Instead of running online RL, ReasonMaxxer:

1. samples a few hundred problems,
2. generates multiple base-model rollouts,
3. computes token entropies with teacher-forced scoring,
4. selects medium-pass-rate problems with both correct and incorrect rollouts,
5. trains a LoRA adapter with **contrastive loss only on entropy-gated decision tokens**, while anchoring the rest of the distribution to the base model.

The public repo includes:

- rollout generation with model-specific prompt auto-resolution,
- entropy scoring for generated rollouts,
- training-data preparation,
- ReasonMaxxer LoRA training,
- checkpoint evaluation on fixed holdout splits,
- example shell scripts for the default Qwen2.5-1.5B pipeline.

## Repository layout

```text
ReasonMaxxer/
├── assets/
├── examples/
│   └── qwen25_1p5b/
├── reasonmaxxer/
│   ├── answer_extraction.py
│   ├── answer_verification.py
│   ├── config.py
│   ├── eval_lib.py
│   └── generation.py
├── scripts/
│   ├── eval_checkpoints.py
│   ├── generate_rollouts.py
│   ├── prepare_training_data.py
│   ├── sample_simplerl_records.py
│   ├── score_rollouts.py
│   ├── select_mid_pool.py
│   └── train_reasonmaxxer.py
└── requirements.txt
```

## Installation

```bash
conda create -n reasonmaxxer python=3.10 -y
conda activate reasonmaxxer
pip install -r requirements.txt
```

## Data format

For custom local benchmarks, pass `--records_file` with a JSON file of the form:

```json
{
  "records": [
    {
      "problem_id": "example-1",
      "problem_text": "...",
      "ground_truth": "...",
      "category": "math"
    }
  ]
}
```

Built-in dataset loading is provided for:

- `math500` via `nlile/hendrycks-MATH-benchmark`
- `gsm8k` via `openai/gsm8k`

For `aime24`, `amc23`, `minerva_math`, and `olympiadbench`, either:

- place benchmark JSONs under `data/benchmarks/`, or
- pass `--records_file` directly.

## Prompting defaults

Prompt style is resolved automatically from the model name unless overridden.

- Qwen2.5 base models: `qwen_boxed`
- Qwen3 instruct/reasoning models: `qwen3_chat` or `chat_template`
- DeepSeek-R1-Distill / ORZ / Open-RS / related reasoning-chat models: `chat_template`
- LLaMA / Mistral: `llama_abel`
- OLMo math checkpoints: `qwen_boxed`, `olmo3_math`, or `olmo3_rlzero_math` depending on the checkpoint name

The default generation settings in this repo match our paper-facing runs:

- `temperature=0.6`
- `top_p=0.95`
- `max_tokens=8192`
- `seed=42`

## Quick start: default Qwen2.5-1.5B run

The example shell scripts under `examples/qwen25_1p5b/` implement the default direct pipeline:

- sample `3 x 100` problems from SimpleRL levels `3/4/5`
- generate `20` rollouts per problem
- entropy-score the rollouts
- select the global **mid-50** problem pool
- trim the longest `20%`
- train a `tau=1.4` or tau sweep LoRA adapter
- evaluate checkpoints on a fixed holdout split

### 1. Sample the 300-problem candidate set

```bash
bash examples/qwen25_1p5b/01_sample_300.sh
```

### 2. Generate and score `3 x 100 x 20` rollouts

```bash
bash examples/qwen25_1p5b/02_generate_score_3x100x20.sh
```

### 3. Select the mid-50 pool and apply trim-80

```bash
bash examples/qwen25_1p5b/03_select_mid50_trim80.sh
```

### 4. Train ReasonMaxxer

```bash
bash examples/qwen25_1p5b/04_train_tau1p4.sh
```

Optional tau sweep:

```bash
bash examples/qwen25_1p5b/04_train_tau_sweep.sh
```

### 5. Evaluate checkpoints on a fixed holdout split

```bash
bash examples/qwen25_1p5b/05_eval_holdout60.sh
```

### 6. Evaluate a chosen checkpoint on the full benchmark suite

```bash
bash examples/qwen25_1p5b/06_eval_fullsuite.sh
```

## Example end-to-end defaults

The example Qwen2.5-1.5B scripts use the same paper-facing defaults:

- source candidate pool: `300` SimpleRL train problems, balanced as `100` each from levels `3/4/5`
- rollout count: `20` per problem
- pool selection: globally closest to `0.5` empirical pass rate
- trim: global shortest `80%`
- LoRA target modules: `q_proj,k_proj,v_proj,o_proj`
- rank: `32`
- alpha: `64`
- dropout: `0.0`
- objective: contrastive decision-token training + KL anchor

## Notes

- `scripts/generate_rollouts.py` can evaluate either a base model or a LoRA adapter via vLLM LoRA requests.
- `scripts/score_rollouts.py` performs teacher-forced entropy extraction using Hugging Face models.
- `scripts/train_reasonmaxxer.py` is the training entrypoint for the contrastive LoRA method.
- `scripts/eval_checkpoints.py` sweeps saved checkpoints over a fixed holdout ID file and writes a summary CSV.

## Before you publish

This repo is intentionally minimal, but you still need to verify three external dependencies before making it public:

1. benchmark files you reference locally are either redistributed legally or replaced with instructions,
2. model identifiers in the example scripts point to public model names or environment variables,
3. the citation metadata in `CITATION.cff` and the BibTeX block below are updated with the final paper metadata.

## Publishing checklist

```bash
git init
git branch -M main
git add .
git commit -m "Initial public release"
gh repo create ReasonMaxxer --public --source=. --remote=origin --push
```

If you do not use `gh`, create an empty GitHub repository first, then run:

```bash
git init
git branch -M main
git add .
git commit -m "Initial public release"
git remote add origin git@github.com:<your-org>/ReasonMaxxer.git
git push -u origin main
```

## Citation

```bibtex
@article{reasonmaxxer2026placeholder,
  title={ReasonMaxxer: Sparse Policy Selection for RL-Free Reasoning},
  author={Anonymous until release},
  year={2026}
}
```
