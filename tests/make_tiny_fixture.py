#!/usr/bin/env python3
"""Build a CPU-sized fixture (tiny random Qwen2 + synthetic rollouts).

The v2 study changes the training objective, so the training script needs a
regression test that runs without a GPU. Everything here is constructed
offline: a randomly initialised Qwen2 of a few thousand parameters and a
tokenizer over a toy vocabulary.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

VOCAB = 512


def build_model(out_dir: Path, seed: int) -> None:
    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        tie_word_embeddings=True,
        bos_token_id=0,
        eos_token_id=1,
    )
    model = Qwen2ForCausalLM(cfg)
    model.save_pretrained(out_dir)

    vocab = {"<pad>": 0, "<eos>": 1}
    for i in range(2, VOCAB):
        vocab[f"t{i}"] = i
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<pad>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>")
    tok.save_pretrained(out_dir)


def build_rollouts(path: Path, *, n_problems: int, n_gens: int, seed: int) -> None:
    rng = random.Random(seed)
    rollouts = []
    for pid in range(n_problems):
        rewards = [1.0 if rng.random() < 0.5 else 0.0 for _ in range(n_gens)]
        # Guarantee both signs so the balanced sampler has something to work with.
        rewards[0], rewards[-1] = 1.0, 0.0
        mean_r = sum(rewards) / len(rewards)
        var = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
        std_r = max(var**0.5, 1e-8)
        for g, r in enumerate(rewards):
            prompt_len = rng.randint(6, 12)
            comp_len = rng.randint(20, 40)
            ids = [rng.randrange(2, VOCAB) for _ in range(prompt_len + comp_len)]
            gen_pred_start = prompt_len - 1
            entropies = [max(0.0, rng.gauss(1.0, 0.8)) for _ in range(comp_len)]
            decisions = [
                gen_pred_start + i
                for i, e in enumerate(entropies)
                if e > 1.4 and gen_pred_start + i < len(ids) - 1
            ]
            rollouts.append(
                {
                    "problem_id": f"prob-{pid}",
                    "gen_index": g,
                    "input_ids": ids,
                    "prompt_length": prompt_len,
                    "completion_length": comp_len,
                    "correct": bool(r > 0.5),
                    "reward": float(r),
                    "advantage": float((r - mean_r) / std_r),
                    "entropies": entropies,
                    "generation_pred_start": gen_pred_start,
                    "decision_positions": decisions,
                    "num_tokens": len(ids),
                    "num_pred_tokens": len(ids) - 1,
                }
            )
    path.write_text(
        json.dumps({"rollouts": rollouts, "config": {"synthetic": True}}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Build the tiny CPU test fixture.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_problems", type=int, default=4)
    p.add_argument("--n_gens", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out_dir)
    (out / "tiny_model").mkdir(parents=True, exist_ok=True)
    build_model(out / "tiny_model", seed=int(args.seed))
    build_rollouts(out / "processed.json", n_problems=args.n_problems, n_gens=args.n_gens, seed=int(args.seed))
    print(f"[fixture] model={out / 'tiny_model'} rollouts={out / 'processed.json'}")


if __name__ == "__main__":
    main()
