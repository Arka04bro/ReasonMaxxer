#!/usr/bin/env python3
"""Sample pilot train/holdout problems from a public train split.

The v1 example pipeline draws training problems from the SimpleRL-Zoo
`level3to5` parquet, which suits a 1.5B model. The 0.5B pilot needs an easier
band: mid-difficulty selection keeps problems whose pass rate is near 0.5, and a
0.5B base model solves almost nothing at MATH level 3-5, so no mid pool would
form. The pilot therefore shifts the *difficulty band*, not the selection rule.

Both splits come from the training data, never from the evaluation sets, and the
holdout used for checkpoint selection is disjoint from the training problems.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from pathlib import Path

from datasets import load_dataset

from reasonmaxxer.answer_extraction import get_ground_truth

SOURCES = {
    "gsm8k": {
        "hf_name": "openai/gsm8k",
        "config_name": "main",
        "split": "train",
        "problem_key": "question",
        "answer_key": "answer",
        "dataset_tag": "gsm8k",
        "id_prefix": "GSM8K/train",
    },
    "math_train": {
        "hf_name": "nlile/hendrycks-MATH-benchmark",
        "config_name": None,
        "split": "train",
        "problem_key": "problem",
        "answer_key": "solution",
        "dataset_tag": "math500",
        "id_prefix": "MATH/train",
    },
}


def load_source(name: str, levels: list[int] | None) -> list[dict]:
    cfg = SOURCES[name]
    if cfg["config_name"]:
        rows = load_dataset(cfg["hf_name"], cfg["config_name"], split=cfg["split"])
    else:
        rows = load_dataset(cfg["hf_name"], split=cfg["split"])

    records = []
    for idx, row in enumerate(rows):
        if levels:
            level_raw = str(row.get("level", ""))
            digits = "".join(ch for ch in level_raw if ch.isdigit())
            if not digits or int(digits) not in levels:
                continue
        records.append(
            {
                "problem_id": f"{cfg['id_prefix']}/{idx}",
                "problem_text": str(row[cfg["problem_key"]]),
                "ground_truth": get_ground_truth(str(row[cfg["answer_key"]]), dataset=cfg["dataset_tag"]),
                "category": str(row.get("type", cfg["dataset_tag"])),
            }
        )
    return [r for r in records if r["ground_truth"] not in (None, "")]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Sample pilot train/holdout records.")
    p.add_argument("--source", choices=sorted(SOURCES), default="gsm8k")
    p.add_argument("--levels", default="", help="MATH levels to keep, e.g. 1,2,3 (ignored for gsm8k).")
    p.add_argument("--n_train", type=int, default=100)
    p.add_argument("--n_holdout", type=int, default=60)
    p.add_argument("--shard_size", type=int, default=0, help="Split the train records into shards of this size (0 = one file).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    levels = [int(x) for x in str(args.levels).split(",") if x.strip()]
    records = load_source(args.source, levels)
    need = int(args.n_train) + int(args.n_holdout)
    if len(records) < need:
        raise ValueError(f"Source has {len(records)} usable records, need {need}.")

    rng = random.Random(int(args.seed))
    picked = rng.sample(records, k=need)
    train, holdout = picked[: int(args.n_train)], picked[int(args.n_train) :]

    train_ids = {r["problem_id"] for r in train}
    holdout_ids = {r["problem_id"] for r in holdout}
    assert not (train_ids & holdout_ids), "train and holdout overlap"

    out = Path(args.out_dir)
    meta = {"source": args.source, "levels": levels, "seed": int(args.seed), "pool_size": len(records)}

    write_json(out / "records_train.json", {"meta": meta, "records": train})
    write_json(out / "records_holdout.json", {"meta": meta, "records": holdout})
    write_json(out / "ids_train.json", {"problem_ids": sorted(train_ids)})
    write_json(out / "ids_holdout.json", {"problem_ids": sorted(holdout_ids)})

    if int(args.shard_size) > 0:
        size = int(args.shard_size)
        for i in range(0, len(train), size):
            shard = train[i : i + size]
            write_json(
                out / f"records_train_shard{i // size}.json",
                {"meta": {**meta, "shard": i // size}, "records": shard},
            )

    print(f"[done] pool={len(records)} train={len(train)} holdout={len(holdout)} disjoint=True")


if __name__ == "__main__":
    main()
