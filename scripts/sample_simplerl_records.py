#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _sample_level(df: pd.DataFrame, level: int, n: int, seed: int) -> pd.DataFrame:
    part = df[df["level"] == level].copy()
    if part.empty:
        raise ValueError(f"No rows found for level={level}")
    replace = n > len(part)
    return part.sample(n=n, replace=replace, random_state=seed).sort_index()


def _to_record(row_idx: int, row: pd.Series) -> dict:
    level = int(row["level"])
    pid = f"simplelr_l{level}_idx{row_idx}"
    ground_truth = row.get("gt_answer")
    if ground_truth is None:
        ground_truth = row.get("answer")
    if ground_truth is None:
        ground_truth = row.get("target")
    return {
        "problem_id": pid,
        "problem_text": str(row["question"]),
        "ground_truth": "" if ground_truth is None else str(ground_truth),
        "raw_question": str(row["question"]),
        "category": f"simplelr_level{level}",
        "answer_type": "math",
        "extra_info": {
            "level": level,
            "subject": str(row.get("subject", "")),
            "data_source": str(row.get("data_source", "")),
            "row_index": int(row_idx),
        },
    }


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _build_shards(records: list[dict], shard_size: int) -> list[list[dict]]:
    return [records[i : i + shard_size] for i in range(0, len(records), shard_size)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample balanced L3/L4/L5 SimpleRL records for ReasonMaxxer rollout generation.")
    ap.add_argument(
        "--train_parquet",
        default="data/external/simpleRL/simplelr_abel_level3to5/train.parquet",
    )
    ap.add_argument("--levels", default="3,4,5", help="Comma-separated levels to sample, e.g. 3,4,5")
    ap.add_argument("--per_level", type=int, default=200)
    ap.add_argument("--shard_size", type=int, default=100, help="Number of problems per generation shard")
    ap.add_argument(
        "--merged_shard_size",
        type=int,
        default=75,
        help="If >0, also shard the merged balanced pool into equal-sized generation files.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    levels = [int(x.strip()) for x in str(args.levels).split(",") if x.strip()]
    if not levels:
        raise ValueError("No levels requested.")
    if args.shard_size <= 0:
        raise ValueError("--shard_size must be positive.")

    df = pd.read_parquet(args.train_parquet)
    required_cols = {"question", "level"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in parquet: {sorted(missing)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "source_parquet": str(args.train_parquet),
        "seed": int(args.seed),
        "per_level": int(args.per_level),
        "shard_size": int(args.shard_size),
        "levels": levels,
        "source_level_counts": {str(int(k)): int(v) for k, v in df["level"].value_counts().sort_index().items()},
        "files": {},
    }

    all_records: list[dict] = []
    for offset, level in enumerate(levels):
        sampled = _sample_level(df, level, n=int(args.per_level), seed=int(args.seed) + offset + 1)
        records = [_to_record(int(idx), row) for idx, row in sampled.iterrows()]
        all_records.extend(records)

        level_records_path = out_dir / f"records_level{level}_{args.per_level}.json"
        level_ids_path = out_dir / f"ids_level{level}_{args.per_level}.json"
        _write_json(level_records_path, {"records": records})
        _write_json(level_ids_path, {"problem_ids": [r["problem_id"] for r in records]})

        shard_payloads = _build_shards(records, int(args.shard_size))
        shard_names: list[str] = []
        for shard_idx, shard_records in enumerate(shard_payloads):
            shard_name = f"records_level{level}_{args.per_level}_shard{shard_idx}.json"
            shard_path = out_dir / shard_name
            _write_json(shard_path, {"records": shard_records})
            shard_names.append(shard_name)

        manifest["files"][f"level{level}"] = {
            "records_json": str(level_records_path),
            "ids_json": str(level_ids_path),
            "num_records": len(records),
            "shards": shard_names,
        }

    merged_records_path = out_dir / f"records_l345_{len(all_records)}.json"
    merged_ids_path = out_dir / f"ids_l345_{len(all_records)}.json"
    _write_json(merged_records_path, {"records": all_records})
    _write_json(merged_ids_path, {"problem_ids": [r["problem_id"] for r in all_records]})

    manifest["files"]["merged"] = {
        "records_json": str(merged_records_path),
        "ids_json": str(merged_ids_path),
        "num_records": len(all_records),
    }

    if int(args.merged_shard_size) > 0:
        merged_shards = _build_shards(all_records, int(args.merged_shard_size))
        merged_shard_names: list[str] = []
        for shard_idx, shard_records in enumerate(merged_shards):
            shard_name = f"records_l345_{len(all_records)}_shard{shard_idx}.json"
            shard_path = out_dir / shard_name
            _write_json(shard_path, {"records": shard_records})
            merged_shard_names.append(shard_name)
        manifest["files"]["merged"]["shards"] = merged_shard_names
        manifest["files"]["merged"]["merged_shard_size"] = int(args.merged_shard_size)

    manifest_path = out_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    print(f"[saved] {manifest_path}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
