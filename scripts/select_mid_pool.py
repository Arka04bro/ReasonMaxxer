#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_rollouts(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for resolved in sorted(glob.glob(path)):
            payload = json.loads(Path(resolved).read_text(encoding="utf-8"))
            shard_rows = payload.get("rollouts", []) if isinstance(payload, dict) else payload
            if not isinstance(shard_rows, list):
                raise ValueError(f"Unsupported rollout file: {resolved}")
            rows.extend(shard_rows)
    return rows


def row_len(row: dict[str, Any]) -> int:
    return int(row.get("num_completion_tokens", row.get("completion_length", 10**9)))


def main() -> None:
    p = argparse.ArgumentParser(description="Select mid-pass-rate ReasonMaxxer problems and optionally trim the selected rollouts.")
    p.add_argument("--input", nargs="+", required=True, help="One or more scored rollout JSONs or glob patterns")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_target_problems", type=int, default=50)
    p.add_argument("--min_pass", type=float, default=0.0)
    p.add_argument("--max_pass", type=float, default=1.0)
    p.add_argument("--require_both_signs", action="store_true")
    p.add_argument("--trim_fraction", type=float, default=0.8)
    args = p.parse_args()

    rows = load_rollouts(args.input)
    if not rows:
        raise ValueError("No rollout rows loaded.")

    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exemplar: dict[str, dict[str, Any]] = {}
    for row in rows:
        pid = str(row["problem_id"])
        by_problem[pid].append(row)
        exemplar.setdefault(pid, row)

    infos = []
    for pid, grp in by_problem.items():
        n = len(grp)
        c = sum(1 for r in grp if bool(r.get("correct", False)))
        pass_rate = c / n if n else 0.0
        if not (args.min_pass <= pass_rate <= args.max_pass):
            continue
        if args.require_both_signs and not (1 <= c <= n - 1):
            continue
        infos.append(
            {
                "problem_id": pid,
                "num_rollouts": n,
                "num_correct": c,
                "pass_rate": pass_rate,
                "distance_to_midpoint": abs(pass_rate - 0.5),
                "trunc_count": sum(1 for r in grp if bool(r.get("truncated", False))),
                "avg_completion_tokens": sum(row_len(r) for r in grp) / n,
                "problem_text": exemplar[pid].get("problem_text"),
                "ground_truth": exemplar[pid].get("ground_truth"),
                "category": exemplar[pid].get("category"),
            }
        )

    infos.sort(
        key=lambda x: (
            x["distance_to_midpoint"],
            x["trunc_count"],
            x["avg_completion_tokens"],
            x["problem_id"],
        )
    )
    chosen = infos[: args.max_target_problems]
    chosen_ids = {x["problem_id"] for x in chosen}
    selected_rows = [r for r in rows if str(r["problem_id"]) in chosen_ids]

    selected_sorted = sorted(
        selected_rows,
        key=lambda r: (
            row_len(r),
            1 if bool(r.get("truncated", False)) else 0,
            str(r.get("problem_id")),
            int(r.get("gen_index", 10**9)),
        ),
    )
    keep_n = int(round(float(args.trim_fraction) * len(selected_sorted)))
    trimmed = selected_sorted[:keep_n]
    cutoff = row_len(trimmed[-1]) if trimmed else 0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selected_problem_ids.json").write_text(
        json.dumps(
            {
                "target_problem_ids": [x["problem_id"] for x in chosen],
                "target_problem_stats": chosen,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "selected_rollouts_entropy.json").write_text(
        json.dumps({"rollouts": selected_rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "selected_rollouts_trim80_entropy.json").write_text(
        json.dumps({"rollouts": trimmed}, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "selected_rollouts_trim80_stats.json").write_text(
        json.dumps(
            {
                "keep_n": keep_n,
                "cutoff_completion_tokens": cutoff,
                "trim_fraction": float(args.trim_fraction),
                "selected_problems": len(chosen),
                "selected_rollouts": len(selected_rows),
                "trimmed_rollouts": len(trimmed),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "selected_records.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "problem_id": x["problem_id"],
                        "problem_text": x["problem_text"],
                        "ground_truth": x["ground_truth"],
                        "category": x["category"],
                    }
                    for x in chosen
                ]
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(out_dir / "selected_problem_ids.json")
    print(out_dir / "selected_rollouts_trim80_entropy.json")
    print(f"selected_problems = {len(chosen)}")
    print(f"trimmed_rollouts = {len(trimmed)}")


if __name__ == "__main__":
    main()
