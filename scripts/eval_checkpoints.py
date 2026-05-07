from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _pass1(path: Path) -> tuple[float, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results", []) or []
    vals: list[float] = []
    for r in rows:
        gens = r.get("generations", []) or []
        if len(gens) == 0:
            continue
        c = sum(1 for g in gens if bool(g.get("correct", False)))
        vals.append(float(c / len(gens)))
    if not vals:
        return float("nan"), len(rows)
    return float(sum(vals) / len(vals)), len(rows)


def _list_ckpts(run_dir: Path) -> list[Path]:
    ckpts: list[Path] = []
    for p in run_dir.iterdir():
        if not p.is_dir():
            continue
        if p.name.startswith("epochf_") or p.name.startswith("epoch_"):
            ckpts.append(p)
    ckpts = sorted(ckpts, key=lambda x: x.name.replace("epochf_", "epoch_"))
    return ckpts


def _update_metrics_csv(metrics_csv: Path, tag_to_pass: dict[str, float], out_csv: Path) -> None:
    rows: list[dict[str, Any]] = []
    with metrics_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fields = list(r.fieldnames or [])
        for row in r:
            tag = str(row.get("checkpoint_tag", "")).strip()
            if tag in tag_to_pass:
                row["holdout60_pass1"] = f"{float(tag_to_pass[tag]):.8f}"
            rows.append(row)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser(description="Eval all epoch checkpoints on a fixed holdout set and summarize.")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--max_lora_rank", type=int, default=32)
    p.add_argument("--holdout_ids_file", default=None)
    p.add_argument("--holdout_ids_key", default="problem_ids")
    p.add_argument("--records_file", default=None)
    p.add_argument("--dataset", default="math500")
    p.add_argument("--problem_set", default="full")
    p.add_argument("--max_problems", type=int, default=60)
    p.add_argument("--output_tag", default="holdout")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_generations", type=int, default=1)
    p.add_argument("--device", default="0")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=16384)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--match_timeout_s", type=float, default=0.2)
    p.add_argument("--prompt_style", default="auto")
    p.add_argument("--stop_profile", default="auto")
    p.add_argument("--math500_extractor", default="auto")
    p.add_argument(
        "--qwen3_enable_thinking",
        default="auto",
        choices=["auto", "true", "false"],
        help="Forwarded to generate_rollouts.py when prompt_style=qwen3_chat/chat_template.",
    )
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--update_metrics_csv", action="store_true")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpts = _list_ckpts(run_dir)
    if not ckpts:
        raise ValueError(f"No epoch checkpoints found in {run_dir}")

    results: list[dict[str, Any]] = []
    tag_to_pass: dict[str, float] = {}

    for ckpt in ckpts:
        tag = ckpt.name
        out_json = out_dir / f"{run_dir.name}_{tag}_{args.dataset}_{args.output_tag}_n{int(args.num_generations)}.json"
        if out_json.exists() and args.skip_existing and not args.force:
            p1, rows = _pass1(out_json)
            results.append(
                {
                    "checkpoint_tag": tag,
                    "checkpoint_path": str(ckpt),
                    "output_json": str(out_json),
                    "rows": int(rows),
                    "pass1": float(p1),
                }
            )
            tag_to_pass[tag] = float(p1)
            continue

        cmd = [
            sys.executable,
            "generate_rollouts.py",
            "--model_path",
            args.base_model,
            "--lora_adapter",
            str(ckpt),
            "--max_lora_rank",
            str(int(args.max_lora_rank)),
            "--lora_name",
            f"{run_dir.name}_{tag}",
            "--condition_name",
            f"{run_dir.name}_{tag}",
            "--prompt_condition",
            "base",
            "--dataset",
            str(args.dataset),
            "--problem_set",
            str(args.problem_set),
            "--max_problems",
            str(int(args.max_problems)),
            "--num_generations",
            str(int(args.num_generations)),
            "--batch_size",
            str(int(args.batch_size)),
            "--tensor_parallel_size",
            "1",
            "--temperature",
            str(float(args.temperature)),
            "--top_p",
            str(float(args.top_p)),
            "--max_tokens",
            str(int(args.max_tokens)),
            "--seed",
            str(int(args.seed)),
            "--match_timeout_s",
            str(float(args.match_timeout_s)),
            "--prompt_style",
            str(args.prompt_style),
            "--stop_profile",
            str(args.stop_profile),
            "--math500_extractor",
            str(args.math500_extractor),
            "--qwen3_enable_thinking",
            str(args.qwen3_enable_thinking),
            "--output_dir",
            str(out_dir),
            "--output_name",
            str(out_json.name),
        ]
        if args.records_file:
            cmd.extend(["--records_file", str(args.records_file)])
        if args.holdout_ids_file:
            cmd.extend(["--problem_ids_file", str(args.holdout_ids_file), "--problem_ids_key", str(args.holdout_ids_key)])
        if args.force:
            cmd.append("--force")

        env = dict(os.environ)
        device_arg = str(args.device).strip()
        # Respect caller-provided CUDA_VISIBLE_DEVICES for multi-process launches
        # (e.g., one parent process per physical GPU). Only set a fallback when
        # the caller did not pin visibility.
        if "CUDA_VISIBLE_DEVICES" not in env:
            env["CUDA_VISIBLE_DEVICES"] = device_arg
        # Compatibility path: allow explicit non-zero/non-auto device override
        # even when CUDA_VISIBLE_DEVICES is already present.
        elif device_arg not in {"", "0", "auto", "cuda"}:
            env["CUDA_VISIBLE_DEVICES"] = device_arg
        subprocess.run(cmd, check=True, cwd=str(Path.cwd()), env=env)
        p1, rows = _pass1(out_json)
        results.append(
            {
                "checkpoint_tag": tag,
                "checkpoint_path": str(ckpt),
                "output_json": str(out_json),
                "rows": int(rows),
                "pass1": float(p1),
            }
        )
        tag_to_pass[tag] = float(p1)

    summary_csv = out_dir / "holdout60_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["checkpoint_tag", "checkpoint_path", "rows", "pass1", "output_json"],
        )
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"[saved] {summary_csv}")

    if args.update_metrics_csv:
        metrics_csv = run_dir / "checkpoint_metrics.csv"
        if metrics_csv.exists():
            out_metrics = run_dir / "checkpoint_metrics.with_holdout60.csv"
            _update_metrics_csv(metrics_csv, tag_to_pass=tag_to_pass, out_csv=out_metrics)
            print(f"[saved] {out_metrics}")

    print("| checkpoint | pass@1 | rows |")
    print("|---|---:|---:|")
    for r in sorted(results, key=lambda x: str(x["checkpoint_tag"])):
        print(f"| {r['checkpoint_tag']} | {float(r['pass1']):.4f} | {int(r['rows'])} |")


if __name__ == "__main__":
    main()
