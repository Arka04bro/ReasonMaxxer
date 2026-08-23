#!/usr/bin/env python3
"""Pick the entropy threshold that hits a target token budget.

tau=1.4 nats is a property of Qwen2.5-1.5B's entropy distribution, not a
universal constant. A different model, a different sampling temperature or a
different problem difficulty shifts that distribution, and the same tau then
selects a wildly different fraction of positions. Comparing v1 against v2 across
model scales with a fixed tau would confound the gate with the budget.

This resolves tau from the *budget* instead: give a target fraction of generated
positions (or a target token count per rollout) and it returns the threshold
that achieves it on the scored rollouts, separately for correct and incorrect
rollouts if asked.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from pathlib import Path
from typing import Any


def load_rollouts(patterns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pattern in patterns:
        for resolved in sorted(glob.glob(pattern)):
            payload = json.loads(Path(resolved).read_text(encoding="utf-8"))
            shard = payload.get("rollouts", []) if isinstance(payload, dict) else payload
            if not isinstance(shard, list):
                raise ValueError(f"Unsupported rollout file: {resolved}")
            rows.extend(shard)
    return rows


def quantile_threshold(values: list[float], keep_fraction: float) -> float:
    """Smallest tau such that at most keep_fraction of values exceed it."""
    if not values:
        return float("nan")
    keep = max(0.0, min(1.0, float(keep_fraction)))
    ordered = sorted(values)
    if keep <= 0.0:
        return float(ordered[-1])
    if keep >= 1.0:
        return float(ordered[0]) - 1e-9
    idx = int(round((1.0 - keep) * (len(ordered) - 1)))
    return float(ordered[idx])


def describe(values: list[float], tau: float) -> dict[str, Any]:
    if not values:
        return {"n_positions": 0}
    above = [v for v in values if v > tau]
    return {
        "n_positions": len(values),
        "tau": float(tau),
        "selected_fraction": len(above) / len(values),
        "entropy_mean": float(statistics.mean(values)),
        "entropy_median": float(statistics.median(values)),
        "entropy_p90": float(sorted(values)[int(0.90 * (len(values) - 1))]),
        "entropy_p99": float(sorted(values)[int(0.99 * (len(values) - 1))]),
        "selected_entropy_mean": float(statistics.mean(above)) if above else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Calibrate the entropy threshold to a token budget.")
    p.add_argument("--input", nargs="+", required=True, help="Scored rollout JSONs or globs.")
    p.add_argument(
        "--target_fraction",
        type=float,
        default=0.0,
        help="Target fraction of generated positions to select (mutually exclusive with --target_tokens_per_rollout).",
    )
    p.add_argument("--target_tokens_per_rollout", type=float, default=0.0)
    p.add_argument(
        "--separate_signs",
        action="store_true",
        help="Calibrate tau_pos and tau_neg separately on correct and incorrect rollouts.",
    )
    p.add_argument("--output", default="", help="Optional JSON path for the calibration report.")
    args = p.parse_args()

    if (args.target_fraction > 0) == (args.target_tokens_per_rollout > 0):
        raise SystemExit("Pass exactly one of --target_fraction or --target_tokens_per_rollout.")

    rows = load_rollouts(args.input)
    if not rows:
        raise SystemExit("No rollouts loaded.")

    pos_ent: list[float] = []
    neg_ent: list[float] = []
    lengths: list[int] = []
    for r in rows:
        ent = [float(x) for x in (r.get("entropies") or [])]
        if not ent:
            continue
        lengths.append(len(ent))
        (pos_ent if bool(r.get("correct", False)) else neg_ent).extend(ent)

    all_ent = pos_ent + neg_ent
    if not all_ent:
        raise SystemExit("Rollouts contain no entropies; run scripts/score_rollouts.py first.")

    mean_len = statistics.mean(lengths)
    if args.target_tokens_per_rollout > 0:
        target_fraction = float(args.target_tokens_per_rollout) / float(mean_len)
        if target_fraction > 1.0:
            raise SystemExit(
                f"Target of {args.target_tokens_per_rollout} tokens exceeds the mean rollout length {mean_len:.1f}."
            )
    else:
        target_fraction = float(args.target_fraction)

    report: dict[str, Any] = {
        "n_rollouts": len(lengths),
        "mean_generated_positions": float(mean_len),
        "median_generated_positions": float(statistics.median(lengths)),
        "target_fraction": target_fraction,
        "target_tokens_per_rollout": target_fraction * float(mean_len),
    }

    if args.separate_signs and pos_ent and neg_ent:
        tau_pos = quantile_threshold(pos_ent, target_fraction)
        tau_neg = quantile_threshold(neg_ent, target_fraction)
        report["tau_pos"] = tau_pos
        report["tau_neg"] = tau_neg
        report["correct_rollouts"] = describe(pos_ent, tau_pos)
        report["incorrect_rollouts"] = describe(neg_ent, tau_neg)
    else:
        tau = quantile_threshold(all_ent, target_fraction)
        report["tau_pos"] = tau
        report["tau_neg"] = tau
        report["pooled"] = describe(all_ent, tau)
        if pos_ent and neg_ent:
            # Same threshold, different distributions: this asymmetry is itself
            # informative, since incorrect rollouts tend to run hotter.
            report["correct_rollouts_at_pooled_tau"] = describe(pos_ent, tau)
            report["incorrect_rollouts_at_pooled_tau"] = describe(neg_ent, tau)

    print(json.dumps(report, indent=2))
    print(
        f"\n--tau_pos {report['tau_pos']:.4f} --tau_neg {report['tau_neg']:.4f}"
        f"   (~{report['target_tokens_per_rollout']:.1f} tokens per rollout)"
    )
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[saved] {out}")


if __name__ == "__main__":
    main()
