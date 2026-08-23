"""Sampling-based accuracy metrics: pass@k, maj@k and the coverage curve.

v1 reported pass@1 only. The central claim of the v1 paper is that RL performs
*policy selection* rather than capability acquisition, and that claim is
directly testable with pass@k: selecting harder among paths the base model can
already produce should raise pass@1 while leaving pass@k at large k roughly
unchanged. If pass@k rises substantially too, the claim is in trouble. Reporting
only pass@1 leaves the paper's own hypothesis untested.

pass@k uses the unbiased estimator of Chen et al. (2021),

    pass@k = 1 - C(n - c, k) / C(n, k)

computed in a numerically stable product form, which needs n >> k samples per
problem. maj@k is estimated by resampling k-subsets, since it has no comparable
closed form.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any, Iterable, Sequence


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for one problem: n samples drawn, c of them correct."""
    n, c, k = int(n), int(c), int(k)
    if n <= 0 or k <= 0:
        return float("nan")
    if k > n:
        raise ValueError(f"pass@k needs k <= n, got k={k} n={n}")
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    # prod_{i=n-c+1}^{n} (1 - k/i) == C(n-k, c) / C(n, c) == 1 - pass@k
    prob_all_wrong = 1.0
    for i in range(n - c + 1, n + 1):
        prob_all_wrong *= 1.0 - k / i
    return 1.0 - prob_all_wrong


def maj_at_k(
    answers: Sequence[str | None],
    correct: Sequence[bool],
    k: int,
    *,
    trials: int = 1000,
    seed: int = 0,
) -> float:
    """Majority-vote accuracy over k samples, estimated by resampling k-subsets.

    A subset scores 1 when the most common non-empty answer in it is correct.
    Ties are broken by first appearance, matching the usual maj@k convention.
    """
    n = len(answers)
    if n == 0 or k <= 0:
        return float("nan")
    if k > n:
        raise ValueError(f"maj@k needs k <= n, got k={k} n={n}")

    # An answer string is judged correct if any sample carrying it was judged correct.
    correct_answers = {a for a, ok in zip(answers, correct) if ok and a is not None}
    idx = list(range(n))
    rng = random.Random(seed)
    hits = 0
    for _ in range(int(trials)):
        pick = rng.sample(idx, k=k)
        counts: Counter[str] = Counter()
        order: dict[str, int] = {}
        for rank, i in enumerate(pick):
            a = answers[i]
            if a is None or a == "":
                continue
            counts[a] += 1
            order.setdefault(a, rank)
        if not counts:
            continue
        best = min(counts.items(), key=lambda kv: (-kv[1], order[kv[0]]))[0]
        if best in correct_answers:
            hits += 1
    return float(hits) / float(trials)


def _mean_and_stderr(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return float(mean), float("nan")
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return float(mean), float((var / n) ** 0.5)


def summarize(
    rows: Iterable[dict[str, Any]],
    *,
    ks: Sequence[int] = (1, 2, 4, 8, 16),
    maj_trials: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Aggregate pass@k / maj@k over evaluation rows.

    Each row is one problem with a "generations" list, as written by
    scripts/generate_rollouts.py. Standard errors are across problems, which is
    the variation that matters when comparing checkpoints.
    """
    per_problem: list[dict[str, Any]] = []
    for row in rows:
        gens = row.get("generations") or []
        if not gens:
            continue
        flags = [bool(g.get("correct", False)) for g in gens]
        answers = [g.get("extracted_answer") for g in gens]
        per_problem.append(
            {
                "problem_id": row.get("problem_id"),
                "n": len(gens),
                "c": sum(1 for f in flags if f),
                "answers": answers,
                "correct": flags,
            }
        )

    if not per_problem:
        return {"n_problems": 0, "metrics": {}}

    n_min = min(int(p["n"]) for p in per_problem)
    usable_ks = [int(k) for k in ks if 1 <= int(k) <= n_min]
    skipped_ks = [int(k) for k in ks if int(k) > n_min]

    metrics: dict[str, Any] = {}
    for k in usable_ks:
        pk = [pass_at_k(p["n"], p["c"], k) for p in per_problem]
        mean, se = _mean_and_stderr(pk)
        metrics[f"pass@{k}"] = {"mean": mean, "stderr": se}

        mk = [
            maj_at_k(p["answers"], p["correct"], k, trials=maj_trials, seed=seed + i)
            for i, p in enumerate(per_problem)
        ]
        mmean, mse = _mean_and_stderr(mk)
        metrics[f"maj@{k}"] = {"mean": mmean, "stderr": mse}

    solved_any = sum(1 for p in per_problem if int(p["c"]) > 0)
    return {
        "n_problems": len(per_problem),
        "samples_per_problem_min": n_min,
        "samples_per_problem_max": max(int(p["n"]) for p in per_problem),
        "ks_reported": usable_ks,
        "ks_skipped_insufficient_samples": skipped_ks,
        "coverage": float(solved_any) / float(len(per_problem)),
        "metrics": metrics,
    }


def format_table(summary: dict[str, Any]) -> str:
    """Markdown table of the summary, for stdout."""
    m = summary.get("metrics", {})
    if not m:
        return "(no metrics: no generations found)"
    lines = ["| metric | mean | stderr |", "|---|---:|---:|"]
    for key in sorted(m, key=lambda x: (x.split("@")[0], int(x.split("@")[1]))):
        lines.append(f"| {key} | {m[key]['mean']:.4f} | {m[key]['stderr']:.4f} |")
    lines.append(f"| coverage (solved by any sample) | {summary.get('coverage', float('nan')):.4f} | |")
    return "\n".join(lines)
