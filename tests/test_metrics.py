#!/usr/bin/env python3
"""Checks for the pass@k / maj@k estimators.

The unbiasedness check matters: the naive estimator (fraction of k-subsets that
contain a correct sample, computed on one draw) is biased, and pass@k is the
metric the policy-selection claim is tested with.

Run: python tests/test_metrics.py
"""

from __future__ import annotations

import random
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reasonmaxxer.metrics import maj_at_k, pass_at_k, summarize


def brute_force_pass_at_k(n: int, c: int, k: int) -> float:
    """Exact probability that a uniformly drawn k-subset contains a correct sample."""
    items = [True] * c + [False] * (n - c)
    subsets = list(combinations(range(n), k))
    hits = sum(1 for s in subsets if any(items[i] for i in s))
    return hits / len(subsets)


def main() -> None:
    for n in range(1, 13):
        for c in range(0, n + 1):
            for k in range(1, n + 1):
                got = pass_at_k(n, c, k)
                want = brute_force_pass_at_k(n, c, k)
                assert abs(got - want) < 1e-9, f"pass@k mismatch n={n} c={c} k={k}: {got} vs {want}"
    print("pass@k matches exact enumeration for all n<=12")

    assert pass_at_k(16, 0, 8) == 0.0
    assert pass_at_k(16, 16, 8) == 1.0
    assert abs(pass_at_k(16, 1, 1) - 1 / 16) < 1e-12
    assert pass_at_k(16, 2, 8) > pass_at_k(16, 2, 4) > pass_at_k(16, 2, 1)
    print("pass@k monotone in k and correct at the boundaries")

    # Empirical unbiasedness: average the estimator over independent draws of n
    # samples from a Bernoulli(p) and compare with the true 1-(1-p)^k.
    rng = random.Random(0)
    n, k, p, reps = 16, 4, 0.3, 20000
    est = 0.0
    for _ in range(reps):
        c = sum(1 for _ in range(n) if rng.random() < p)
        est += pass_at_k(n, c, k)
    est /= reps
    truth = 1.0 - (1.0 - p) ** k
    assert abs(est - truth) < 0.01, f"estimator biased: {est:.4f} vs {truth:.4f}"
    print(f"empirical unbiasedness: estimate={est:.4f} truth={truth:.4f}")

    # maj@k: a wrong answer held by a majority should beat a correct minority.
    answers = ["7", "7", "7", "42", "42"]
    correct = [False, False, False, True, True]
    assert maj_at_k(answers, correct, 5, trials=200) == 0.0
    answers = ["42", "42", "42", "7", "7"]
    correct = [True, True, True, False, False]
    assert maj_at_k(answers, correct, 5, trials=200) == 1.0
    mid = maj_at_k(["42", "42", "7", "7", "9"], [True, True, False, False, False], 3, trials=4000, seed=1)
    assert 0.0 < mid < 1.0, mid
    print(f"maj@k sane: unanimous cases exact, mixed case = {mid:.3f}")

    # End-to-end shape of the summary, including the k > n guard.
    rows = [
        {"problem_id": "a", "generations": [{"correct": True, "extracted_answer": "1"}] * 2
         + [{"correct": False, "extracted_answer": "2"}] * 2},
        {"problem_id": "b", "generations": [{"correct": False, "extracted_answer": "3"}] * 4},
    ]
    s = summarize(rows, ks=(1, 2, 4, 8), maj_trials=200)
    assert s["n_problems"] == 2 and s["ks_reported"] == [1, 2, 4] and s["ks_skipped_insufficient_samples"] == [8]
    assert abs(s["metrics"]["pass@1"]["mean"] - 0.25) < 1e-9
    assert abs(s["metrics"]["pass@4"]["mean"] - 0.5) < 1e-9
    assert abs(s["coverage"] - 0.5) < 1e-9
    print("summarize(): pass@1=0.2500 pass@4=0.5000 coverage=0.5000, k=8 skipped")

    print()
    print("PASS")


if __name__ == "__main__":
    main()
