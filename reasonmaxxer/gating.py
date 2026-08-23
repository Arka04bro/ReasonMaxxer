"""Decision-token gating strategies.

v1 used a single rule: a generated position is a decision token iff its
teacher-forced base entropy exceeds tau. Phase 1 of the v2 study needs that rule
to become one option among several budget-matched controls, so that "entropy
finds the right positions" becomes a testable claim rather than an assumption.

All gates operate on the per-rollout list of teacher-forced entropies and return
indices into that list. Budget matching is per rollout, not aggregate: a control
gate is asked for exactly as many positions as the entropy gate produced on the
same rollout, which keeps token count identical row by row.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Sequence

GATES = (
    "entropy_tau",
    "entropy_topk",
    "entropy_quantile",
    "random",
    "low_entropy",
    "first_k",
    "last_k",
    "stride",
    "all",
)

# Gates whose budget comes from --budget_* / tau-matching rather than a threshold.
BUDGETED_GATES = ("entropy_topk", "random", "low_entropy", "first_k", "last_k", "stride")


def rollout_rng(seed: int, problem_id: str, gen_index: int) -> random.Random:
    """Deterministic per-rollout RNG, independent of iteration order."""
    key = f"{int(seed)}|{problem_id}|{int(gen_index)}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def tau_count(entropies: Sequence[float], eligible: Sequence[int], tau: float) -> int:
    """Number of positions v1 would have selected on this rollout."""
    return sum(1 for i in eligible if float(entropies[i]) > float(tau))


def resolve_budget(
    *,
    gate: str,
    entropies: Sequence[float],
    eligible: Sequence[int],
    tau: float,
    budget_k: int,
    budget_frac: float,
    budget_match_tau: bool,
) -> int:
    """Per-rollout token budget K for budgeted gates."""
    if gate not in BUDGETED_GATES:
        return len(eligible)
    if budget_match_tau:
        k = tau_count(entropies, eligible, tau)
    elif budget_k > 0:
        k = int(budget_k)
    elif budget_frac > 0:
        k = int(round(float(budget_frac) * len(eligible)))
    else:
        raise ValueError(
            f"Gate '{gate}' needs a budget: pass --budget_match_tau, --budget_k or --budget_frac."
        )
    return max(0, min(int(k), len(eligible)))


def select_positions(
    entropies: Sequence[float],
    eligible: Sequence[int],
    *,
    gate: str,
    tau: float,
    k: int,
    rng: random.Random,
    stride_offset: int = 0,
) -> list[int]:
    """Return the gated subset of `eligible` (indices into `entropies`), sorted."""
    if gate not in GATES:
        raise ValueError(f"Unsupported gate: {gate}")
    if not eligible:
        return []

    if gate == "all":
        return list(eligible)
    if gate == "entropy_tau":
        return [i for i in eligible if float(entropies[i]) > float(tau)]

    k = max(0, min(int(k), len(eligible)))
    if k == 0:
        return []

    if gate == "entropy_topk":
        # Ties broken by position so the choice is reproducible.
        ranked = sorted(eligible, key=lambda i: (-float(entropies[i]), int(i)))
        return sorted(ranked[:k])
    if gate == "low_entropy":
        ranked = sorted(eligible, key=lambda i: (float(entropies[i]), int(i)))
        return sorted(ranked[:k])
    if gate == "entropy_quantile":
        # k here is already the resolved count for the requested quantile.
        ranked = sorted(eligible, key=lambda i: (-float(entropies[i]), int(i)))
        return sorted(ranked[:k])
    if gate == "random":
        return sorted(rng.sample(list(eligible), k=k))
    if gate == "first_k":
        return list(eligible[:k])
    if gate == "last_k":
        return list(eligible[-k:])
    if gate == "stride":
        step = max(1, len(eligible) // k)
        picked = list(eligible[int(stride_offset) % step :: step])[:k]
        return sorted(picked)
    raise ValueError(f"Unhandled gate: {gate}")


def signal_mass(entropies: Sequence[float], positions: Sequence[int]) -> float:
    """Sum of base entropy over selected positions.

    Reported so that budget-matched controls can also be checked for
    *signal-mass* mismatch: equal token counts do not imply equal gradient mass,
    because cross-entropy correlates with base entropy.
    """
    return float(sum(float(entropies[i]) for i in positions))


def gate_diagnostics(
    entropies: Sequence[float],
    eligible: Sequence[int],
    positions: Sequence[int],
) -> dict[str, Any]:
    sel = set(int(x) for x in positions)
    sel_ent = [float(entropies[i]) for i in eligible if i in sel]
    unsel_ent = [float(entropies[i]) for i in eligible if i not in sel]
    n_elig = max(1, len(eligible))
    return {
        "num_selected": len(sel_ent),
        "num_eligible": len(eligible),
        "selected_frac": float(len(sel_ent)) / float(n_elig),
        "selected_entropy_mean": (sum(sel_ent) / len(sel_ent)) if sel_ent else 0.0,
        "unselected_entropy_mean": (sum(unsel_ent) / len(unsel_ent)) if unsel_ent else 0.0,
        "signal_mass": float(sum(sel_ent)),
        "relative_position_mean": (
            float(sum((i - eligible[0]) / n_elig for i in eligible if i in sel)) / len(sel_ent)
            if sel_ent
            else 0.0
        ),
    }
