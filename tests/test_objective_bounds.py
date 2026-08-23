#!/usr/bin/env python3
"""Boundedness of the decision-token objectives.

v1 (`adv_ce`) uses A * CE. For a negative advantage this is minimised by driving
the token probability to zero, and the loss has no lower bound: it falls
linearly in log-probability forever. v1 carries three separate patches around
this (advantage clipping, a probability floor for negatives, a decaying
beta_neg), which is what a missing bound usually looks like in code.

v2 (`clipped_ratio`) uses the clipped off-policy surrogate. Plain PPO clipping
already fixes the lower bound, but for A < 0 the unclipped branch is the one
that survives min(), so the loss still grows linearly in the ratio and the
gradient can blow up. Dual clipping (Ye et al., 2020) floors the surrogate at
c*A and bounds it on both sides.

Run: python tests/test_objective_bounds.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


def v1_loss(log_ratio: torch.Tensor, adv: float, base_logp: float = -2.0) -> torch.Tensor:
    """A * CE, as in decision_objective=adv_ce."""
    student_logp = base_logp + log_ratio
    return adv * (-student_logp)


def v2_loss(
    log_ratio: torch.Tensor,
    adv: float,
    *,
    eps_low: float = 0.2,
    eps_high: float = 0.2,
    dual_clip_c: float = 3.0,
) -> torch.Tensor:
    """Mirror of the clipped_ratio branch in scripts/train_reasonmaxxer.py."""
    ratio = torch.exp(log_ratio.clamp(min=-20.0, max=20.0))
    surrogate = torch.min(ratio * adv, ratio.clamp(1.0 - eps_low, 1.0 + eps_high) * adv)
    if dual_clip_c > 1.0 and adv < 0:
        surrogate = surrogate.clamp_min(dual_clip_c * adv)
    return -surrogate


def grad_at(fn, x: float) -> float:
    t = torch.tensor([x], requires_grad=True)
    fn(t).sum().backward()
    return float(t.grad.detach())


def main() -> None:
    adv_neg, adv_pos = -1.0, 1.0
    eps, c = 0.2, 3.0
    extreme = torch.tensor([-50.0, -20.0, -5.0, 0.0, 5.0, 20.0, 50.0])

    print("A = -1 (the rollout was wrong: push this token down)")
    print(f"{'log rho':>8} {'rho':>10} {'v1 loss':>9} {'v1 grad':>8} {'v2 loss':>9} {'v2 grad':>8}")
    for lr in (-8.0, -4.0, -0.5, 0.0, 0.5, 2.0, 6.0):
        l1 = float(v1_loss(torch.tensor([lr]), adv_neg).detach())
        l2 = float(v2_loss(torch.tensor([lr]), adv_neg).detach())
        g1 = grad_at(lambda t: v1_loss(t, adv_neg), lr)
        g2 = grad_at(lambda t: v2_loss(t, adv_neg), lr)
        print(f"{lr:8.2f} {float(torch.exp(torch.tensor(lr))):10.3f} {l1:9.3f} {g1:8.3f} {l2:9.3f} {g2:8.3f}")

    # v1 has no lower bound: the further the token is pushed down, the lower the loss.
    assert float(v1_loss(torch.tensor([-50.0]), adv_neg)) < float(v1_loss(torch.tensor([-8.0]), adv_neg))
    assert abs(grad_at(lambda t: v1_loss(t, adv_neg), -50.0) - 1.0) < 1e-6, "v1 gradient never vanishes"

    # Plain PPO clipping (no dual clip) is bounded below but not above for A < 0.
    no_dual = v2_loss(extreme, adv_neg, dual_clip_c=0.0)
    assert float(no_dual.min()) >= (1.0 - eps) * abs(adv_neg) - 1e-6
    assert float(no_dual.max()) > 100.0, "expected the ratio branch to grow without a dual clip"

    # With dual clipping the loss is bounded on both sides for either advantage sign.
    bound = max(1.0 + eps, c)
    for a in (adv_neg, adv_pos):
        vals = v2_loss(extreme, a)
        assert bool((vals.abs() <= bound * abs(a) + 1e-6).all()), f"v2 unbounded for A={a}: {vals}"

    # Gradient vanishes once the ratio is outside the trust region in the correcting direction.
    assert abs(grad_at(lambda t: v2_loss(t, adv_neg), -3.0)) < 1e-6, "A<0, ratio already small"
    assert abs(grad_at(lambda t: v2_loss(t, adv_neg), 6.0)) < 1e-6, "A<0, past the dual-clip floor"
    assert abs(grad_at(lambda t: v2_loss(t, adv_pos), 3.0)) < 1e-6, "A>0, ratio already large"
    assert grad_at(lambda t: v2_loss(t, adv_pos), 0.0) < 0.0, "A>0 near rho=1 should still push up"

    print()
    print(f"v1  A<0 : unbounded below, gradient constant at {1.0:.1f} forever")
    print(f"v2  A<0 : loss in [{float(v2_loss(extreme, adv_neg).min()):.2f}, "
          f"{float(v2_loss(extreme, adv_neg).max()):.2f}], zero gradient outside the region")
    print(f"v2  A>0 : loss in [{float(v2_loss(extreme, adv_pos).min()):.2f}, "
          f"{float(v2_loss(extreme, adv_pos).max()):.2f}]")
    print()
    print("PASS")


if __name__ == "__main__":
    main()
