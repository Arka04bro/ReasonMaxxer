# ReasonMaxxer2 — research plan

## Research question

ReasonMaxxer v1 shows that a sparse, entropy-gated offline update recovers much of
the benefit of RL on mathematical reasoning. It operationalises "decision point"
as "token where the base model's teacher-forced entropy exceeds tau".

Entropy answers *how uncertain is the model here*. The claim needs *how much does
this choice change the final answer*. Those come apart:

| | changes the outcome | does not change the outcome |
|---|---|---|
| **high entropy** | pivotal — the intended target | stylistic forks: "So" / "Thus" / formatting |
| **low entropy** | forced-but-fragile steps | the bulk of the sequence |

So the question this work asks is not "how do we improve ReasonMaxxer" but:

> **Is token entropy actually a good indicator of the decision points responsible
> for RL-induced reasoning gains, and what happens if we measure pivotality
> directly instead?**

Entropy is kept as a *proposal distribution* rather than discarded: measuring
pivotality everywhere is unaffordable, and entropy is a cheap candidate filter.

## Falsifiable claims

| # | Claim | Falsified if |
|---|---|---|
| C1 | Entropy gating beats budget-matched random gating | entropy ≈ random at equal K and equal signal mass |
| C2 | The effect is sparse: most of the gain comes from few tokens per rollout | the budget curve is linear in K with no knee |
| C3 | Pivotality beats entropy at equal K | pivotality ≈ entropy, or wins only by spending more compute |
| C4 | The gain is policy selection, not new capability | pass@k at large k rises as much as pass@1 |
| C5 | The bounded objective removes collapse without costing accuracy | clipped_ratio is stable but consistently less accurate than adv_ce |

C1 and C4 are the ones that can sink the whole framing, so they come first and
are run before any new method exists. A negative C1 is itself a publishable
result about v1's mechanism.

## Phases

**Phase 0 — reproduce v1.** Fixed pipeline, 3 seeds, confidence intervals, held-out
checkpoint selection only. Nothing is comparable to v1 until v1's own variance is
known.

**Phase 1 — the causal test (C1).** Same data, same objective, same number of
updates; only the gate changes:

`entropy_tau` · `random` · `low_entropy` · `first_k` · `last_k` · `stride` · `all` · `shuffled advantages`

Budgets are matched **per rollout** (`--budget_match_tau`), not in aggregate.
Token count alone is not enough: cross-entropy correlates with base entropy, so
budget-matched gates still differ in gradient mass. `gate.signal_mass` in the
stats file quantifies this (~6x spread between entropy and low-entropy on
synthetic data), and it goes in the results table next to accuracy.

**Phase 2 — the budget curve (C2).** K ∈ {1, 2, 5, 10, 20, 50, all} for the entropy
and random gates. Accuracy vs edited tokens per rollout. This is the paper's main
figure and the direct answer to *how many tokens is an RL run worth*.

**Phase 3 — pivotality (C3).** Replace the entropy criterion with an outcome-based
one:

1. *Shared-prefix Monte Carlo.* Build a token-level trie over a problem's rollouts,
   set V(prefix) = pass rate of continuations through that node, and use
   A_t = V(s_{t+1}) − V(s_t) where overlap is sufficient. Free from existing data.
2. *Targeted continuations* where the trie is too thin: take the top entropy
   candidates as proposals, sample m short continuations from each, and estimate
   the value gap directly.

Measure the trie's coverage **before** building on it: if rollouts diverge after a
few dozen tokens, shared prefixes cover a small fraction of positions and the trie
only calibrates the probe estimator. That measurement is a figure either way.

Probe cost (K candidates × m continuations × problems × rollouts) is counted in
the compute column. Without that the "cheaper than RL" claim does not survive
review.

**Phase 4 — objective (C5).** 2×2: {entropy, pivotality} × {adv_ce, clipped_ratio},
separating better credit assignment from better optimisation. Tracked: accuracy,
KL(base‖policy), token-probability collapse, entropy drift, training stability.

**Throughout — C4.** pass@k and maj@k with n=16, unbiased estimator, base and
tuned model at identical temperature.

## Controls and confounds

- **Signal mass**, as above.
- **Length.** Truncated rollouts count as incorrect and `--trim_fraction 0.8`
  drops the longest fifth, so the method is biased toward shorter outputs. Report
  token counts and truncation rates; a length-matched comparison is required.
- **Positional.** `first_k` / `last_k` isolate a position effect; the gate
  diagnostics log mean relative position of selected tokens.
- **Advantage shuffling** breaks the outcome association while keeping the
  marginal — any surviving gain is not credit assignment.
- **Own RL baseline.** One in-house GRPO run on the same problems and base model.
  Published SimpleRL-Zoo numbers are not comparable on the compute axis.
- **Checkpoint selection** on held-out data only, never on the benchmark suite.

## Compute

Pilot scale (free tier): Qwen2.5-0.5B, 4096 tokens, 100 problems × 16 rollouts,
evaluation on GSM8K + a MATH500 subset at n=16. About 1.5–3 T4-hours per
condition; Phase 0+1 fits in roughly 30 GPU-hours.

Because mid-difficulty selection needs pass rates near 0.5, the pilot shifts the
*difficulty band* (GSM8K / MATH L1–3) rather than the selection rule — a 0.5B
model has a near-zero pass rate on MATH L3–5 and no mid pool would form.

Paper scale: Qwen2.5-1.5B and one larger family member, ~200 A100-hours total.

## Implementation status

| Component | State |
|---|---|
| `reasonmaxxer/gating.py` — 9 gates, per-rollout budget matching, diagnostics | done |
| `--shuffle_advantages` control | done |
| `decision_objective=clipped_ratio` with dual clipping | done |
| `--behavior_temperature` (ratio against pi_base^(1/T)) | done |
| `--kl_decision_weight` (trust region at decision tokens) | done |
| `reasonmaxxer/metrics.py` — pass@k / maj@k / coverage | done |
| CPU fixture + objective and metric tests | done |
| Precomputed base log-probs (drop the base model from the training loop) | todo |
| Pivotality estimator (trie + targeted continuations) | todo |
| Pilot configs and runner | todo |
| In-house GRPO baseline | todo |

Defaults everywhere reproduce v1; every v2 behaviour is opt-in.
