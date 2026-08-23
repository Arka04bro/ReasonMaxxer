# 0.5B pilot — Phase 0 and Phase 1 on free-tier hardware

This recipe runs the two phases that do not require any new method: reproducing
v1 with error bars, and the budget-matched gate controls that test whether
entropy actually finds the decision points. It is sized for a single 16 GB T4,
which is what Kaggle gives away 30 hours of per week.

## What differs from the 1.5B recipe, and why

| | 1.5B recipe | pilot | reason |
|---|---|---|---|
| Model | Qwen2.5-1.5B | Qwen2.5-0.5B | fits a T4 with room for the frozen base model |
| Precision | bf16 | fp16 | Turing has no bf16 |
| Problems | MATH L3–5 | GSM8K train | a 0.5B base solves almost no L3–5 problems, so no mid-difficulty pool forms |
| max_tokens | 8192 | 1024 | GSM8K solutions are short; the single biggest cost lever |
| Rollouts | 300 × 20 | 100 × 16 | 16 samples still supports pass@k up to k=8 |
| tau | fixed 1.4 nats | calibrated to a token budget | 1.4 is a property of the 1.5B entropy distribution and does not transfer |
| Length trimming | drop longest 20% | none | the pilot measures the length confound instead of baking it in |

The difficulty *band* moves; the selection rule does not. Mid-difficulty
selection still keeps problems whose pass rate is near 0.5.

## Run order

```bash
bash examples/qwen25_0p5b_pilot/01_sample.sh            # GSM8K train/holdout, disjoint
bash examples/qwen25_0p5b_pilot/02_generate_score.sh    # 100 x 16 rollouts + entropy
bash examples/qwen25_0p5b_pilot/03_calibrate_and_pool.sh
bash examples/qwen25_0p5b_pilot/07_eval_base.sh         # base pass@k reference
bash examples/qwen25_0p5b_pilot/04_phase0_baseline.sh   # v1, 3 seeds
bash examples/qwen25_0p5b_pilot/05_phase1_gates.sh      # 6 gates + shuffled advantages, 3 seeds
```

Then, per run, pick a checkpoint on the holdout and only afterwards score it on
the test split:

```bash
RUN=phase1_random_seed42 STAGE=holdout bash examples/qwen25_0p5b_pilot/06_eval_passk.sh
RUN=phase1_random_seed42 STAGE=test CKPT_TAG=epochf_1.00 bash examples/qwen25_0p5b_pilot/06_eval_passk.sh
```

Step 02 is the only one that needs the full rollout budget; every later
condition reuses its output. Do not rerun it between conditions.

## Rough budget

| Step | T4-hours |
|---|---|
| 02 generate + score (once) | 1–2 |
| 07 base pass@k reference | 0.5 |
| 04 Phase 0, 3 seeds | 2–3 |
| 05 Phase 1, 7 conditions × 3 seeds | 12–18 |
| 06 evaluation, holdout + test | 6–10 |
| **total** | **~25–35** |

That is one to two weeks of the Kaggle free tier. These are estimates from
parameter counts and sequence lengths, not measurements; recalibrate after
step 02 finishes.

## Kaggle notes

- Kaggle sessions cap at ~12 hours and the filesystem is wiped afterwards. Set
  `ROOT=/kaggle/working/pilot` and save that directory as a Kaggle Dataset at
  the end of each session, then mount it as input in the next one. Every script
  is restartable and skips completed work.
- Two T4s are available. Steps 04 and 05 are independent per condition, so run
  two shells with `CUDA_VISIBLE_DEVICES=0` and `=1` over disjoint seed lists.
- vLLM needs `--dtype float16` on Turing; `00_config.sh` already sets it.
- Install with `pip install -r requirements.txt` but keep the preinstalled
  torch: `pip install --no-deps -r requirements.txt` first, then add whatever
  import errors reveal.

## Reading the output

`05_phase1_gates.sh` prints a budget table at the end:

```
run                                    tok/roll   match   H_sel       mass  relpos
phase1_entropy_tau_seed42                 10.00   1.000   1.848      568.3    0.50
phase1_random_seed42                      10.00   1.000   1.009      315.0    0.47
phase1_low_entropy_seed42                 10.00   1.000   0.271       90.2    0.48
```

`match` near 1.000 confirms the per-rollout budgets are identical. `mass` is the
summed base entropy over selected positions and it is *not* matched — it cannot
be, since that is the very thing the entropy gate selects for. Report it beside
accuracy so the comparison stays honest: part of any entropy advantage is more
gradient mass, not better position selection. (The numbers above are from a
synthetic smoke test, not from real rollouts.)

## What this pilot cannot settle

A 0.5B model on GSM8K is not evidence about 1.5B–7B models on competition math.
The pilot is powered to answer one question — does entropy gating beat
budget-matched controls at all — and to produce the figure that makes a compute
request credible. Phase 2 onward needs real hardware.
