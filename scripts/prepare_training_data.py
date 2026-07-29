from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any



def _bucket_name(pass_rate: float) -> str:
    if pass_rate < 0.25:
        return "[0.00,0.25)"
    if pass_rate < 0.50:
        return "[0.25,0.50)"
    if pass_rate < 0.75:
        return "[0.50,0.75)"
    return "[0.75,1.00]"


def _load_rollouts(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {}, payload
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported rollout file format: {path}")
    rows = payload.get("rollouts")
    if not isinstance(rows, list):
        raise ValueError(f"Missing 'rollouts' list in: {path}")
    return payload.get("meta", {}), rows


def _safe_std(xs: list[float], eps: float) -> float:
    if not xs:
        return eps
    if len(xs) == 1:
        return eps
    value = statistics.pstdev(xs)
    return float(value if value > eps else eps)


def _select_targets(
    by_problem: dict[str, list[dict[str, Any]]],
    *,
    min_pass: float,
    max_pass: float,
    max_targets: int,
    seed: int,
    strategy: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    infos: list[dict[str, Any]] = []
    for pid, rows in by_problem.items():
        n = len(rows)
        c = int(sum(1 for r in rows if bool(r.get("correct", False))))
        p = (c / n) if n > 0 else 0.0
        infos.append(
            {
                "problem_id": pid,
                "num_rollouts": int(n),
                "num_correct": int(c),
                "pass_rate": float(p),
                "bucket": _bucket_name(float(p)),
            }
        )

    eligible = [x for x in infos if float(min_pass) <= float(x["pass_rate"]) <= float(max_pass)]
    if not eligible:
        return [], infos

    if max_targets <= 0 or len(eligible) <= max_targets:
        chosen = eligible
    elif strategy == "random":
        rng = random.Random(int(seed))
        chosen = rng.sample(eligible, k=int(max_targets))
    else:
        # Default: keep problems closest to 0.5 pass-rate (strongest contrastive signal).
        chosen = sorted(
            eligible,
            key=lambda x: (abs(float(x["pass_rate"]) - 0.5), -int(x["num_rollouts"]), str(x["problem_id"])),
        )[: int(max_targets)]

    chosen_ids = [str(x["problem_id"]) for x in chosen]
    return chosen_ids, infos


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare ReasonMaxxer training data from base entropy rollouts.")
    p.add_argument("--rollouts_file", default="data/contrastive_experiments/base_rollouts_math500_train_n40.json")
    p.add_argument("--tau", type=float, default=0.8, help="Legacy symmetric entropy threshold.")
    p.add_argument(
        "--tau_pos",
        type=float,
        default=None,
        help="Entropy threshold for decision tokens from correct rollouts (default: --tau).",
    )
    p.add_argument(
        "--tau_neg",
        type=float,
        default=None,
        help="Entropy threshold for decision tokens from incorrect rollouts (default: --tau_pos).",
    )
    p.add_argument("--min_pass", type=float, default=0.25)
    p.add_argument("--max_pass", type=float, default=0.75)
    p.add_argument("--max_target_problems", type=int, default=50)
    p.add_argument("--selection_strategy", choices=["closest_midpoint", "random"], default="closest_midpoint")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--adv_eps", type=float, default=1e-8)
    p.add_argument(
        "--positive_only",
        action="store_true",
        help=(
            "Keep only correct rollouts and assign advantage=1.0 to each retained rollout. "
            "Use this with decision_objective=correct_sft for strict positive-only training."
        ),
    )
    p.add_argument(
        "--code_decision_scope",
        choices=["full_completion", "extracted_code"],
        default="full_completion",
        help="For code rollouts, restrict decision-token supervision to the extracted executable code span.",
    )
    p.add_argument("--target_ids_output", default="data/contrastive_experiments/target_problem_ids.json")
    p.add_argument("--processed_output", default="data/contrastive_experiments/training_data_processed.json")
    p.add_argument("--training_examples_output", default="data/contrastive_experiments/training_examples.json")
    p.add_argument("--stats_output", default="data/contrastive_experiments/training_stats.json")
    p.add_argument(
        "--selected_records_output",
        default="data/contrastive_experiments/target_train_records.json",
        help="Records JSON for evaluating in-distribution train problems with generate_rollouts.py --records_file.",
    )
    p.add_argument(
        "--curriculum_order_output",
        default="",
        help=(
            "Optional path to save curriculum ordering metadata "
            "(signal_clarity sorted rollout references)."
        ),
    )
    args = p.parse_args()

    tau_pos = float(args.tau if args.tau_pos is None else args.tau_pos)
    tau_neg = float(tau_pos if args.tau_neg is None else args.tau_neg)

    rollouts_path = Path(args.rollouts_file)
    meta, rollouts = _load_rollouts(rollouts_path)
    if not rollouts:
        raise ValueError(f"No rollouts found in {rollouts_path}")

    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rollouts:
        pid = str(r.get("problem_id", ""))
        if not pid:
            continue
        by_problem[pid].append(r)
    if not by_problem:
        raise ValueError("No problem_id entries found in rollout rows.")

    target_ids, all_infos = _select_targets(
        by_problem,
        min_pass=float(args.min_pass),
        max_pass=float(args.max_pass),
        max_targets=int(args.max_target_problems),
        seed=int(args.seed),
        strategy=str(args.selection_strategy),
    )
    target_id_set = set(target_ids)

    info_by_pid = {str(x["problem_id"]): x for x in all_infos}
    target_infos = [info_by_pid[pid] for pid in target_ids if pid in info_by_pid]
    bucket_counts = Counter(x["bucket"] for x in all_infos)
    target_bucket_counts = Counter(x["bucket"] for x in target_infos)

    target_payload = {
        "source_rollouts_file": str(rollouts_path),
        "selection": {
            "min_pass": float(args.min_pass),
            "max_pass": float(args.max_pass),
            "max_target_problems": int(args.max_target_problems),
            "selection_strategy": args.selection_strategy,
            "seed": int(args.seed),
        },
        "num_total_problems": len(by_problem),
        "num_target_problems": len(target_ids),
        "bucket_counts_all": dict(bucket_counts),
        "bucket_counts_target": dict(target_bucket_counts),
        "target_problem_ids": target_ids,
        "target_problem_stats": target_infos,
        "all_problem_stats": all_infos,
    }

    target_path = Path(args.target_ids_output)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(target_payload, indent=2) + "\n", encoding="utf-8")

    processed_rollouts: list[dict[str, Any]] = []
    training_examples: list[dict[str, Any]] = []

    decision_counts: list[int] = []
    decision_frac: list[float] = []
    decision_entropies: list[float] = []
    advantages: list[float] = []
    label_counts = Counter()
    pos_decision_tokens_total = 0
    neg_decision_tokens_total = 0

    for pid in target_ids:
        rows = by_problem.get(pid, [])
        if not rows:
            continue
        rewards = [1.0 if bool(r.get("correct", False)) else 0.0 for r in rows]
        mean_r = float(sum(rewards) / len(rewards))
        std_r = _safe_std(rewards, eps=float(args.adv_eps))

        for r in rows:
            correct = bool(r.get("correct", False))
            if args.positive_only and not correct:
                continue

            input_ids = r.get("tokens") or r.get("input_ids") or []
            if not isinstance(input_ids, list) or len(input_ids) < 2:
                continue
            input_ids = [int(x) for x in input_ids]

            prompt_length = int(r.get("prompt_length", 0))
            prompt_length = max(1, min(prompt_length, len(input_ids)))

            entropies = r.get("entropies") or []
            entropies = [float(x) for x in entropies]
            gen_pred_start = max(prompt_length - 1, 0)
            max_pred_index = len(input_ids) - 1

            decision_positions: list[int] = []
            thr = float(tau_pos if correct else tau_neg)
            code_span_start = r.get("code_span_start_in_full")
            code_span_end = r.get("code_span_end_in_full")
            for i, ent in enumerate(entropies):
                pred_pos = gen_pred_start + i
                if pred_pos >= max_pred_index:
                    break
                if float(ent) > thr:
                    if (
                        args.code_decision_scope == "extracted_code"
                        and code_span_start is not None
                        and code_span_end is not None
                        and not (int(code_span_start) <= int(pred_pos) < int(code_span_end))
                    ):
                        continue
                    decision_positions.append(int(pred_pos))
                    decision_entropies.append(float(ent))

            reward = 1.0 if correct else 0.0
            if args.positive_only:
                advantage = 1.0
            else:
                advantage = float((reward - mean_r) / std_r)

            completion_length = int(r.get("num_completion_tokens", len(entropies)))
            completion_length = max(0, min(completion_length, len(input_ids) - prompt_length))
            num_pred = len(input_ids) - 1

            processed_rollout = {
                "problem_id": pid,
                "gen_index": int(r.get("gen_index", 0)),
                "input_ids": input_ids,
                "prompt_length": int(prompt_length),
                "completion_length": int(completion_length),
                "correct": correct,
                "reward": float(reward),
                "advantage": float(advantage),
                "entropies": entropies,
                "generation_pred_start": int(gen_pred_start),
                "decision_positions": decision_positions,
                "num_tokens": int(len(input_ids)),
                "num_pred_tokens": int(num_pred),
                "source_seed": (None if r.get("seed", None) is None else int(r.get("seed"))),
                "tau_pos": float(tau_pos),
                "tau_neg": float(tau_neg),
                "code_span_found": bool(r.get("code_span_found", False)),
                "code_span_start_in_full": code_span_start,
                "code_span_end_in_full": code_span_end,
            }
            processed_rollouts.append(processed_rollout)

            lbl = "+1" if correct else "-1"
            label_counts[lbl] += 1
            advantages.append(float(advantage))

            dc = len(decision_positions)
            decision_counts.append(dc)
            if correct:
                pos_decision_tokens_total += int(dc)
            else:
                neg_decision_tokens_total += int(dc)
            if num_pred > 0:
                decision_frac.append(float(dc / num_pred))

            for pos in decision_positions:
                training_examples.append(
                    {
                        "problem_id": pid,
                        "gen_index": int(r.get("gen_index", 0)),
                        "position": int(pos),
                        "target_token": int(input_ids[pos + 1]),
                        "label": 1 if correct else -1,
                        "entropy": float(entropies[pos - gen_pred_start]) if 0 <= (pos - gen_pred_start) < len(entropies) else None,
                        "advantage": float(advantage),
                    }
                )

    processed_payload = {
        "rollouts": processed_rollouts,
        "config": {
            "tau": float(args.tau),
            "tau_pos": float(tau_pos),
            "tau_neg": float(tau_neg),
            "min_pass": float(args.min_pass),
            "max_pass": float(args.max_pass),
            "max_target_problems": int(args.max_target_problems),
            "selection_strategy": args.selection_strategy,
            "seed": int(args.seed),
            "adv_eps": float(args.adv_eps),
            "positive_only": bool(args.positive_only),
            "source_rollouts_file": str(rollouts_path),
            "source_meta": meta,
            "n_problems_total": len(by_problem),
            "n_problems_selected": len(target_ids),
            "n_rollouts_selected": len(processed_rollouts),
            "code_decision_scope": str(args.code_decision_scope),
        },
    }
    processed_path = Path(args.processed_output)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.write_text(json.dumps(processed_payload, indent=2) + "\n", encoding="utf-8")

    examples_payload = {
        "source_rollouts_file": str(rollouts_path),
        "tau": float(args.tau),
        "tau_pos": float(tau_pos),
        "tau_neg": float(tau_neg),
        "positive_only": bool(args.positive_only),
        "n_examples": len(training_examples),
        "examples": training_examples,
    }
    examples_path = Path(args.training_examples_output)
    examples_path.parent.mkdir(parents=True, exist_ok=True)
    examples_path.write_text(json.dumps(examples_payload, indent=2) + "\n", encoding="utf-8")

    stats_payload = {
        "source_rollouts_file": str(rollouts_path),
        "target_ids_file": str(target_path),
        "processed_output_file": str(processed_path),
        "training_examples_file": str(examples_path),
        "n_total_problems": len(by_problem),
        "n_target_problems": len(target_ids),
        "n_rollouts_target": len(processed_rollouts),
        "label_counts": dict(label_counts),
        "tau_pos": float(tau_pos),
        "tau_neg": float(tau_neg),
        "positive_only": bool(args.positive_only),
        "pos_decision_tokens_total": int(pos_decision_tokens_total),
        "neg_decision_tokens_total": int(neg_decision_tokens_total),
        "neg_to_pos_decision_ratio": (
            float(neg_decision_tokens_total / pos_decision_tokens_total)
            if pos_decision_tokens_total > 0
            else None
        ),
        "decision_positions_per_rollout_mean": float(statistics.mean(decision_counts)) if decision_counts else 0.0,
        "decision_positions_per_rollout_median": float(statistics.median(decision_counts)) if decision_counts else 0.0,
        "decision_fraction_mean": float(statistics.mean(decision_frac)) if decision_frac else 0.0,
        "decision_entropy_mean": float(statistics.mean(decision_entropies)) if decision_entropies else 0.0,
        "advantage_mean": float(statistics.mean(advantages)) if advantages else 0.0,
        "advantage_std": float(statistics.pstdev(advantages)) if len(advantages) > 1 else 0.0,
        "bucket_counts_all": dict(bucket_counts),
        "bucket_counts_target": dict(target_bucket_counts),
        "code_decision_scope": str(args.code_decision_scope),
    }
    stats_path = Path(args.stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats_payload, indent=2) + "\n", encoding="utf-8")

    # Save selected source records for in-distribution evaluation. Reconstruct
    # them directly from the rollout payload so non-math custom datasets keep
    # their original prompt messages and reward metadata.
    selected_records = []
    for pid in target_ids:
        rows = by_problem.get(pid, [])
        if not rows:
            continue
        src = rows[0]
        rec = {
            "problem_id": str(pid),
            "problem_text": str(src.get("problem_text", src.get("prompt_text", ""))),
            "ground_truth": src.get("ground_truth"),
            "category": str(src.get("category", "unknown")),
        }
        for key in ("task_id", "entry_point", "prompt_messages", "reward_model", "extra_info"):
            if key in src and src.get(key) is not None:
                rec[key] = src.get(key)
        selected_records.append(rec)
    selected_records_payload = {
        "dataset": meta.get("dataset", "unknown"),
        "description": "Selected ReasonMaxxer target source records for in-distribution evaluation.",
        "problem_ids": target_ids,
        "records": selected_records,
    }
    selected_records_path = Path(args.selected_records_output)
    selected_records_path.parent.mkdir(parents=True, exist_ok=True)
    selected_records_path.write_text(json.dumps(selected_records_payload, indent=2) + "\n", encoding="utf-8")

    if str(args.curriculum_order_output).strip():
        def _clarity(row: dict[str, Any]) -> float:
            adv = abs(float(row.get("advantage", 0.0)))
            ndec = len(row.get("decision_positions", []) or [])
            clen = int(row.get("completion_length", 0))
            return float(adv * (float(ndec) / float(max(1, clen))))

        ordering_rows = []
        for row in processed_rollouts:
            ordering_rows.append(
                {
                    "problem_id": str(row.get("problem_id", "")),
                    "gen_index": int(row.get("gen_index", 0)),
                    "clarity_score": _clarity(row),
                    "advantage_abs": abs(float(row.get("advantage", 0.0))),
                    "decision_tokens": int(len(row.get("decision_positions", []) or [])),
                    "completion_length": int(row.get("completion_length", 0)),
                }
            )
        ordering_rows = sorted(
            ordering_rows,
            key=lambda x: (
                -float(x["clarity_score"]),
                -float(x["advantage_abs"]),
                -int(x["decision_tokens"]),
                str(x["problem_id"]),
                int(x["gen_index"]),
            ),
        )
        curr_payload = {
            "source_processed_output": str(processed_path),
            "curriculum_key": "signal_clarity",
            "tau_pos": float(tau_pos),
            "tau_neg": float(tau_neg),
            "positive_only": bool(args.positive_only),
            "rows": ordering_rows,
        }
        curr_path = Path(args.curriculum_order_output)
        curr_path.parent.mkdir(parents=True, exist_ok=True)
        curr_path.write_text(json.dumps(curr_payload, indent=2) + "\n", encoding="utf-8")
        print(f"[saved] {curr_path}")

    print(f"[saved] {target_path}")
    print(f"[saved] {processed_path}")
    print(f"[saved] {examples_path}")
    print(f"[saved] {stats_path}")
    print(f"[saved] {selected_records_path}")
    print(
        f"[done] target_problems={len(target_ids)} rollouts={len(processed_rollouts)} "
        f"examples={len(training_examples)} mean_decisions={stats_payload['decision_positions_per_rollout_mean']:.2f} "
        f"mean_decision_frac={stats_payload['decision_fraction_mean']:.4f}"
    )


if __name__ == "__main__":
    main()
