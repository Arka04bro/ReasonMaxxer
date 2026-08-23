from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _dtype_from_name(name: str) -> torch.dtype | None:
    n = name.strip().lower()
    if n == "auto":
        return None
    if n in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if n in {"fp16", "float16"}:
        return torch.float16
    if n in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


@dataclass
class ReasonMaxxerItem:
    problem_id: str
    gen_index: int
    input_ids: list[int]
    prompt_length: int
    completion_length: int
    advantage: float
    decision_positions: list[int]
    problem_weight: float = 1.0
    clarity_score: float = 0.0


class ReasonMaxxerRolloutDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        max_seq_len: int,
        truncate_side: str,
        normalize_by_problem: bool = False,
    ) -> None:
        self.items: list[ReasonMaxxerItem] = []
        for row in rows:
            input_ids = row.get("input_ids") or []
            if not isinstance(input_ids, list) or len(input_ids) < 2:
                continue
            ids = [int(x) for x in input_ids]

            prompt_len = int(row.get("prompt_length", 1))
            prompt_len = max(1, min(prompt_len, len(ids)))

            decision_positions = [int(x) for x in row.get("decision_positions", [])]

            if max_seq_len > 0 and len(ids) > max_seq_len:
                if truncate_side == "left":
                    offset = len(ids) - int(max_seq_len)
                    ids = ids[offset:]
                    prompt_len = max(1, prompt_len - offset)
                    decision_positions = [p - offset for p in decision_positions if (p - offset) >= 0]
                else:
                    ids = ids[: int(max_seq_len)]
                    prompt_len = min(prompt_len, len(ids))
                    decision_positions = [p for p in decision_positions if p < (len(ids) - 1)]

            if len(ids) < 2:
                continue

            # Prediction positions are [0, len(ids)-2].
            pred_cap = len(ids) - 1
            decision_positions = [p for p in decision_positions if 0 <= p < pred_cap]

            self.items.append(
                ReasonMaxxerItem(
                    problem_id=str(row.get("problem_id", "")),
                    gen_index=int(row.get("gen_index", 0)),
                    input_ids=ids,
                    prompt_length=prompt_len,
                    completion_length=int(
                        max(1, min(int(row.get("completion_length", max(1, len(ids) - prompt_len))), len(ids) - prompt_len))
                    ),
                    advantage=float(row.get("advantage", 0.0)),
                    decision_positions=decision_positions,
                )
            )

        if normalize_by_problem and self.items:
            counts: dict[str, int] = {}
            for item in self.items:
                counts[item.problem_id] = counts.get(item.problem_id, 0) + 1
            mean_count = float(len(self.items)) / float(max(1, len(counts)))
            for item in self.items:
                c = counts.get(item.problem_id, 1)
                item.problem_weight = float(mean_count / float(max(1, c)))

        for item in self.items:
            # Curriculum key used by --curriculum_ordering=signal_clarity.
            item.clarity_score = float(
                abs(float(item.advantage))
                * (float(len(item.decision_positions)) / float(max(1, int(item.completion_length))))
            )

    def sort_items(self, key: str) -> None:
        if key == "signal_clarity":
            self.items.sort(
                key=lambda x: (
                    -float(x.clarity_score),
                    -abs(float(x.advantage)),
                    -int(len(x.decision_positions)),
                    str(x.problem_id),
                    int(x.gen_index),
                )
            )
            return
        if key == "deterministic_key":
            self.items.sort(key=lambda x: (str(x.problem_id), int(x.gen_index)))
            return
        raise ValueError(f"Unsupported sort key: {key}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> ReasonMaxxerItem:
        return self.items[idx]


class BalancedAccumWindowSampler(Sampler[int]):
    def __init__(
        self,
        dataset: ReasonMaxxerRolloutDataset,
        *,
        pos_per_window: int,
        neg_per_window: int,
        pattern: str,
    ) -> None:
        self.dataset = dataset
        self.pos_per_window = int(pos_per_window)
        self.neg_per_window = int(neg_per_window)
        self.pattern = str(pattern)
        self.window_size = self.pos_per_window + self.neg_per_window

        if self.pos_per_window <= 0 or self.neg_per_window <= 0:
            raise ValueError("BalancedAccumWindowSampler requires positive pos/neg counts.")
        if self.window_size <= 0:
            raise ValueError("BalancedAccumWindowSampler requires a nonzero window size.")

        self.pos_indices = [i for i, item in enumerate(self.dataset.items) if float(item.advantage) >= 0.0]
        self.neg_indices = [i for i, item in enumerate(self.dataset.items) if float(item.advantage) < 0.0]
        if not self.pos_indices or not self.neg_indices:
            raise ValueError(
                "BalancedAccumWindowSampler requires both positive and negative rollout items. "
                f"Found pos={len(self.pos_indices)} neg={len(self.neg_indices)}."
            )

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _take_cycled(pool: list[int], start: int, n: int) -> tuple[list[int], int]:
        out: list[int] = []
        idx = int(start)
        plen = len(pool)
        for _ in range(int(n)):
            out.append(pool[idx])
            idx += 1
            if idx >= plen:
                idx = 0
        return out, idx

    def _emit_window(self, pos_chunk: list[int], neg_chunk: list[int]) -> list[int]:
        if self.pattern == "grouped_pos_first":
            return list(pos_chunk) + list(neg_chunk)
        if self.pattern == "grouped_neg_first":
            return list(neg_chunk) + list(pos_chunk)
        if self.pattern == "alternating_neg_first":
            out: list[int] = []
            for p, n in zip(pos_chunk, neg_chunk):
                out.extend([n, p])
            return out
        if self.pattern == "alternating_pos_first":
            out = []
            for p, n in zip(pos_chunk, neg_chunk):
                out.extend([p, n])
            return out
        raise ValueError(f"Unsupported accum window pattern: {self.pattern}")

    def __iter__(self):
        total = len(self.dataset)
        num_windows = math.ceil(float(total) / float(self.window_size))
        pos_cursor = 0
        neg_cursor = 0
        order: list[int] = []

        for _ in range(int(num_windows)):
            pos_chunk, pos_cursor = self._take_cycled(self.pos_indices, pos_cursor, self.pos_per_window)
            neg_chunk, neg_cursor = self._take_cycled(self.neg_indices, neg_cursor, self.neg_per_window)
            order.extend(self._emit_window(pos_chunk, neg_chunk))

        return iter(order[:total])


@dataclass
class ReasonMaxxerBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    valid_pred_mask: torch.Tensor
    generated_pred_mask: torch.Tensor
    decision_pred_mask: torch.Tensor
    advantages: torch.Tensor
    problem_weights: torch.Tensor
    problem_ids: list[str]
    gen_indices: torch.Tensor


def collate_fn(batch: list[ReasonMaxxerItem], pad_id: int) -> ReasonMaxxerBatch:
    bsz = len(batch)
    max_len = max(len(x.input_ids) for x in batch)
    max_pred = max_len - 1

    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    valid_pred_mask = torch.zeros((bsz, max_pred), dtype=torch.bool)
    generated_pred_mask = torch.zeros((bsz, max_pred), dtype=torch.bool)
    decision_pred_mask = torch.zeros((bsz, max_pred), dtype=torch.bool)
    advantages = torch.zeros((bsz,), dtype=torch.float32)
    problem_weights = torch.ones((bsz,), dtype=torch.float32)
    gen_indices = torch.zeros((bsz,), dtype=torch.long)
    problem_ids: list[str] = []

    for i, item in enumerate(batch):
        seq_len = len(item.input_ids)
        input_ids[i, :seq_len] = torch.tensor(item.input_ids, dtype=torch.long)
        attention_mask[i, :seq_len] = 1

        pred_len = seq_len - 1
        valid_pred_mask[i, :pred_len] = True

        gen_start = max(item.prompt_length - 1, 0)
        if gen_start < pred_len:
            generated_pred_mask[i, gen_start:pred_len] = True

        for pos in item.decision_positions:
            if 0 <= pos < pred_len:
                decision_pred_mask[i, pos] = True

        advantages[i] = float(item.advantage)
        problem_weights[i] = float(item.problem_weight)
        gen_indices[i] = int(item.gen_index)
        problem_ids.append(item.problem_id)

    return ReasonMaxxerBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        valid_pred_mask=valid_pred_mask,
        generated_pred_mask=generated_pred_mask,
        decision_pred_mask=decision_pred_mask,
        advantages=advantages,
        problem_weights=problem_weights,
        problem_ids=problem_ids,
        gen_indices=gen_indices,
    )


def _to_device(batch: ReasonMaxxerBatch, device: torch.device) -> ReasonMaxxerBatch:
    return ReasonMaxxerBatch(
        input_ids=batch.input_ids.to(device, non_blocking=True),
        attention_mask=batch.attention_mask.to(device, non_blocking=True),
        valid_pred_mask=batch.valid_pred_mask.to(device, non_blocking=True),
        generated_pred_mask=batch.generated_pred_mask.to(device, non_blocking=True),
        decision_pred_mask=batch.decision_pred_mask.to(device, non_blocking=True),
        advantages=batch.advantages.to(device, non_blocking=True),
        problem_weights=batch.problem_weights.to(device, non_blocking=True),
        problem_ids=batch.problem_ids,
        gen_indices=batch.gen_indices.to(device, non_blocking=True),
    )


def _epoch_dir(output_dir: Path, epoch: int) -> Path:
    return output_dir / f"epoch_{epoch}"


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    if den == 0:
        return default
    return float(num / den)


def _epochf_dir(output_dir: Path, epoch_float: float) -> Path:
    label = f"{epoch_float:.2f}".rstrip("0").rstrip(".").replace(".", "p")
    return output_dir / f"epochf_{label}"


def _safe_mean(vals: list[float], default: float = float("nan")) -> float:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    if not xs:
        return float(default)
    return float(sum(xs) / len(xs))


def _window_mean(steps: list[dict[str, Any]], lo_opt_step: int, hi_opt_step: int, key: str) -> float:
    win = [
        float(s.get(key, float("nan")))
        for s in steps
        if lo_opt_step < int(s.get("optimizer_step", 0)) <= hi_opt_step
    ]
    return _safe_mean(win)


def _load_heldout_rollouts(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        rows = payload.get("rollouts")
        if isinstance(rows, list):
            return rows
    raise ValueError(f"Unsupported heldout rollout format: {path}")


def _normalize_cgap_item(row: dict[str, Any], tau: float) -> tuple[list[int], list[int], bool]:
    ids = row.get("input_ids") or row.get("tokens") or []
    if not isinstance(ids, list) or len(ids) < 2:
        return [], [], bool(row.get("correct", False))
    input_ids = [int(x) for x in ids]
    pred_cap = len(input_ids) - 1

    decision_positions = row.get("decision_positions")
    if isinstance(decision_positions, list):
        dpos = [int(p) for p in decision_positions if 0 <= int(p) < pred_cap]
        return input_ids, dpos, bool(row.get("correct", False))

    entropies = row.get("entropies") or []
    entropies = [float(x) for x in entropies]
    prompt_len = int(row.get("prompt_length", 1))
    prompt_len = max(1, min(prompt_len, len(input_ids)))
    gen_pred_start = int(row.get("generation_pred_start", max(0, prompt_len - 1)))

    dpos: list[int] = []
    for i, ent in enumerate(entropies):
        pred_pos = gen_pred_start + i
        if pred_pos >= pred_cap:
            break
        if float(ent) > float(tau):
            dpos.append(int(pred_pos))
    return input_ids, dpos, bool(row.get("correct", False))


def _compute_contrastive_gap(
    *,
    policy,
    base_model,
    heldout_rows: list[dict[str, Any]],
    tau: float,
    policy_device: torch.device,
    base_device: torch.device,
    amp_dtype: torch.dtype | None,
) -> tuple[float, float, float, int, int]:
    pos_sum = 0.0
    neg_sum = 0.0
    pos_n = 0
    neg_n = 0
    use_amp = amp_dtype is not None and policy_device.type == "cuda"
    for row in heldout_rows:
        ids, dpos, is_correct = _normalize_cgap_item(row, tau=float(tau))
        if len(ids) < 2 or not dpos:
            continue
        input_ids = torch.tensor(ids, dtype=torch.long, device=policy_device).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=policy_device)
        target = input_ids[:, 1:]
        base_input_ids = input_ids if base_device == policy_device else input_ids.to(base_device, non_blocking=True)
        base_attention_mask = (
            attention_mask if base_device == policy_device else attention_mask.to(base_device, non_blocking=True)
        )

        autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
        with torch.no_grad():
            with autocast_ctx:
                s_logits = policy(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
                b_logits = base_model(input_ids=base_input_ids, attention_mask=base_attention_mask).logits[:, :-1, :]
                if base_device != policy_device:
                    b_logits = b_logits.to(policy_device, non_blocking=True)

            s_logp = F.log_softmax(s_logits, dim=-1)
            b_logp = F.log_softmax(b_logits, dim=-1)
            s_t = s_logp.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)[0]
            b_t = b_logp.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)[0]
            delta = s_t - b_t

        for p in dpos:
            if 0 <= int(p) < delta.shape[0]:
                v = float(delta[int(p)].item())
                if is_correct:
                    pos_sum += v
                    pos_n += 1
                else:
                    neg_sum += v
                    neg_n += 1

    dpos_mean = _safe_div(pos_sum, float(pos_n), default=float("nan"))
    dneg_mean = _safe_div(neg_sum, float(neg_n), default=float("nan"))
    gap = (dpos_mean - dneg_mean) if (math.isfinite(dpos_mean) and math.isfinite(dneg_mean)) else float("nan")
    return float(dpos_mean), float(dneg_mean), float(gap), int(pos_n), int(neg_n)


def _build_lora(
    model,
    *,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: list[str],
):
    cfg = LoraConfig(
        r=int(rank),
        lora_alpha=int(alpha),
        target_modules=target_modules,
        lora_dropout=float(dropout),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, cfg)


def main() -> None:
    p = argparse.ArgumentParser(description="Train a ReasonMaxxer LoRA adapter from offline entropy-tagged rollouts.")
    p.add_argument("--training_data", default="data/contrastive_experiments/training_data_processed.json")
    p.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument(
        "--init_adapter",
        default="",
        help="Optional LoRA adapter directory to warm-start from.",
    )

    p.add_argument("--variant", choices=["reasonmaxxer", "fullseq"], default="reasonmaxxer")
    p.add_argument("--target_modules", default="o_proj")
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum_steps", type=int, default=4)
    p.add_argument("--learning_rate", "--lr", type=float, default=2e-4, dest="lr")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--kl_weight", type=float, default=0.1)
    p.add_argument(
        "--kl_on_generated_only",
        action="store_true",
        help="If set, apply KL only over generated-token prediction positions.",
    )
    p.add_argument("--adv_clip", type=float, default=2.5)
    p.add_argument(
        "--clip_eps_low",
        type=float,
        default=0.2,
        help="Lower PPO clip bound for decision_objective=clipped_ratio.",
    )
    p.add_argument(
        "--clip_eps_high",
        type=float,
        default=0.2,
        help="Upper PPO clip bound (set larger than --clip_eps_low for clip-higher).",
    )
    p.add_argument(
        "--dual_clip_c",
        type=float,
        default=3.0,
        help=(
            "Dual-clip constant for negative advantages (Ye et al., 2020). Plain PPO clipping is "
            "bounded below but grows without limit when a negative-advantage ratio runs large, so "
            "the surrogate is additionally floored at c*A. Set 0 to disable."
        ),
    )
    p.add_argument(
        "--behavior_temperature",
        type=float,
        default=1.0,
        help=(
            "Sampling temperature used to generate the rollouts. The importance ratio is taken "
            "against the behaviour policy pi_base^(1/T), not against the T=1 base distribution. "
            "The KL anchor stays at T=1 because it anchors the model, not the sampler."
        ),
    )
    p.add_argument(
        "--kl_decision_weight",
        type=float,
        default=0.0,
        help=(
            "KL weight applied at decision tokens as well. v1 left decision tokens with no trust "
            "region at all, which is where the distribution can collapse."
        ),
    )
    p.add_argument(
        "--decision_objective",
        choices=["adv_ce", "delta_hinge", "correct_sft", "clipped_ratio"],
        default="adv_ce",
        help=(
            "Decision-token objective: "
            "legacy signed-adv CE, anchored delta hinge, "
            "correct-rollout-only CE (entropy-gated SFT), "
            "or the bounded clipped off-policy surrogate."
        ),
    )
    p.add_argument(
        "--beta_neg",
        type=float,
        default=1.0,
        help="Scale for negative-rollout pressure at decision tokens.",
    )
    p.add_argument(
        "--neg_prob_floor",
        type=float,
        default=0.0,
        help=(
            "Optional floor for negative-token suppression. "
            "Negative decision tokens with student prob <= floor are ignored."
        ),
    )
    p.add_argument(
        "--pos_only_decisions",
        action="store_true",
        help="If set, only positive (adv>0) decision tokens are optimized.",
    )
    p.add_argument(
        "--normalize_by_problem",
        action="store_true",
        help="If set, reweight rollouts by inverse problem frequency to equalize per-problem contribution.",
    )
    p.add_argument(
        "--pos_margin",
        type=float,
        default=0.0,
        help="For delta_hinge: target lower bound on (logp_student-logp_base) for positive decisions.",
    )
    p.add_argument(
        "--neg_margin",
        type=float,
        default=0.0,
        help="For delta_hinge: target lower bound on -(logp_student-logp_base) for negative decisions.",
    )
    p.add_argument(
        "--balance_pos_neg",
        action="store_true",
        help="If set, average positive and negative decision losses equally when both exist.",
    )
    p.add_argument(
        "--accum_window_pos",
        type=int,
        default=0,
        help=(
            "If >0 together with --accum_window_neg, build accumulation windows with a fixed "
            "number of positive microbatches per optimizer step."
        ),
    )
    p.add_argument(
        "--accum_window_neg",
        type=int,
        default=0,
        help=(
            "If >0 together with --accum_window_pos, build accumulation windows with a fixed "
            "number of negative microbatches per optimizer step."
        ),
    )
    p.add_argument(
        "--accum_window_pattern",
        choices=[
            "alternating_pos_first",
            "alternating_neg_first",
            "grouped_pos_first",
            "grouped_neg_first",
        ],
        default="alternating_pos_first",
        help="Order of positive and negative examples inside a balanced accumulation window.",
    )
    p.add_argument(
        "--separate_pos_neg_norm",
        action="store_true",
        help=(
            "For adv_ce: normalize positive and negative decision losses separately before combining. "
            "This removes batch-composition dependence in pos/neg weighting."
        ),
    )
    p.add_argument(
        "--adaptive_betaneg",
        action="store_true",
        help="Enable adaptive beta_neg updates from dlogp+ trend.",
    )
    p.add_argument("--betaneg_init", type=float, default=1.0)
    p.add_argument("--betaneg_decay", type=float, default=0.7)
    p.add_argument("--betaneg_restore", type=float, default=1.05)
    p.add_argument("--betaneg_min", type=float, default=0.05)
    p.add_argument("--betaneg_ema_alpha", type=float, default=0.3)

    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--truncate_side", choices=["right", "left"], default="right")
    p.add_argument(
        "--curriculum_ordering",
        action="store_true",
        help="Use fixed curriculum ordering instead of shuffled sampling.",
    )
    p.add_argument(
        "--curriculum_key",
        choices=["signal_clarity", "deterministic_key"],
        default="signal_clarity",
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--policy_device",
        default="",
        help="Optional device for the trainable policy model. Defaults to --device.",
    )
    p.add_argument(
        "--base_device",
        default="",
        help="Optional device for the frozen base model. Defaults to the policy device.",
    )
    p.add_argument("--dtype", default="bfloat16", choices=["auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"])
    p.add_argument("--hf_attn_implementation", default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Diagnostic deterministic mode (fixed ordering, float32, no dropout, no AMP, no gradient checkpointing).",
    )
    p.add_argument(
        "--checkpoint_metrics_csv",
        default="checkpoint_metrics.csv",
        help="Per-checkpoint metrics CSV path (relative to output_dir if not absolute).",
    )
    p.add_argument(
        "--heldout_cgap_rollouts",
        default="",
        help="Optional heldout rollout file for contrastive-gap computation at each checkpoint.",
    )
    p.add_argument(
        "--cgap_tau",
        type=float,
        default=2.5,
        help="Entropy threshold for contrastive-gap decision positions when not precomputed.",
    )
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--save_every_epoch", action="store_true", default=True)
    p.add_argument("--no_save_every_epoch", dest="save_every_epoch", action="store_false")
    p.add_argument(
        "--save_every_fractional_epoch",
        type=float,
        default=0.0,
        help="If >0, save additional checkpoints every this many epochs (e.g., 0.5).",
    )
    p.add_argument(
        "--save_every_optimizer_steps",
        type=int,
        default=0,
        help="If >0, save additional checkpoints every N optimizer steps (overrides fractional setting).",
    )
    p.add_argument(
        "--max_optimizer_steps",
        type=int,
        default=0,
        help="If >0, stop training after this many optimizer steps.",
    )
    p.add_argument(
        "--early_stop_delta_logp_pos_lt",
        type=float,
        default=None,
        help="If set, stop training once epoch delta_logp_pos_mean falls below this threshold.",
    )
    args = p.parse_args()

    if bool(args.deterministic):
        # Deterministic debug mode: prioritize reproducibility over speed.
        args.lora_dropout = 0.0
        args.dtype = "float32"
        args.gradient_checkpointing = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    _set_seed(int(args.seed))
    if bool(args.deterministic):
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    dtype = _dtype_from_name(args.dtype)
    policy_device = torch.device(str(args.policy_device).strip() or args.device)
    base_device = torch.device(str(args.base_device).strip() or str(policy_device))
    device = policy_device

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(Path(args.training_data).read_text(encoding="utf-8"))
    rows = list(payload.get("rollouts", []))
    if not rows:
        raise ValueError(f"No rollout rows in {args.training_data}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = int(tokenizer.pad_token_id)

    dataset = ReasonMaxxerRolloutDataset(
        rows,
        max_seq_len=int(args.max_seq_len),
        truncate_side=str(args.truncate_side),
        normalize_by_problem=bool(args.normalize_by_problem),
    )
    if len(dataset) == 0:
        raise ValueError("All rows were filtered out (likely due to max_seq_len/truncation).")

    if bool(args.curriculum_ordering):
        dataset.sort_items(str(args.curriculum_key))
    elif bool(args.deterministic):
        dataset.sort_items("deterministic_key")

    use_balanced_windows = int(args.accum_window_pos) > 0 or int(args.accum_window_neg) > 0
    if use_balanced_windows:
        if int(args.batch_size) != 1:
            raise ValueError("Balanced accumulation windows currently require --batch_size 1.")
        if int(args.accum_window_pos) <= 0 or int(args.accum_window_neg) <= 0:
            raise ValueError("Balanced accumulation windows require both --accum_window_pos and --accum_window_neg > 0.")
        if int(args.accum_window_pos) + int(args.accum_window_neg) != int(args.grad_accum_steps):
            raise ValueError(
                "Balanced accumulation windows require accum_window_pos + accum_window_neg == grad_accum_steps. "
                f"Got {args.accum_window_pos} + {args.accum_window_neg} vs {args.grad_accum_steps}."
            )
        sampler = BalancedAccumWindowSampler(
            dataset,
            pos_per_window=int(args.accum_window_pos),
            neg_per_window=int(args.accum_window_neg),
            pattern=str(args.accum_window_pattern),
        )
        print(
            "[info] balanced accumulation windows enabled: "
            f"{args.accum_window_pos} pos + {args.accum_window_neg} neg, "
            f"pattern={args.accum_window_pattern}, "
            f"pos_pool={len(sampler.pos_indices)}, neg_pool={len(sampler.neg_indices)}"
        )
        loader = DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            sampler=sampler,
            collate_fn=lambda b: collate_fn(b, pad_id=pad_id),
        )
    else:
        shuffle_loader = not (bool(args.curriculum_ordering) or bool(args.deterministic))
        loader = DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=shuffle_loader,
            collate_fn=lambda b: collate_fn(b, pad_id=pad_id),
        )

    print(
        f"[info] device placement: policy={policy_device} "
        f"base={base_device}"
    )
    print(f"[info] loading frozen base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        torch_dtype=dtype if dtype is not None else "auto",
        attn_implementation=args.hf_attn_implementation,
    ).to(base_device)
    base_model.eval()
    for p_ in base_model.parameters():
        p_.requires_grad_(False)
    base_model.config.use_cache = False

    print(f"[info] loading trainable policy model: {args.base_model}")
    policy = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        torch_dtype=dtype if dtype is not None else "auto",
        attn_implementation=args.hf_attn_implementation,
    ).to(policy_device)
    policy.config.use_cache = False

    target_modules = [x.strip() for x in str(args.target_modules).split(",") if x.strip()]
    if not target_modules:
        raise ValueError("No LoRA target modules resolved from --target_modules")

    if str(args.init_adapter).strip():
        init_adapter = Path(str(args.init_adapter))
        if not init_adapter.exists():
            raise FileNotFoundError(f"init_adapter path not found: {init_adapter}")
        print(f"[info] warm-starting policy adapter from: {init_adapter}")
        policy = PeftModel.from_pretrained(policy, str(init_adapter), is_trainable=True)
    else:
        policy = _build_lora(
            policy,
            rank=int(args.lora_rank),
            alpha=int(args.lora_alpha),
            dropout=float(args.lora_dropout),
            target_modules=target_modules,
        )
    if args.gradient_checkpointing:
        try:
            policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            policy.gradient_checkpointing_enable()
    policy.train()
    try:
        policy.print_trainable_parameters()
    except Exception:
        pass

    trainable = [p_ for p_ in policy.parameters() if p_.requires_grad]
    optimizer = AdamW(trainable, lr=float(args.lr), weight_decay=float(args.weight_decay))

    opt_steps_per_epoch = math.ceil(len(loader) / max(1, int(args.grad_accum_steps)))
    total_opt_steps = opt_steps_per_epoch * int(args.epochs)
    total_opt_steps = max(1, total_opt_steps)
    warmup_steps = max(0, min(int(args.warmup_steps), total_opt_steps - 1))
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_opt_steps,
    )

    save_every_opt_steps = int(args.save_every_optimizer_steps)
    if save_every_opt_steps <= 0 and float(args.save_every_fractional_epoch) > 0:
        save_every_opt_steps = max(1, int(round(float(args.save_every_fractional_epoch) * float(opt_steps_per_epoch))))

    amp_dtype = None
    if device.type == "cuda":
        if dtype in {torch.bfloat16, torch.float16}:
            amp_dtype = dtype
        elif dtype is None:
            amp_dtype = torch.bfloat16

    heldout_rows: list[dict[str, Any]] = []
    if str(args.heldout_cgap_rollouts).strip():
        heldout_path = Path(str(args.heldout_cgap_rollouts))
        if not heldout_path.exists():
            raise FileNotFoundError(f"Heldout contrastive-gap file not found: {heldout_path}")
        heldout_rows = _load_heldout_rollouts(heldout_path)
        print(f"[info] loaded heldout contrastive-gap rows: {len(heldout_rows)} from {heldout_path}")

    metrics_csv_path = Path(args.checkpoint_metrics_csv)
    if not metrics_csv_path.is_absolute():
        metrics_csv_path = out_dir / metrics_csv_path
    metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_fields = [
        "step",
        "epoch",
        "epoch_float",
        "checkpoint_tag",
        "checkpoint_path",
        "optimizer_step",
        "global_step",
        "avg_loss",
        "avg_decision_loss",
        "avg_kl_loss",
        "dlogp_plus",
        "dlogp_minus",
        "decision_pressure",
        "decision_fraction",
        "dlogp_plus_val",
        "dlogp_minus_val",
        "contrastive_gap",
        "holdout60_pass1",
        "beta_neg_current",
        "ema_slope",
        "cgap_pos_tokens",
        "cgap_neg_tokens",
    ]
    if not metrics_csv_path.exists():
        with metrics_csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=metrics_fields)
            w.writeheader()

    train_log: dict[str, Any] = {
        "config": vars(args),
        "training_data": args.training_data,
        "source_config": payload.get("config", {}),
        "steps": [],
        "epochs": [],
        "checkpoints": [],
    }

    global_step = 0
    optimizer_step = 0
    t0 = time.time()
    last_epoch = 0
    prev_ckpt_opt_step = 0
    prev_dlogp_plus: float | None = None
    ema_slope = 0.0
    current_beta_neg = float(args.beta_neg)
    if bool(args.adaptive_betaneg):
        current_beta_neg = float(args.betaneg_init)

    def _append_checkpoint_metrics_row(*, ckpt_tag: str, ckpt_path: str, epoch_i: int, epoch_float: float) -> None:
        nonlocal prev_ckpt_opt_step
        lo = int(prev_ckpt_opt_step)
        hi = int(optimizer_step)
        steps = list(train_log.get("steps", []))

        avg_loss = _window_mean(steps, lo, hi, "loss")
        avg_reasonmaxxer = _window_mean(steps, lo, hi, "decision_loss")
        avg_kl = _window_mean(steps, lo, hi, "kl_loss")
        dlogp_plus = _window_mean(steps, lo, hi, "delta_logp_pos_mean")
        dlogp_minus = _window_mean(steps, lo, hi, "delta_logp_neg_mean")
        decision_fraction = _window_mean(steps, lo, hi, "decision_frac_generated")

        pos_press = [
            float(s.get("pos_pressure_sum", 0.0))
            for s in steps
            if lo < int(s.get("optimizer_step", 0)) <= hi
        ]
        neg_press = [
            float(s.get("neg_pressure_sum", 0.0))
            for s in steps
            if lo < int(s.get("optimizer_step", 0)) <= hi
        ]
        pos_sum = float(sum(pos_press))
        neg_sum = float(sum(neg_press))
        decision_pressure = _safe_div(neg_sum, pos_sum, default=float("nan")) if pos_sum > 0 else float("nan")

        dpos_val = float("nan")
        dneg_val = float("nan")
        cgap = float("nan")
        cgap_pos_n = 0
        cgap_neg_n = 0
        if heldout_rows:
            was_training = bool(policy.training)
            policy.eval()
            try:
                    dpos_val, dneg_val, cgap, cgap_pos_n, cgap_neg_n = _compute_contrastive_gap(
                        policy=policy,
                        base_model=base_model,
                        heldout_rows=heldout_rows,
                        tau=float(args.cgap_tau),
                        policy_device=policy_device,
                        base_device=base_device,
                        amp_dtype=amp_dtype,
                    )
            finally:
                if was_training:
                    policy.train()

        row = {
            "step": int(global_step),
            "epoch": int(epoch_i),
            "epoch_float": float(epoch_float),
            "checkpoint_tag": str(ckpt_tag),
            "checkpoint_path": str(ckpt_path),
            "optimizer_step": int(optimizer_step),
            "global_step": int(global_step),
            "avg_loss": avg_loss,
            "avg_decision_loss": avg_reasonmaxxer,
            "avg_kl_loss": avg_kl,
            "dlogp_plus": dlogp_plus,
            "dlogp_minus": dlogp_minus,
            "decision_pressure": decision_pressure,
            "decision_fraction": decision_fraction,
            "dlogp_plus_val": dpos_val,
            "dlogp_minus_val": dneg_val,
            "contrastive_gap": cgap,
            "holdout60_pass1": float("nan"),
            "beta_neg_current": float(current_beta_neg),
            "ema_slope": float(ema_slope),
            "cgap_pos_tokens": int(cgap_pos_n),
            "cgap_neg_tokens": int(cgap_neg_n),
        }
        with metrics_csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=metrics_fields)
            w.writerow(row)

        prev_ckpt_opt_step = int(optimizer_step)

    for epoch in range(1, int(args.epochs) + 1):
        optimizer.zero_grad(set_to_none=True)
        epoch_loss_sum = 0.0
        epoch_reasonmaxxer_sum = 0.0
        epoch_kl_sum = 0.0
        epoch_decision_tokens = 0
        epoch_nondec_tokens = 0
        epoch_valid_tokens = 0
        epoch_generated_tokens = 0
        epoch_pos_decision_tokens = 0
        epoch_neg_decision_tokens = 0
        epoch_pos_ce_sum = 0.0
        epoch_neg_ce_sum = 0.0
        epoch_delta_logp_pos_sum = 0.0
        epoch_delta_logp_neg_sum = 0.0
        epoch_pos_pressure_sum = 0.0
        epoch_neg_pressure_sum = 0.0
        epoch_batches = 0

        pbar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")
        stop_training = False
        for batch_idx, batch in enumerate(pbar, start=1):
            b = _to_device(batch, policy_device)
            if base_device == policy_device:
                base_input_ids = b.input_ids
                base_attention_mask = b.attention_mask
            else:
                # Keep the frozen base on a separate GPU for large-model runs.
                base_input_ids = batch.input_ids.to(base_device, non_blocking=True)
                base_attention_mask = batch.attention_mask.to(base_device, non_blocking=True)
            use_amp = amp_dtype is not None
            autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()

            with autocast_ctx:
                student_out = policy(
                    input_ids=b.input_ids,
                    attention_mask=b.attention_mask,
                )
                student_logits = student_out.logits[:, :-1, :]
                target = b.input_ids[:, 1:]

                with torch.no_grad():
                    base_out = base_model(
                        input_ids=base_input_ids,
                        attention_mask=base_attention_mask,
                    )
                    base_logits = base_out.logits[:, :-1, :]
                    if base_device != policy_device:
                        base_logits = base_logits.to(policy_device, non_blocking=True)

                student_logp = F.log_softmax(student_logits, dim=-1)
                token_ce = -student_logp.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
                base_logp = F.log_softmax(base_logits, dim=-1)
                student_target_logp = student_logp.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
                base_target_logp = base_logp.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
                delta_logp = student_target_logp - base_target_logp

                # The rollouts were sampled at temperature T, so the behaviour policy is
                # pi_base^(1/T) renormalised, not pi_base itself. Only the importance ratio
                # uses it; base_logp above stays at T=1 for the KL anchor.
                if abs(float(args.behavior_temperature) - 1.0) > 1e-6:
                    behavior_logp = F.log_softmax(base_logits / float(args.behavior_temperature), dim=-1)
                    behavior_target_logp = behavior_logp.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
                else:
                    behavior_target_logp = base_target_logp
                log_ratio = student_target_logp - behavior_target_logp

                if args.variant == "fullseq":
                    contrastive_mask = b.generated_pred_mask & b.valid_pred_mask
                else:
                    contrastive_mask = b.decision_pred_mask & b.valid_pred_mask

                adv = b.advantages
                if float(args.adv_clip) > 0:
                    adv = adv.clamp(min=-float(args.adv_clip), max=float(args.adv_clip))
                adv = adv.unsqueeze(1).expand_as(token_ce)
                row_w = b.problem_weights.unsqueeze(1).expand_as(token_ce)
                if not bool(args.normalize_by_problem):
                    row_w = torch.ones_like(row_w)

                pos_mask = contrastive_mask & (adv > 0)
                neg_mask = contrastive_mask & (adv < 0)
                if bool(args.pos_only_decisions):
                    neg_mask = torch.zeros_like(neg_mask)
                if args.decision_objective == "correct_sft":
                    # Correct-rollout-only mode: no explicit negative pressure.
                    neg_mask = torch.zeros_like(neg_mask)
                if float(args.neg_prob_floor) > 0:
                    student_target_prob = student_target_logp.exp()
                    neg_mask = neg_mask & (student_target_prob > float(args.neg_prob_floor))
                contrastive_effective_mask = pos_mask | neg_mask

                decision_count = int(contrastive_effective_mask.sum().item())
                pos_decision_count = int(pos_mask.sum().item())
                neg_decision_count = int(neg_mask.sum().item())
                if args.decision_objective == "delta_hinge":
                    adv_abs = adv.abs()
                    pos_obj_tok = F.relu(float(args.pos_margin) - delta_logp) * adv_abs
                    neg_obj_tok = F.relu(delta_logp + float(args.neg_margin)) * adv_abs * float(current_beta_neg)

                    if pos_decision_count > 0:
                        pos_num = (pos_obj_tok * pos_mask.float() * row_w).sum()
                        pos_den = (pos_mask.float() * row_w).sum().clamp_min(1e-9)
                        pos_loss = pos_num / pos_den
                    else:
                        pos_loss = token_ce.new_zeros(())

                    if neg_decision_count > 0:
                        neg_num = (neg_obj_tok * neg_mask.float() * row_w).sum()
                        neg_den = (neg_mask.float() * row_w).sum().clamp_min(1e-9)
                        neg_loss = neg_num / neg_den
                    else:
                        neg_loss = token_ce.new_zeros(())

                    if decision_count > 0:
                        if bool(args.balance_pos_neg) and pos_decision_count > 0 and neg_decision_count > 0:
                            decision_loss = 0.5 * (pos_loss + neg_loss)
                        else:
                            num = (pos_obj_tok * pos_mask.float() * row_w).sum() + (
                                neg_obj_tok * neg_mask.float() * row_w
                            ).sum()
                            den = (contrastive_effective_mask.float() * row_w).sum().clamp_min(1e-9)
                            decision_loss = num / den
                    else:
                        decision_loss = token_ce.new_zeros(())

                    pos_pressure_sum = (
                        float((pos_obj_tok * pos_mask.float() * row_w).sum().item()) if pos_decision_count > 0 else 0.0
                    )
                    neg_pressure_sum = (
                        float((neg_obj_tok * neg_mask.float() * row_w).sum().item()) if neg_decision_count > 0 else 0.0
                    )
                elif args.decision_objective == "clipped_ratio":
                    # Bounded off-policy surrogate. Unlike A*CE this cannot be driven to
                    # -inf by negative advantages: once the ratio leaves the trust region the
                    # clipped term takes over and its gradient vanishes.
                    ratio = torch.exp(log_ratio.clamp(min=-20.0, max=20.0))
                    lo = 1.0 - float(args.clip_eps_low)
                    hi = 1.0 + float(args.clip_eps_high)
                    adv_eff = adv
                    if float(current_beta_neg) != 1.0:
                        adv_eff = torch.where(adv < 0, adv * float(current_beta_neg), adv)
                    surrogate_tok = torch.min(ratio * adv_eff, ratio.clamp(lo, hi) * adv_eff)
                    if float(args.dual_clip_c) > 1.0:
                        # For A < 0 the unclipped branch is the one that survives min(), so the
                        # surrogate falls linearly in the ratio and the gradient can blow up.
                        # Flooring it at c*A bounds the loss and zeroes the gradient past the floor.
                        surrogate_tok = torch.where(
                            adv_eff < 0,
                            torch.maximum(surrogate_tok, float(args.dual_clip_c) * adv_eff),
                            surrogate_tok,
                        )
                    obj_tok = -surrogate_tok

                    if pos_decision_count > 0:
                        pos_num = (obj_tok * pos_mask.float() * row_w).sum()
                        pos_den = (pos_mask.float() * row_w).sum().clamp_min(1e-9)
                        pos_loss = pos_num / pos_den
                    else:
                        pos_loss = token_ce.new_zeros(())

                    if neg_decision_count > 0:
                        neg_num = (obj_tok * neg_mask.float() * row_w).sum()
                        neg_den = (neg_mask.float() * row_w).sum().clamp_min(1e-9)
                        neg_loss = neg_num / neg_den
                    else:
                        neg_loss = token_ce.new_zeros(())

                    if decision_count > 0:
                        if bool(args.balance_pos_neg) and pos_decision_count > 0 and neg_decision_count > 0:
                            decision_loss = 0.5 * (pos_loss + neg_loss)
                        else:
                            num = (obj_tok * contrastive_effective_mask.float() * row_w).sum()
                            den = (contrastive_effective_mask.float() * row_w).sum().clamp_min(1e-9)
                            decision_loss = num / den
                    else:
                        decision_loss = token_ce.new_zeros(())

                    pos_pressure_sum = (
                        float((obj_tok * pos_mask.float() * row_w).sum().item()) if pos_decision_count > 0 else 0.0
                    )
                    neg_pressure_sum = (
                        float((obj_tok * neg_mask.float() * row_w).sum().item()) if neg_decision_count > 0 else 0.0
                    )
                elif args.decision_objective == "correct_sft":
                    # Entropy-gated SFT: optimize CE only on decision tokens from correct rollouts.
                    if pos_decision_count > 0:
                        pos_num = (token_ce * pos_mask.float() * row_w).sum()
                        pos_den = (pos_mask.float() * row_w).sum().clamp_min(1e-9)
                        decision_loss = pos_num / pos_den
                        pos_pressure_sum = float(pos_num.item())
                    else:
                        decision_loss = token_ce.new_zeros(())
                        pos_pressure_sum = 0.0
                    neg_pressure_sum = 0.0
                else:
                    pos_adv = torch.clamp(adv, min=0.0)
                    neg_adv = torch.clamp(adv, max=0.0)
                    if float(current_beta_neg) != 1.0:
                        neg_adv = neg_adv * float(current_beta_neg)

                    pos_weighted_ce = token_ce * pos_adv
                    neg_weighted_ce = token_ce * neg_adv
                    if bool(args.separate_pos_neg_norm):
                        if pos_decision_count > 0:
                            pos_num = (pos_weighted_ce * pos_mask.float() * row_w).sum()
                            pos_den = (pos_mask.float() * row_w).sum().clamp_min(1e-9)
                            pos_loss = pos_num / pos_den
                        else:
                            pos_loss = token_ce.new_zeros(())

                        if neg_decision_count > 0:
                            neg_num = (neg_weighted_ce * neg_mask.float() * row_w).sum()
                            neg_den = (neg_mask.float() * row_w).sum().clamp_min(1e-9)
                            neg_loss = neg_num / neg_den
                        else:
                            neg_loss = token_ce.new_zeros(())

                        if pos_decision_count > 0 and neg_decision_count > 0 and bool(args.balance_pos_neg):
                            decision_loss = 0.5 * (pos_loss + neg_loss)
                        else:
                            decision_loss = pos_loss + neg_loss
                    else:
                        weighted_ce = pos_weighted_ce + neg_weighted_ce
                        if decision_count > 0:
                            num = (weighted_ce * contrastive_effective_mask.float() * row_w).sum()
                            den = (contrastive_effective_mask.float() * row_w).sum().clamp_min(1e-9)
                            decision_loss = num / den
                        else:
                            decision_loss = token_ce.new_zeros(())

                    pos_pressure_sum = (
                        float((pos_weighted_ce * pos_mask.float() * row_w).sum().item()) if pos_decision_count > 0 else 0.0
                    )
                    neg_pressure_sum = (
                        float(((-neg_weighted_ce) * neg_mask.float() * row_w).sum().item())
                        if neg_decision_count > 0
                        else 0.0
                    )

                pos_ce_sum = float(token_ce[pos_mask].sum().item()) if pos_decision_count > 0 else 0.0
                neg_ce_sum = float(token_ce[neg_mask].sum().item()) if neg_decision_count > 0 else 0.0
                pos_ce_mean = _safe_div(pos_ce_sum, float(pos_decision_count), default=0.0)
                neg_ce_mean = _safe_div(neg_ce_sum, float(neg_decision_count), default=0.0)

                delta_logp_pos_sum = float(delta_logp[pos_mask].sum().item()) if pos_decision_count > 0 else 0.0
                delta_logp_neg_sum = float(delta_logp[neg_mask].sum().item()) if neg_decision_count > 0 else 0.0
                delta_logp_pos_mean = _safe_div(delta_logp_pos_sum, float(pos_decision_count), default=0.0)
                delta_logp_neg_mean = _safe_div(delta_logp_neg_sum, float(neg_decision_count), default=0.0)

                neg_pos_pressure_ratio = _safe_div(neg_pressure_sum, pos_pressure_sum, default=0.0)

                nondecision_mask = b.valid_pred_mask & (~contrastive_effective_mask)
                if args.kl_on_generated_only:
                    nondecision_mask = nondecision_mask & b.generated_pred_mask

                nondec_count = int(nondecision_mask.sum().item())
                valid_count = int(b.valid_pred_mask.sum().item())
                generated_count = int((b.valid_pred_mask & b.generated_pred_mask).sum().item())
                decision_frac_valid = _safe_div(float(decision_count), float(valid_count), default=0.0)
                decision_frac_generated = _safe_div(float(decision_count), float(generated_count), default=0.0)
                kl_decision_weight = float(args.kl_decision_weight)
                need_kl = (nondec_count > 0 and float(args.kl_weight) > 0) or (
                    decision_count > 0 and kl_decision_weight > 0
                )
                kl_decision_loss = token_ce.new_zeros(())
                kl_loss = token_ce.new_zeros(())
                if need_kl:
                    base_prob = F.softmax(base_logits, dim=-1)
                    kl_per_tok = F.kl_div(student_logp, base_prob, reduction="none").sum(dim=-1)
                    if nondec_count > 0 and float(args.kl_weight) > 0:
                        kl_num = (kl_per_tok * nondecision_mask.float() * row_w).sum()
                        kl_den = (nondecision_mask.float() * row_w).sum().clamp_min(1e-9)
                        kl_loss = kl_num / kl_den
                    if decision_count > 0 and kl_decision_weight > 0:
                        dkl_num = (kl_per_tok * contrastive_effective_mask.float() * row_w).sum()
                        dkl_den = (contrastive_effective_mask.float() * row_w).sum().clamp_min(1e-9)
                        kl_decision_loss = dkl_num / dkl_den

                loss = decision_loss + float(args.kl_weight) * kl_loss + kl_decision_weight * kl_decision_loss

            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(
                    f"Encountered non-finite loss at epoch={epoch} batch={batch_idx}: {float(loss.item())}"
                )

            if not loss.requires_grad:
                continue

            (loss / max(1, int(args.grad_accum_steps))).backward()

            epoch_loss_sum += float(loss.item())
            epoch_reasonmaxxer_sum += float(decision_loss.item())
            epoch_kl_sum += float(kl_loss.item())
            epoch_decision_tokens += decision_count
            epoch_nondec_tokens += nondec_count
            epoch_valid_tokens += valid_count
            epoch_generated_tokens += generated_count
            epoch_pos_decision_tokens += pos_decision_count
            epoch_neg_decision_tokens += neg_decision_count
            epoch_pos_ce_sum += pos_ce_sum
            epoch_neg_ce_sum += neg_ce_sum
            epoch_delta_logp_pos_sum += delta_logp_pos_sum
            epoch_delta_logp_neg_sum += delta_logp_neg_sum
            epoch_pos_pressure_sum += pos_pressure_sum
            epoch_neg_pressure_sum += neg_pressure_sum
            epoch_batches += 1
            global_step += 1

            should_step = (batch_idx % max(1, int(args.grad_accum_steps)) == 0) or (batch_idx == len(loader))
            if should_step:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=float(args.max_grad_norm))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                if save_every_opt_steps > 0 and (optimizer_step % save_every_opt_steps == 0):
                    epoch_float = float(optimizer_step) / float(max(1, opt_steps_per_epoch))
                    ckpt_dir = _epochf_dir(out_dir, epoch_float)
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    policy.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    ckpt_rec = {
                        "type": "fractional",
                        "tag": ckpt_dir.name,
                        "path": str(ckpt_dir),
                        "epoch_float": float(epoch_float),
                        "epoch": int(epoch),
                        "global_step": int(global_step),
                        "optimizer_step": int(optimizer_step),
                    }
                    train_log["checkpoints"].append(ckpt_rec)
                    _append_checkpoint_metrics_row(
                        ckpt_tag=str(ckpt_dir.name),
                        ckpt_path=str(ckpt_dir),
                        epoch_i=int(epoch),
                        epoch_float=float(epoch_float),
                    )
                    print(f"[saved] {ckpt_dir}")
                if int(args.max_optimizer_steps) > 0 and int(optimizer_step) >= int(args.max_optimizer_steps):
                    stop_training = True

            if global_step % max(1, int(args.logging_steps)) == 0:
                if bool(args.adaptive_betaneg):
                    dlp = float(delta_logp_pos_mean)
                    if prev_dlogp_plus is not None:
                        slope = float(dlp - prev_dlogp_plus)
                        alpha = float(args.betaneg_ema_alpha)
                        ema_slope = float(alpha * slope + (1.0 - alpha) * ema_slope)
                        if ema_slope < 0.0:
                            current_beta_neg = max(
                                float(args.betaneg_min),
                                float(current_beta_neg * float(args.betaneg_decay)),
                            )
                        else:
                            current_beta_neg = min(
                                float(args.betaneg_init),
                                float(current_beta_neg * float(args.betaneg_restore)),
                            )
                    prev_dlogp_plus = dlp

                rec = {
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                    "optimizer_step": int(optimizer_step),
                    "loss": float(loss.item()),
                    "decision_loss": float(decision_loss.item()),
                    "kl_loss": float(kl_loss.item()),
                    "decision_tokens": int(decision_count),
                    "pos_decision_tokens": int(pos_decision_count),
                    "neg_decision_tokens": int(neg_decision_count),
                    "nondecision_tokens": int(nondec_count),
                    "valid_tokens": int(valid_count),
                    "generated_tokens": int(generated_count),
                    "decision_frac_valid": float(decision_frac_valid),
                    "decision_frac_generated": float(decision_frac_generated),
                    "ce_pos_mean": float(pos_ce_mean),
                    "ce_neg_mean": float(neg_ce_mean),
                    "delta_logp_pos_mean": float(delta_logp_pos_mean),
                    "delta_logp_neg_mean": float(delta_logp_neg_mean),
                    "neg_pos_pressure_ratio": float(neg_pos_pressure_ratio),
                    "pos_pressure_sum": float(pos_pressure_sum),
                    "neg_pressure_sum": float(neg_pressure_sum),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "beta_neg_current": float(current_beta_neg),
                    "ema_slope": float(ema_slope),
                }
                train_log["steps"].append(rec)
                pbar.set_postfix(
                    loss=f"{rec['loss']:.4f}",
                    reasonmaxxer=f"{rec['decision_loss']:.4f}",
                    kl=f"{rec['kl_loss']:.4f}",
                    dec=rec["decision_tokens"],
                    dfrac=f"{rec['decision_frac_generated']:.3f}",
                    npr=f"{rec['neg_pos_pressure_ratio']:.2f}",
                    bneg=f"{rec['beta_neg_current']:.2f}",
                    lr=f"{rec['lr']:.2e}",
                )
            if stop_training:
                break

        epoch_rec = {
            "epoch": int(epoch),
            "avg_loss": float(epoch_loss_sum / max(1, epoch_batches)),
            "avg_decision_loss": float(epoch_reasonmaxxer_sum / max(1, epoch_batches)),
            "avg_kl_loss": float(epoch_kl_sum / max(1, epoch_batches)),
            "total_decision_tokens": int(epoch_decision_tokens),
            "total_pos_decision_tokens": int(epoch_pos_decision_tokens),
            "total_neg_decision_tokens": int(epoch_neg_decision_tokens),
            "total_nondecision_tokens": int(epoch_nondec_tokens),
            "total_valid_tokens": int(epoch_valid_tokens),
            "total_generated_tokens": int(epoch_generated_tokens),
            "decision_frac_valid": _safe_div(float(epoch_decision_tokens), float(epoch_valid_tokens), default=0.0),
            "decision_frac_generated": _safe_div(float(epoch_decision_tokens), float(epoch_generated_tokens), default=0.0),
            "ce_pos_mean": _safe_div(epoch_pos_ce_sum, float(epoch_pos_decision_tokens), default=0.0),
            "ce_neg_mean": _safe_div(epoch_neg_ce_sum, float(epoch_neg_decision_tokens), default=0.0),
            "delta_logp_pos_mean": _safe_div(
                epoch_delta_logp_pos_sum, float(epoch_pos_decision_tokens), default=0.0
            ),
            "delta_logp_neg_mean": _safe_div(
                epoch_delta_logp_neg_sum, float(epoch_neg_decision_tokens), default=0.0
            ),
            "pos_pressure_sum": float(epoch_pos_pressure_sum),
            "neg_pressure_sum": float(epoch_neg_pressure_sum),
            "neg_pos_pressure_ratio": _safe_div(epoch_neg_pressure_sum, epoch_pos_pressure_sum, default=0.0),
            "num_batches": int(epoch_batches),
            "optimizer_steps": int(optimizer_step),
            "elapsed_sec": float(time.time() - t0),
        }
        train_log["epochs"].append(epoch_rec)
        print(
            f"Epoch {epoch}/{args.epochs}: avg_loss={epoch_rec['avg_loss']:.6f} "
            f"avg_reasonmaxxer={epoch_rec['avg_decision_loss']:.6f} avg_kl={epoch_rec['avg_kl_loss']:.6f} "
            f"decision_tokens={epoch_rec['total_decision_tokens']} "
            f"dec_frac_gen={epoch_rec['decision_frac_generated']:.4f} "
            f"neg/pos={epoch_rec['neg_pos_pressure_ratio']:.3f} "
            f"dlogp(+/-)=({epoch_rec['delta_logp_pos_mean']:.4f},{epoch_rec['delta_logp_neg_mean']:.4f})"
        )

        if args.save_every_epoch:
            ckpt_dir = _epoch_dir(out_dir, epoch)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            train_log["checkpoints"].append(
                {
                    "type": "epoch",
                    "tag": ckpt_dir.name,
                    "path": str(ckpt_dir),
                    "epoch": int(epoch),
                    "epoch_float": float(epoch),
                    "global_step": int(global_step),
                    "optimizer_step": int(optimizer_step),
                }
            )
            _append_checkpoint_metrics_row(
                ckpt_tag=str(ckpt_dir.name),
                ckpt_path=str(ckpt_dir),
                epoch_i=int(epoch),
                epoch_float=float(epoch),
            )
            print(f"[saved] {ckpt_dir}")

        last_epoch = int(epoch)
        (out_dir / "training_log.json").write_text(json.dumps(train_log, indent=2) + "\n", encoding="utf-8")

        if args.early_stop_delta_logp_pos_lt is not None:
            if float(epoch_rec["delta_logp_pos_mean"]) < float(args.early_stop_delta_logp_pos_lt):
                train_log["early_stop"] = {
                    "reason": "delta_logp_pos_below_threshold",
                    "threshold": float(args.early_stop_delta_logp_pos_lt),
                    "epoch": int(epoch),
                    "delta_logp_pos_mean": float(epoch_rec["delta_logp_pos_mean"]),
                }
                print(
                    f"[early-stop] epoch={epoch} delta_logp_pos_mean={epoch_rec['delta_logp_pos_mean']:.6f} "
                    f"< threshold={float(args.early_stop_delta_logp_pos_lt):.6f}"
                )
                break
        if int(args.max_optimizer_steps) > 0 and int(optimizer_step) >= int(args.max_optimizer_steps):
            print(f"[early-stop] reached max_optimizer_steps={int(args.max_optimizer_steps)}")
            break

    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    train_log["checkpoints"].append(
        {
            "type": "final",
            "tag": final_dir.name,
            "path": str(final_dir),
            "epoch": int(last_epoch),
            "epoch_float": float(last_epoch),
            "global_step": int(global_step),
            "optimizer_step": int(optimizer_step),
        }
    )
    if int(optimizer_step) > int(prev_ckpt_opt_step):
        _append_checkpoint_metrics_row(
            ckpt_tag=str(final_dir.name),
            ckpt_path=str(final_dir),
            epoch_i=int(last_epoch),
            epoch_float=float(last_epoch),
        )
    train_log["completed"] = True
    train_log["total_elapsed_sec"] = float(time.time() - t0)
    (out_dir / "training_log.json").write_text(json.dumps(train_log, indent=2) + "\n", encoding="utf-8")

    print(f"[saved] {final_dir}")
    print(f"[saved] {out_dir / 'training_log.json'}")
    print(f"[saved] {metrics_csv_path}")


if __name__ == "__main__":
    main()
