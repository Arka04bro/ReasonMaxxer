from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from reasonmaxxer import eval_lib as ev

try:
    from peft import PeftModel
except Exception:  # pragma: no cover
    PeftModel = None  # type: ignore[assignment]


def _resolve_dtype(name: str) -> torch.dtype | None:
    value = str(name).strip().lower()
    if value == "auto":
        return None
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _load_base_model(model_name_or_path: str, *, dtype: str, hf_attn_implementation: str, local_files_only: bool):
    torch_dtype = _resolve_dtype(dtype)
    return AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
        torch_dtype=torch_dtype if torch_dtype is not None else "auto",
        attn_implementation=hf_attn_implementation,
    ).to("cuda")


def _load_model(
    model_name_or_path: str,
    *,
    dtype: str,
    hf_attn_implementation: str,
    local_files_only: bool,
    adapter_path: str = "",
    base_model_name_or_path: str = "",
):
    adapter_path = adapter_path.strip()
    base_model_name_or_path = base_model_name_or_path.strip()
    if adapter_path:
        if PeftModel is None:
            raise RuntimeError("peft is required for adapter-based scoring.")
        base_name = base_model_name_or_path or model_name_or_path
        base = _load_base_model(
            base_name,
            dtype=dtype,
            hf_attn_implementation=hf_attn_implementation,
            local_files_only=local_files_only,
        )
        return PeftModel.from_pretrained(base, adapter_path, is_trainable=False).to("cuda")
    return _load_base_model(
        model_name_or_path,
        dtype=dtype,
        hf_attn_implementation=hf_attn_implementation,
        local_files_only=local_files_only,
    )


def _chat_prompt_string(tokenizer, prompt_text: str, *, enable_thinking: bool | None) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        return prompt_text
    messages = [{"role": "user", "content": prompt_text}]
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if enable_thinking is not None:
        kwargs["enable_thinking"] = bool(enable_thinking)
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def _compose_full_text(
    *,
    prompt_text: str,
    completion_text: str,
    prompt_style: str,
    tokenizer,
    enable_thinking: bool | None,
) -> tuple[str, str]:
    if completion_text.startswith(prompt_text):
        rendered_prompt = prompt_text
        return completion_text, rendered_prompt
    if prompt_style in {"chat_template", "qwen3_chat"}:
        rendered_prompt = _chat_prompt_string(tokenizer, prompt_text, enable_thinking=enable_thinking)
        return rendered_prompt + completion_text, rendered_prompt
    return prompt_text + completion_text, prompt_text


def _entropy_from_logits(logits: torch.Tensor) -> list[float]:
    logp = torch.log_softmax(logits.float(), dim=-1)
    probs = torch.exp(logp)
    return [float(x) for x in (-(probs * logp).sum(dim=-1)).tolist()]


def _stream_writer(path: Path, meta: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("w", encoding="utf-8")
    f.write("{\n")
    f.write('  "meta": ')
    f.write(json.dumps(meta, ensure_ascii=False, indent=2))
    f.write(",\n")
    f.write('  "rollouts": [\n')
    return f


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Teacher-forced HF entropy scoring for ReasonMaxxer rollout JSONs.")
    p.add_argument("--input_json", required=True, help="generate_rollouts.py-style json")
    p.add_argument("--output_file", required=True, help="teacher-forced rollout-entropy json")
    p.add_argument("--model_path", required=True, help="HF model used for teacher-forced entropy scoring")
    p.add_argument("--lora_adapter", default="", help="Optional adapter path to score with")
    p.add_argument("--base_model_name_or_path", default="", help="Base model path when --model_path is adapter-only")
    p.add_argument("--tokenizer_model", default="", help="Defaults to scorer base/model")
    p.add_argument("--dataset", default="", help="Override dataset if missing in input")
    p.add_argument("--prompt_style", default="auto", help="Override prompt style if needed")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--hf_attn_implementation", default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--qwen3_enable_thinking", default="auto", choices=["auto", "true", "false"])
    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--max_problems", type=int, default=-1)
    p.add_argument("--n_gens", type=int, default=-1, help="<=0 means all generations")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input_json)
    out_path = Path(args.output_file)
    if out_path.exists() and not args.force:
        raise FileExistsError(f"{out_path} exists. Use --force to overwrite.")

    payload = json.loads(in_path.read_text(encoding="utf-8"))
    dataset = args.dataset.strip() or str(payload.get("dataset", "")).strip()
    if not dataset:
        raise ValueError("Could not resolve dataset. Pass --dataset.")
    prompt_style_payload = str(payload.get("prompt_style", "") or "").strip().lower()
    if args.prompt_style == "auto":
        resolved_prompt_style = prompt_style_payload or ev.infer_prompt_style(args.model_path, prompt_style="auto")
    else:
        resolved_prompt_style = ev.infer_prompt_style(args.model_path, prompt_style=args.prompt_style)

    thinking_opt = str(args.qwen3_enable_thinking).strip().lower()
    if thinking_opt == "true":
        qwen3_enable_thinking: bool | None = True
    elif thinking_opt == "false":
        qwen3_enable_thinking = False
    else:
        qwen3_enable_thinking = None

    tokenizer_name = args.tokenizer_model.strip() or args.base_model_name_or_path.strip() or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_model(
        args.model_path,
        dtype=args.dtype,
        hf_attn_implementation=args.hf_attn_implementation,
        local_files_only=args.local_files_only,
        adapter_path=args.lora_adapter,
        base_model_name_or_path=args.base_model_name_or_path,
    )
    model.eval()

    rows = payload.get("results", [])
    if not isinstance(rows, list):
        raise ValueError(f"Unexpected input format: {in_path}")
    if args.max_problems > 0:
        rows = rows[: args.max_problems]

    meta = {
        "source_eval_json": str(in_path),
        "model_path": args.model_path,
        "lora_adapter": args.lora_adapter,
        "dataset": dataset,
        "problem_set": payload.get("problem_set", ""),
        "prompt_style": resolved_prompt_style,
        "qwen3_enable_thinking": args.qwen3_enable_thinking,
        "max_seq_len": int(args.max_seq_len),
        "created_at_unix": time.time(),
    }

    f = _stream_writer(out_path, meta)
    first = True
    n_rollouts = 0
    n_trunc = 0
    try:
        for row in tqdm(rows, desc="problems"):
            problem_id = str(row.get("problem_id"))
            problem_text = str(row.get("problem_text", ""))
            ground_truth = row.get("ground_truth")
            category = str(row.get("category", "unknown"))
            prompt_text = str(row.get("prompt_text", "") or "")
            if not prompt_text:
                prompt_text = ev.build_evaluation_prompt(
                    "base",
                    problem_text,
                    dataset=dataset,
                    problem_id=problem_id,
                    prompt_style=resolved_prompt_style,
                )

            generations = row.get("generations", [])
            if args.n_gens > 0:
                generations = generations[: args.n_gens]

            for gi, gen in enumerate(generations):
                gen_idx = int(gen.get("index", gi))
                completion_text = str(gen.get("text", "") or "")
                full_text, rendered_prompt = _compose_full_text(
                    prompt_text=prompt_text,
                    completion_text=completion_text,
                    prompt_style=resolved_prompt_style,
                    tokenizer=tokenizer,
                    enable_thinking=qwen3_enable_thinking,
                )

                enc_prompt = tokenizer(
                    rendered_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_seq_len,
                )
                prompt_len = int(enc_prompt["input_ids"].shape[1])

                enc = tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_seq_len,
                )
                input_ids = enc["input_ids"].to("cuda")
                attention_mask = enc["attention_mask"].to("cuda")
                raw_len = len(tokenizer(full_text, add_special_tokens=True)["input_ids"])
                was_truncated = raw_len > args.max_seq_len
                n_trunc += int(was_truncated)

                seq_len = int(attention_mask.shape[1])
                if seq_len <= prompt_len:
                    continue

                with torch.no_grad():
                    logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    ).logits[0, :-1, :].detach().float().cpu()

                full_ids = enc["input_ids"][0].detach().cpu().tolist()
                gen_pred_start = max(prompt_len - 1, 0)
                end = seq_len - 1
                if end <= gen_pred_start:
                    continue

                entropy_rows = _entropy_from_logits(logits[gen_pred_start:end])
                completion_tokens = full_ids[prompt_len:seq_len]

                item = {
                    "problem_id": problem_id,
                    "category": category,
                    "problem_text": problem_text,
                    "gen_index": gen_idx,
                    "prompt_text": prompt_text,
                    "text": completion_text,
                    "full_text": full_text,
                    "tokens": [int(x) for x in full_ids[:seq_len]],
                    "prompt_length": int(prompt_len),
                    "completion_tokens": [int(x) for x in completion_tokens],
                    "num_tokens": int(seq_len),
                    "num_completion_tokens": int(len(completion_tokens)),
                    "entropies": [float(x) for x in entropy_rows],
                    "extracted_answer": gen.get("extracted_answer"),
                    "ground_truth": ground_truth,
                    "correct": bool(gen.get("correct", False)),
                    "scoring_source": gen.get("scoring_source"),
                    "verifier_used": gen.get("verifier_used"),
                    "verifier_student_answer": gen.get("verifier_student_answer"),
                    "verifier_response": gen.get("verifier_response"),
                    "truncated": bool(was_truncated),
                }
                if not first:
                    f.write(",\n")
                f.write("    ")
                f.write(json.dumps(item, ensure_ascii=False))
                first = False
                n_rollouts += 1
    finally:
        f.write("\n  ]\n}\n")
        f.close()
        del model
        try:
            import gc

            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    stats_path = out_path.with_suffix(out_path.suffix + ".stats.json")
    stats = {
        "input_json": str(in_path),
        "output_file": str(out_path),
        "num_rollouts": n_rollouts,
        "truncated_sequences": n_trunc,
        "dataset": dataset,
        "model_path": args.model_path,
        "lora_adapter": args.lora_adapter,
        "prompt_style": resolved_prompt_style,
    }
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {out_path}")
    print(f"[saved] {stats_path}")
    print(f"[done] rollouts={n_rollouts} truncated={n_trunc}")


if __name__ == "__main__":
    main()
