from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from datasets import load_dataset

from reasonmaxxer import config
from reasonmaxxer.answer_extraction import extract_answer_with_mode, get_ground_truth
from reasonmaxxer.generation import generate_completions

PROMPT_STYLE_VALUES = {
    "qwen_boxed",
    "qwen_math",
    "qwen3_chat",
    "chat_template",
    "olmo3_math",
    "olmo3_rlzero_math",
    "llama_abel",
    "llama_boxed",
    "llama_round0_boxed",
    "oat_r1",
    "raw_problem",
}

STOP_PROFILE_VALUES = {
    "auto",
    "legacy",
    "qwen_math",
    "olmo3_math",
    "olmo3_rlzero_math",
    "llama_eot",
    "llama_turn",
    "oat_r1",
    "none",
}


def _is_reasoning_chat_family(model_name_or_path: str) -> bool:
    model_l = (model_name_or_path or "").lower()
    keywords = (
        "deepseek-r1-distill-qwen",
        "open-reasoner-zero",
        "orz-r1",
        "deepscaler",
        "open-rs",
        "still-",
        "ii-thought-1.5b-preview",
        "tina-yi/r1-distill-qwen-1.5b",
        "deepcoder",
        "coder1",
        "code-r1",
        "mimo",
    )
    return any(k in model_l for k in keywords)


def load_records_file(path: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("records", payload) if isinstance(payload, (dict, list)) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Unsupported records file format: {path}")
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        ptxt = row.get("problem_text") or row.get("problem")
        if ptxt is None:
            continue
        item = dict(row)
        item["problem_id"] = str(row.get("problem_id", f"custom/{idx}"))
        item["problem_text"] = str(ptxt)
        if row.get("ground_truth") is not None:
            item["ground_truth"] = str(row.get("ground_truth"))
        item["category"] = str(row.get("category", "custom"))
        out.append(item)
    if not out:
        raise ValueError(f"No usable rows in records file: {path}")
    return out


def _load_local_benchmark_records(path: Path, dataset: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing benchmark file for dataset={dataset}: {path}. Pass --records_file or place the JSON there."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("records", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Unsupported benchmark format in {path}")
    out = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        ptxt = row.get("problem_text") or row.get("problem")
        gt = row.get("ground_truth") or row.get("answer")
        if ptxt is None or gt is None:
            continue
        out.append(
            {
                "problem_id": str(row.get("problem_id", f"{dataset}/full/{idx}")),
                "problem_text": str(ptxt),
                "ground_truth": str(gt),
                "category": str(row.get("category", dataset)),
                **({"extra_info": row.get("extra_info")} if row.get("extra_info") is not None else {}),
            }
        )
    if not out:
        raise ValueError(f"No valid records found in benchmark file: {path}")
    return out


def load_dataset_records(dataset: str) -> list[dict[str, Any]]:
    if dataset in {"math500", "gsm8k"}:
        dataset_cfg = config.DATASET_CONFIG[dataset]
        split = dataset_cfg["split"]
        hf_name = dataset_cfg["hf_name"]
        config_name = dataset_cfg.get("config_name")
        if config_name:
            rows = load_dataset(hf_name, config_name, split=split)
        else:
            rows = load_dataset(hf_name, split=split)
        out = []
        if dataset == "math500":
            for idx, row in enumerate(rows):
                solution = row.get(dataset_cfg["answer_key"], "")
                out.append(
                    {
                        "problem_id": f"MATH500/{split}/{row.get('type','unknown')}/{idx}",
                        "problem_text": str(row[dataset_cfg["problem_key"]]),
                        "ground_truth": get_ground_truth(solution, dataset="math500"),
                        "category": str(row.get("type", "unknown")),
                    }
                )
        else:
            for idx, row in enumerate(rows):
                answer = row.get(dataset_cfg["answer_key"], "")
                out.append(
                    {
                        "problem_id": f"GSM8K/{split}/{idx}",
                        "problem_text": str(row[dataset_cfg["problem_key"]]),
                        "ground_truth": get_ground_truth(answer, dataset="gsm8k"),
                        "category": "gsm8k",
                    }
                )
        return out
    if dataset in config.BENCHMARK_DATA_FILES:
        return _load_local_benchmark_records(Path(config.BENCHMARK_DATA_FILES[dataset]), dataset)
    raise ValueError(f"Unsupported dataset: {dataset}")


def load_problem_ids(path: str | None, preferred_key: str = "auto") -> set[str]:
    if not path:
        return set()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if preferred_key != "auto":
            value = data.get(preferred_key)
            return {str(x) for x in value} if isinstance(value, list) else set()
        for key in ["problem_ids", "holdout_ids", "train_ids"]:
            if key in data and isinstance(data[key], list):
                return {str(x) for x in data[key]}
        if isinstance(data.get("records"), list):
            return {str(x["problem_id"]) for x in data["records"] if isinstance(x, dict) and x.get("problem_id")}
        return set()
    if isinstance(data, list):
        out = set()
        for item in data:
            if isinstance(item, dict) and item.get("problem_id"):
                out.add(str(item["problem_id"]))
            else:
                out.add(str(item))
        return out
    return set()


def infer_prompt_style(model_name_or_path: str, prompt_style: str = "auto") -> str:
    style = (prompt_style or "auto").strip().lower()
    if style != "auto":
        if style not in PROMPT_STYLE_VALUES:
            raise ValueError(f"Unknown prompt_style={prompt_style}")
        return style
    model_l = (model_name_or_path or "").lower()
    if "oat-zero" in model_l and "qwen" in model_l:
        return "qwen_math"
    if "qwen2.5-math" in model_l or "qwen2-math" in model_l:
        return "qwen_math"
    if "oat-zero" in model_l:
        return "oat_r1"
    if "olmo-3" in model_l and ("rl-zero-math" in model_l or "rlzero-math" in model_l):
        return "olmo3_rlzero_math"
    if "llama" in model_l or "mistral" in model_l:
        return "llama_abel"
    if _is_reasoning_chat_family(model_l):
        return "chat_template"
    if any(x in model_l for x in ("olmo-2", "olmo_2", "olmo-3", "olmo_3")):
        if any(x in model_l for x in ("instruct", "sft", "dpo", "tulu")):
            return "chat_template"
        return "qwen_boxed"
    if "qwen3" in model_l and "base" not in model_l:
        return "qwen3_chat"
    return "qwen_boxed"


def resolve_stop_sequences(stop_profile: str = "auto", model_name_or_path: str | None = None) -> list[str] | None:
    profile = (stop_profile or "auto").strip().lower()
    if profile not in STOP_PROFILE_VALUES:
        raise ValueError(f"Unknown stop_profile={stop_profile}")
    if profile == "none":
        return None
    if profile == "legacy":
        return list(config.STOP_SEQUENCES)
    if profile == "qwen_math":
        return ["<|im_end|>", "<|end_of_text|>", "<|endoftext|>", "<|eot_id|>"]
    if profile == "olmo3_math":
        return ["Problem:", "Answer:", "Question:", "</s>", "<|eot_id|>"]
    if profile == "olmo3_rlzero_math":
        return ["\n\nProblem:", "\nProblem:", "\nQuestion:", "<|endoftext|>", "<|eot_id|>"]
    if profile == "llama_eot":
        return ["<|eot_id|>", "<|end_of_text|>"]
    if profile == "llama_turn":
        return ["\nUSER:", "\n[Round", "<|eot_id|>", "<|end_of_text|>"]
    if profile == "oat_r1":
        return ["</answer>"]

    model_l = (model_name_or_path or "").lower()
    if "oat-zero" in model_l or "qwen2.5-math" in model_l or "qwen2-math" in model_l:
        return ["<|im_end|>", "<|end_of_text|>", "<|endoftext|>", "<|eot_id|>"]
    if "llama" in model_l or "mistral" in model_l:
        return ["<|eot_id|>", "<|end_of_text|>"]
    if "qwen3" in model_l or _is_reasoning_chat_family(model_l):
        return ["<|im_end|>", "<|end_of_text|>", "<|endoftext|>", "<|eot_id|>"]
    if any(x in model_l for x in ("olmo-2", "olmo_2", "olmo-3", "olmo_3")):
        return ["<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>"]
    return list(config.STOP_SEQUENCES)


def resolve_math500_extractor(math500_extractor: str = "auto", *, prompt_style: str | None = None, model_name_or_path: str | None = None) -> str:
    mode = (math500_extractor or "auto").strip().lower()
    if mode in {"boxed", "flexible"}:
        return mode
    if mode != "auto":
        raise ValueError("math500_extractor must be one of: auto, boxed, flexible")
    style = (prompt_style or "").strip().lower()
    if style in {"llama_abel", "oat_r1", "olmo3_math", "olmo3_rlzero_math"}:
        return "flexible"
    model_l = (model_name_or_path or "").lower()
    if _is_reasoning_chat_family(model_l):
        return "flexible"
    if "llama" in model_l or "mistral" in model_l or any(x in model_l for x in ("olmo-2", "olmo_2", "olmo-3", "olmo_3")):
        return "flexible"
    return "boxed"


def build_prompt(problem_text: str, *, dataset: str, prompt_style: str) -> str:
    if prompt_style == "qwen_boxed":
        if dataset == "gsm8k":
            return (
                "Solve the following math problem step by step.\n\n"
                f"{problem_text}\n\n"
                "Solution:"
            )
        return (
            "Solve the following math problem step by step. Put your final answer in \\boxed{}.\n\n"
            f"Problem: {problem_text}\n\n"
            "Solution:"
        )
    if prompt_style == "qwen_math":
        return (
            "<|im_start|>system\n"
            "Please reason step by step, and put your final answer within \\boxed{}."
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{problem_text}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    if prompt_style in {"qwen3_chat", "chat_template"}:
        if dataset == "gsm8k":
            return (
                "Solve the following math problem step by step.\n\n"
                f"{problem_text}\n\n"
                "Give a concise solution and final numeric answer."
            )
        return (
            "Solve the following math problem step by step. "
            "Give a concise solution and put your final answer in \\boxed{}.\n\n"
            f"Problem: {problem_text}"
        )
    if prompt_style == "olmo3_math":
        if dataset == "gsm8k":
            return (
                "Solve the following grade school math word problem:\n"
                f"{problem_text}\n\n"
                "Show your work and conclude with \"Therefore, the final answer is \\boxed{answer}.\""
            )
        return f"{problem_text}\n\nPresent the answer in LaTex format: \\boxed{{Your answer}}"
    if prompt_style == "olmo3_rlzero_math":
        return (
            "Solve the following math problem step by step. "
            "The last line of your response should be the answer to the problem in form "
            "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
            f"{problem_text}\n\n"
            'Remember to put your answer on its own line after "Answer:"'
        )
    if prompt_style == "llama_abel":
        return f"Question:\n{problem_text}\nAnswer:\nLet's think step by step.\n"
    if prompt_style == "llama_boxed":
        return (
            "Solve the following math problem step by step. "
            "Put your final answer in \\boxed{}.\n\n"
            f"{problem_text}\n\n"
            "Solution:"
        )
    if prompt_style == "llama_round0_boxed":
        if dataset == "gsm8k":
            return (
                "[Round 0]\n"
                f"USER: {problem_text}\n"
                "Please reason step by step and provide the final numeric answer.\n"
                "ASSISTANT:"
            )
        return (
            "[Round 0]\n"
            f"USER: {problem_text}\n"
            "Please reason step by step, and put your final answer within \\boxed{}.\n"
            "ASSISTANT:"
        )
    if prompt_style == "oat_r1":
        return (
            "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
            "The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
            "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively.\n"
            f"User: {problem_text}\n"
            "Assistant: <think>"
        )
    if prompt_style == "raw_problem":
        return problem_text
    raise ValueError(f"Unsupported prompt_style: {prompt_style}")


def extract_prediction(text: str, *, dataset: str, math500_extractor: str) -> str | None:
    return extract_answer_with_mode(text, dataset=dataset, math500_extractor=math500_extractor)


def build_llm(
    model_name: str,
    *,
    tensor_parallel_size: int | None = None,
    max_model_len: int | None = None,
    enable_lora: bool = False,
    max_lora_rank: int = 32,
    dtype: str | None = None,
):
    from vllm import LLM

    kwargs = {
        "model": model_name,
        "trust_remote_code": True,
        "tensor_parallel_size": int(config.TENSOR_PARALLEL_SIZE if tensor_parallel_size is None else tensor_parallel_size),
        "max_model_len": int(config.MAX_MODEL_LEN if max_model_len is None else max_model_len),
        "disable_log_stats": True,
    }
    # vLLM's dtype="auto" resolves to the checkpoint's own dtype, which is
    # bfloat16 for Qwen2.5. Turing (T4, sm_75) has no bf16, so the pilot has to
    # ask for float16 explicitly rather than letting it be inferred.
    resolved_dtype = (dtype or os.environ.get("VLLM_DTYPE", "") or "auto").strip()
    if resolved_dtype and resolved_dtype != "auto":
        kwargs["dtype"] = resolved_dtype
    if enable_lora:
        kwargs["enable_lora"] = True
        kwargs["max_lora_rank"] = int(max_lora_rank)
    tokenizer_mode = os.environ.get("VLLM_TOKENIZER_MODE", "").strip()
    if tokenizer_mode:
        kwargs["tokenizer_mode"] = tokenizer_mode
    tokenizer_path = os.environ.get("VLLM_TOKENIZER", "").strip()
    if tokenizer_path:
        kwargs["tokenizer"] = tokenizer_path
    return LLM(**kwargs)


def generate_batch(
    llm,
    prompts: list[str],
    *,
    n: int,
    model_name: str,
    prompt_style: str,
    stop_sequences: list[str] | None,
    lora_request=None,
    enable_thinking: bool | None = None,
):
    effective_is_chat = prompt_style in {"qwen3_chat", "chat_template"}
    try:
        return generate_completions(
            model_name=model_name,
            prompts=prompts,
            n=n,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            max_tokens=config.MAX_TOKENS,
            stop=stop_sequences,
            is_chat=effective_is_chat,
            enable_thinking=enable_thinking,
            seed=config.SEED,
            tensor_parallel_size=config.TENSOR_PARALLEL_SIZE,
            max_model_len=config.MAX_MODEL_LEN,
            llm=llm,
            lora_request=lora_request,
        )
    except Exception:
        outputs = []
        for prompt in prompts:
            try:
                single = generate_completions(
                    model_name=model_name,
                    prompts=[prompt],
                    n=n,
                    temperature=config.TEMPERATURE,
                    top_p=config.TOP_P,
                    max_tokens=config.MAX_TOKENS,
                    stop=stop_sequences,
                    is_chat=effective_is_chat,
                    enable_thinking=enable_thinking,
                    seed=config.SEED,
                    tensor_parallel_size=config.TENSOR_PARALLEL_SIZE,
                    max_model_len=config.MAX_MODEL_LEN,
                    llm=llm,
                    lora_request=lora_request,
                )[0]
            except Exception:
                single = []
            outputs.append(single)
        return outputs
