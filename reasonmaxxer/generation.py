from __future__ import annotations

from typing import List


def generate_completions(
    model_name: str,
    prompts: list[str],
    n: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    stop: list[str] | None = None,
    is_chat: bool = False,
    enable_thinking: bool | None = None,
    seed: int | None = None,
    tensor_parallel_size: int = 1,
    max_model_len: int | None = None,
    llm=None,
    lora_request=None,
) -> list[list[str]]:
    """
    Generate n completions for each prompt using vllm.
    Returns a list of lists: outer list is per-prompt, inner list is n completions.
    """
    from vllm import LLM, SamplingParams

    created_llm = False
    if llm is None:
        llm_kwargs = {
            "model": model_name,
            "trust_remote_code": True,
            "tensor_parallel_size": tensor_parallel_size,
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        llm = LLM(**llm_kwargs)
        created_llm = True

    sampling_params = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
        stop=stop,
        include_stop_str_in_output=bool(stop and any(s == "</answer>" for s in stop)),
    )

    if is_chat:
        tokenizer = llm.get_tokenizer()
        model_ref = (model_name or getattr(tokenizer, "name_or_path", "") or "").lower()
        thinking_flag = enable_thinking
        if thinking_flag is None and "qwen3" in model_ref:
            # Qwen3 defaults to thinking=True; default to non-thinking for efficiency.
            thinking_flag = False
        formatted_prompts: List[str] = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if thinking_flag is not None:
                kwargs["enable_thinking"] = bool(thinking_flag)
            try:
                formatted = tokenizer.apply_chat_template(messages, **kwargs)
            except TypeError:
                # Some templates/tokenizers don't support enable_thinking.
                kwargs.pop("enable_thinking", None)
                if thinking_flag is False and "qwen3" in model_ref:
                    messages = [{"role": "user", "content": "/no_think\n" + prompt}]
                formatted = tokenizer.apply_chat_template(messages, **kwargs)
            formatted_prompts.append(formatted)
        prompts = formatted_prompts

    if lora_request is not None:
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    else:
        outputs = llm.generate(prompts, sampling_params)

    results: list[list[str]] = []
    for output in outputs:
        completions = [item.text for item in output.outputs]
        results.append(completions)

    if created_llm:
        del llm

    return results
