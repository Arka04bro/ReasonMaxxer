#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
import os
import signal
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from reasonmaxxer import config
from reasonmaxxer.answer_verification import answers_match
from reasonmaxxer.eval_lib import (
    build_llm,
    build_prompt,
    extract_prediction,
    generate_batch,
    infer_prompt_style,
    load_dataset_records,
    load_problem_ids,
    load_records_file,
    resolve_math500_extractor,
    resolve_stop_sequences,
)


class _MatchTimeoutError(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _MatchTimeoutError()


def _safe_answers_match(predicted: str | None, ground_truth: str | None, timeout_s: float) -> tuple[bool, bool]:
    if timeout_s <= 0 or not hasattr(signal, "setitimer"):
        return bool(answers_match(predicted, ground_truth)), False
    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_s))
    try:
        return bool(answers_match(predicted, ground_truth)), False
    except _MatchTimeoutError:
        return False, True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def _load_existing(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    if not path.exists():
        return [], set()
    payload = json.loads(path.read_text(encoding='utf-8'))
    rows = payload.get('results', []) if isinstance(payload, dict) else []
    done = {str(r.get('problem_id')) for r in rows if isinstance(r, dict) and r.get('problem_id') is not None}
    return rows, done


def main() -> None:
    p = argparse.ArgumentParser(description='Generate ReasonMaxxer rollouts and score them with exact-match answer checking.')
    p.add_argument('--model_path', required=True)
    p.add_argument('--lora_adapter', default=None)
    p.add_argument('--max_lora_rank', type=int, default=32)
    p.add_argument('--lora_name', default=None)
    p.add_argument('--condition_name', required=True)
    p.add_argument('--dataset', required=True, choices=['math500', 'gsm8k', 'aime24', 'amc23', 'minerva_math', 'olympiadbench'])
    p.add_argument('--records_file', default=None)
    p.add_argument('--problem_ids_file', default=None)
    p.add_argument('--problem_ids_key', default='auto', choices=['auto', 'problem_ids', 'holdout_ids', 'train_ids'])
    p.add_argument('--problem_set', default='full')
    p.add_argument('--prompt_style', default='auto')
    p.add_argument('--stop_profile', default='auto')
    p.add_argument('--math500_extractor', default='auto', choices=['auto', 'boxed', 'flexible'])
    p.add_argument('--qwen3_enable_thinking', default='auto', choices=['auto', 'true', 'false'])
    p.add_argument('--num_generations', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    p.add_argument('--tensor_parallel_size', type=int, default=config.TENSOR_PARALLEL_SIZE)
    p.add_argument('--temperature', type=float, default=config.TEMPERATURE)
    p.add_argument('--top_p', type=float, default=config.TOP_P)
    p.add_argument('--max_tokens', type=int, default=config.MAX_TOKENS)
    p.add_argument('--max_model_len', type=int, default=config.MAX_MODEL_LEN)
    p.add_argument('--dtype', default='auto',
                   help="vLLM dtype. Turing GPUs (T4) have no bfloat16, so the pilot passes float16.")
    p.add_argument('--seed', type=int, default=config.SEED)
    p.add_argument('--max_problems', type=int, default=None)
    p.add_argument('--match_timeout_s', type=float, default=0.2)
    p.add_argument('--output_dir', default='outputs/rollouts')
    p.add_argument('--output_name', default=None)
    p.add_argument('--force', action='store_true')
    args = p.parse_args()

    config.TEMPERATURE = float(args.temperature)
    config.TOP_P = float(args.top_p)
    config.MAX_TOKENS = int(args.max_tokens)
    config.MAX_MODEL_LEN = int(args.max_model_len)
    config.SEED = int(args.seed)
    config.TENSOR_PARALLEL_SIZE = int(args.tensor_parallel_size)

    resolved_prompt_style = infer_prompt_style(args.model_path, prompt_style=args.prompt_style)
    resolved_stops = resolve_stop_sequences(args.stop_profile, model_name_or_path=args.model_path)
    resolved_math500_extractor = resolve_math500_extractor(
        args.math500_extractor,
        prompt_style=resolved_prompt_style,
        model_name_or_path=args.model_path,
    )

    if args.records_file:
        records = load_records_file(args.records_file)
    else:
        records = load_dataset_records(args.dataset)
    keep_ids = load_problem_ids(args.problem_ids_file, preferred_key=args.problem_ids_key)
    if keep_ids:
        records = [r for r in records if str(r['problem_id']) in keep_ids]
    if args.max_problems is not None:
        records = records[: args.max_problems]
    if not records:
        raise RuntimeError('No records resolved for generation.')

    output_name = args.output_name or f"{args.condition_name}_{args.dataset}.json"
    out_path = Path(args.output_dir) / output_name
    existing, done = ([], set()) if args.force else _load_existing(out_path)
    pending = [r for r in records if str(r['problem_id']) not in done]
    if not pending:
        print(f'already complete -> {out_path}')
        return

    llm = None
    lora_request = None
    results = list(existing)
    timed_out_matches = 0
    total_matches = 0

    try:
        if args.lora_adapter:
            from vllm.lora.request import LoRARequest
            llm = build_llm(args.model_path, tensor_parallel_size=args.tensor_parallel_size, max_model_len=args.max_model_len, enable_lora=True, max_lora_rank=args.max_lora_rank, dtype=args.dtype)
            lora_request = LoRARequest(args.lora_name or args.condition_name, 1, str(Path(args.lora_adapter)))
        else:
            llm = build_llm(args.model_path, tensor_parallel_size=args.tensor_parallel_size, max_model_len=args.max_model_len, dtype=args.dtype)

        thinking_opt = str(args.qwen3_enable_thinking).strip().lower()
        if thinking_opt == 'true':
            qwen3_enable_thinking: bool | None = True
        elif thinking_opt == 'false':
            qwen3_enable_thinking = False
        else:
            qwen3_enable_thinking = None

        for i in tqdm(range(0, len(pending), args.batch_size), desc=f"{args.condition_name}/{args.dataset}"):
            batch = pending[i : i + args.batch_size]
            prompts = [
                build_prompt(item['problem_text'], dataset=args.dataset, prompt_style=resolved_prompt_style)
                for item in batch
            ]
            completions = generate_batch(
                llm,
                prompts,
                n=args.num_generations,
                model_name=args.model_path,
                prompt_style=resolved_prompt_style,
                stop_sequences=resolved_stops,
                lora_request=lora_request,
                enable_thinking=qwen3_enable_thinking,
            )
            for item, prompt_text, gens in zip(batch, prompts, completions):
                row_gens = []
                for idx, text in enumerate(gens):
                    extracted = extract_prediction(text, dataset=args.dataset, math500_extractor=resolved_math500_extractor)
                    correct, timed_out = _safe_answers_match(extracted, item.get('ground_truth'), timeout_s=args.match_timeout_s)
                    total_matches += 1
                    timed_out_matches += int(timed_out)
                    row_gens.append({
                        'index': idx,
                        'text': text,
                        'extracted_answer': extracted,
                        'correct': bool(correct),
                        'scoring_source': 'rule',
                        'timed_out_match': bool(timed_out),
                    })
                results.append({
                    'problem_id': item['problem_id'],
                    'problem_text': item['problem_text'],
                    'ground_truth': item.get('ground_truth'),
                    'category': item.get('category', args.dataset),
                    'prompt_text': prompt_text,
                    'generations': row_gens,
                })
                payload = {
                    'model_path': args.model_path,
                    'lora_adapter': args.lora_adapter,
                    'condition_name': args.condition_name,
                    'dataset': args.dataset,
                    'prompt_style': resolved_prompt_style,
                    'stop_profile': args.stop_profile,
                    'stop_sequences': resolved_stops,
                    'math500_extractor': resolved_math500_extractor,
                    'is_chat_effective': bool(resolved_prompt_style in {'qwen3_chat', 'chat_template'}),
                    'qwen3_enable_thinking': args.qwen3_enable_thinking,
                    'num_generations': int(args.num_generations),
                    'temperature': float(args.temperature),
                    'top_p': float(args.top_p),
                    'max_tokens': int(args.max_tokens),
                    'seed': int(args.seed),
                    'results': results,
                    'timed_out_matches': int(timed_out_matches),
                    'total_matches': int(total_matches),
                    'created_at_unix': time.time(),
                }
                _save(out_path, payload)
    finally:
        if llm is not None:
            del llm

    print(f'[saved] {out_path}')


if __name__ == '__main__':
    main()
