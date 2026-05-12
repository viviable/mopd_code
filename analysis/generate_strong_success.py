#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None


ANSWER_RE = re.compile(r"<answer>\s*([A-D])\s*</answer>", re.IGNORECASE | re.DOTALL)
SYSTEM_SPLIT = "\nsystem\n\n"
USER_SPLIT = "\n\nuser\n"
ASSISTANT_SPLIT = "\nassistant\n"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def default_device() -> str:
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate stronger-model successful responses from candidate prompts.")
    parser.add_argument("--candidates", required=True, help="Path to candidate_responses.jsonl.")
    parser.add_argument("--output", required=True, help="Output strong_success.jsonl path.")
    parser.add_argument(
        "--aggregate-output",
        help="Optional JSON summary path. Defaults to <output>.summary.json.",
    )
    parser.add_argument("--model", required=True, help="Model name or local model path used for generation.")
    parser.add_argument(
        "--backend",
        choices=["openai", "openrouter", "local"],
        default="openai",
        help="Generation backend. Use 'local' for a local/HF causal LM or 'openrouter' for OpenRouter's OpenAI-compatible API.",
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="API key environment variable.")
    parser.add_argument("--base-url", default=None, help="Optional OpenAI-compatible base URL.")
    parser.add_argument("--http-referer", default=None, help="Optional HTTP-Referer header for OpenRouter.")
    parser.add_argument("--x-title", default=None, help="Optional X-Title header for OpenRouter.")
    parser.add_argument("--max-prompts", type=int, default=None, help="Optional cap on prompts to generate.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Generation temperature.")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Maximum new tokens to generate.")
    parser.add_argument("--max-retries", type=int, default=5, help="Retries for transient API failures.")
    parser.add_argument("--sleep-seconds", type=float, default=1.0, help="Base backoff sleep in seconds.")
    parser.add_argument("--device", default=default_device(), help="Torch device for local generation.")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Torch dtype for local generation.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def make_client(api_key: str, base_url: str | None, default_headers: dict[str, str] | None = None) -> Any:
    if OpenAI is None:
        raise RuntimeError(
            "The 'openai' package is not installed. Install it with `python3 -m pip install openai`."
        )
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    if default_headers:
        kwargs["default_headers"] = default_headers
    return OpenAI(**kwargs)


def resolve_api_config(args: argparse.Namespace) -> tuple[str, str | None, dict[str, str] | None]:
    if args.backend == "openrouter":
        api_key_env = args.api_key_env if args.api_key_env != "OPENAI_API_KEY" else "OPENROUTER_API_KEY"
        base_url = args.base_url or OPENROUTER_BASE_URL
        headers: dict[str, str] = {}
        if args.http_referer:
            headers["HTTP-Referer"] = args.http_referer
        if args.x_title:
            headers["X-Title"] = args.x_title
        return api_key_env, base_url, headers or None
    return args.api_key_env, args.base_url, None


def resolve_dtype(dtype_name: str) -> Any:
    if dtype_name == "auto":
        return None
    if torch is None:
        raise RuntimeError("torch is required for local generation.")
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


def load_local_generator(model_name: str, device: str, dtype_name: str) -> tuple[Any, Any]:
    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError(
            "Local generation requires torch and transformers. Install them before using `--backend local`."
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    dtype = resolve_dtype(dtype_name)
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.to(device)
    model.eval()
    return tokenizer, model


def extract_answer_letter(text: str) -> str | None:
    match = ANSWER_RE.search(text)
    return match.group(1).upper() if match else None


def parse_flat_prompt(prompt_text: str) -> list[dict[str, str]]:
    text = prompt_text
    if text.startswith("system\n\n"):
        text = "\n" + text
    if SYSTEM_SPLIT in text and USER_SPLIT in text and ASSISTANT_SPLIT in text:
        system_part = text.split(SYSTEM_SPLIT, 1)[1].split(USER_SPLIT, 1)[0]
        user_part = text.split(USER_SPLIT, 1)[1].rsplit(ASSISTANT_SPLIT, 1)[0]
        return [
            {"role": "system", "content": system_part},
            {"role": "user", "content": user_part},
        ]
    return [{"role": "user", "content": prompt_text}]


def render_local_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return "\n\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\n\nassistant:\n"


def select_prompt_rows(candidates: list[dict[str, Any]], max_prompts: int | None) -> list[dict[str, Any]]:
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_prompt[row["prompt_id"]].append(row)

    selected = []
    for prompt_id, rows in sorted(by_prompt.items()):
        rows = sorted(rows, key=lambda x: x["response_id"])
        anchor = rows[0]
        selected.append(
            {
                "prompt_id": prompt_id,
                "prompt": anchor["prompt"],
                "ground_truth": anchor.get("ground_truth"),
                "difficulty_bucket": anchor.get("difficulty_bucket", "unknown"),
            }
        )
        if max_prompts is not None and len(selected) >= max_prompts:
            break
    return selected


def generate_one(
    client: Any,
    model: str,
    prompt_text: str,
    temperature: float,
    max_new_tokens: int,
    max_retries: int,
    sleep_seconds: float,
) -> str:
    messages = parse_flat_prompt(prompt_text)
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=messages,
                max_tokens=max_new_tokens,
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty generation content from model.")
            return content
        except Exception as exc:  # pragma: no cover
            last_error = exc
            if attempt + 1 == max_retries:
                break
            time.sleep(sleep_seconds * (2**attempt))
    raise RuntimeError(f"Generation failed after {max_retries} attempts: {last_error}")


def generate_one_local(
    tokenizer: Any,
    model: Any,
    device: str,
    prompt_text: str,
    temperature: float,
    max_new_tokens: int,
) -> str:
    messages = parse_flat_prompt(prompt_text)
    rendered_prompt = render_local_prompt(tokenizer, messages)
    inputs = tokenizer(rendered_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    generation_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    }
    if attention_mask is not None:
        generation_kwargs["attention_mask"] = attention_mask
    if temperature > 0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature
    else:
        generation_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**generation_kwargs)
    generated_ids = outputs[0][input_ids.shape[1] :]
    content = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if not content:
        raise ValueError("Empty generation content from local model.")
    return content


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> int:
    args = parse_args()
    candidates = read_jsonl(Path(args.candidates))
    prompt_rows = select_prompt_rows(candidates, max_prompts=args.max_prompts)
    client = None
    tokenizer = None
    local_model = None
    if args.backend in {"openai", "openrouter"}:
        api_key_env, base_url, default_headers = resolve_api_config(args)
        api_key = os.environ.get(api_key_env)
        if not api_key:
            print(f"Missing API key in environment variable: {api_key_env}", file=sys.stderr)
            return 1
        client = make_client(api_key=api_key, base_url=base_url, default_headers=default_headers)
    else:
        tokenizer, local_model = load_local_generator(args.model, device=args.device, dtype_name=args.dtype)

    strong_success_rows = []
    generated_count = 0
    success_count = 0
    for item in prompt_rows:
        if args.backend in {"openai", "openrouter"}:
            generated = generate_one(
                client=client,
                model=args.model,
                prompt_text=item["prompt"],
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                max_retries=args.max_retries,
                sleep_seconds=args.sleep_seconds,
            )
        else:
            generated = generate_one_local(
                tokenizer=tokenizer,
                model=local_model,
                device=args.device,
                prompt_text=item["prompt"],
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )
        pred_answer = extract_answer_letter(generated)
        gt = item.get("ground_truth")
        validated_success = pred_answer is not None and gt is not None and pred_answer == str(gt).strip().upper()
        generated_count += 1
        if validated_success:
            success_count += 1
            strong_success_rows.append(
                {
                    "prompt_id": item["prompt_id"],
                    "response_text": generated,
                    "generator_model": args.model,
                    "source_model": args.model,
                    "validated_success": True,
                    "ground_truth": gt,
                    "predicted_answer": pred_answer,
                    "difficulty_bucket": item["difficulty_bucket"],
                    "judge_info": {"method": "local_answer_match"},
                }
            )

    output_path = Path(args.output)
    aggregate_output = Path(args.aggregate_output) if args.aggregate_output else output_path.with_suffix(".summary.json")
    write_jsonl(output_path, strong_success_rows)
    write_json(
        aggregate_output,
        {
            "candidates": str(args.candidates),
            "model": args.model,
            "backend": args.backend,
            "generated_prompt_count": generated_count,
            "validated_success_count": success_count,
            "validated_success_rate": (success_count / generated_count) if generated_count else 0.0,
        },
    )
    print(f"Wrote strong successes to {output_path}")
    print(f"Wrote summary to {aggregate_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
