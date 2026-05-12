#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score teacher logprobs for context variants and targets.")
    parser.add_argument("--variants", required=True, help="Path to context_variants.jsonl.")
    parser.add_argument("--targets", required=True, help="Path to target_responses.jsonl.")
    parser.add_argument("--output", required=True, help="Output teacher_scores.jsonl path.")
    parser.add_argument("--model", required=True, help="Teacher model path or HF id.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device.")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--max-examples", type=int, default=None, help="Optional cap on scored examples.")
    parser.add_argument("--max-model-len", type=int, default=8192, help="Maximum combined context+response token length.")
    parser.add_argument(
        "--condition-filter",
        nargs="*",
        help="Optional list of conditions to score. If omitted, score all conditions.",
    )
    parser.add_argument(
        "--target-type-filter",
        nargs="*",
        help="Optional list of target types to score. If omitted, score all target types.",
    )
    parser.add_argument(
        "--self-target-only",
        action="store_true",
        help=(
            "Score each context variant only against its own rollout response. "
            "In this mode, --targets should point to candidate_responses.jsonl, and rows are joined by response_id."
        ),
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


def resolve_dtype(dtype_name: str) -> torch.dtype | None:
    if dtype_name == "auto":
        return None
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


def load_model_and_tokenizer(model_name: str, device: str, dtype_name: str):
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError(
            "transformers is not installed. Install it with `python3 -m pip install transformers`."
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


def build_target_index(targets: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_prompt: dict[str, list[dict[str, Any]]] = {}
    for row in targets:
        by_prompt.setdefault(row["prompt_id"], []).append(row)
    return by_prompt


def build_candidate_index_by_response_id(targets: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["response_id"]: row for row in targets}


def count_total_examples(
    variants: list[dict[str, Any]],
    targets_by_prompt: dict[str, list[dict[str, Any]]],
    max_examples: int | None,
) -> int:
    total = 0
    for variant in variants:
        total += len(targets_by_prompt.get(variant["prompt_id"], []))
        if max_examples is not None and total >= max_examples:
            return max_examples
    return total


def count_total_self_examples(
    variants: list[dict[str, Any]],
    targets_by_response_id: dict[str, dict[str, Any]],
    max_examples: int | None,
) -> int:
    total = 0
    for variant in variants:
        if variant["response_id"] in targets_by_response_id:
            total += 1
        if max_examples is not None and total >= max_examples:
            return max_examples
    return total


def maybe_truncate_context(context_ids: list[int], response_ids: list[int], max_model_len: int) -> tuple[list[int], bool]:
    total_len = len(context_ids) + len(response_ids)
    if total_len <= max_model_len:
        return context_ids, False
    allowed_context_len = max(0, max_model_len - len(response_ids))
    return context_ids[-allowed_context_len:], True


@torch.no_grad()
def score_one(
    tokenizer,
    model,
    device: str,
    context_text: str,
    response_text: str,
    max_model_len: int,
) -> dict[str, Any]:
    context_ids = tokenizer(context_text, add_special_tokens=False).input_ids
    response_ids = tokenizer(response_text, add_special_tokens=False).input_ids

    if len(response_ids) == 0:
        return {
            "teacher_seq_logprob": 0.0,
            "teacher_avg_token_logprob": 0.0,
            "teacher_score": 0.0,
            "teacher_response_len": 0,
            "teacher_prompt_len": len(context_ids),
            "teacher_prompt_truncated": False,
        }

    context_ids, truncated = maybe_truncate_context(context_ids, response_ids, max_model_len=max_model_len)

    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    prefix_ids = ([bos_token_id] if bos_token_id is not None else []) + context_ids
    full_ids = prefix_ids + response_ids

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    response_start = len(prefix_ids) - 1
    response_token_log_probs = token_log_probs[:, response_start:]
    seq_logprob = float(response_token_log_probs.sum().item())
    avg_logprob = float(response_token_log_probs.mean().item())

    return {
        "teacher_seq_logprob": seq_logprob,
        "teacher_avg_token_logprob": avg_logprob,
        "teacher_score": avg_logprob,
        "teacher_response_len": len(response_ids),
        "teacher_prompt_len": len(context_ids),
        "teacher_prompt_truncated": truncated,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    variants = read_jsonl(Path(args.variants))
    targets = read_jsonl(Path(args.targets))

    if args.condition_filter:
        allowed_conditions = set(args.condition_filter)
        variants = [row for row in variants if row["condition"] in allowed_conditions]
    if args.target_type_filter:
        allowed_target_types = set(args.target_type_filter)
        targets = [row for row in targets if row.get("target_type", "self_response") in allowed_target_types]

    targets_by_prompt = None
    targets_by_response_id = None
    if args.self_target_only:
        targets_by_response_id = build_candidate_index_by_response_id(targets)
    else:
        targets_by_prompt = build_target_index(targets)
    tokenizer, model = load_model_and_tokenizer(args.model, device=args.device, dtype_name=args.dtype)
    total_examples = (
        count_total_self_examples(variants, targets_by_response_id, args.max_examples)
        if args.self_target_only
        else count_total_examples(variants, targets_by_prompt, args.max_examples)
    )
    progress = tqdm(total=total_examples, desc="Scoring teacher contexts", unit="example") if tqdm is not None else None

    rows_out: list[dict[str, Any]] = []
    processed = 0
    try:
        for variant in variants:
            if args.self_target_only:
                target = targets_by_response_id.get(variant["response_id"])
                prompt_targets = [target] if target is not None else []
            else:
                prompt_targets = targets_by_prompt.get(variant["prompt_id"], [])
            for target in prompt_targets:
                score_dict = score_one(
                    tokenizer=tokenizer,
                    model=model,
                    device=args.device,
                    context_text=variant["assembled_context_text"],
                    response_text=target["response_text"],
                    max_model_len=args.max_model_len,
                )
                rows_out.append(
                    {
                        "variant_id": variant["variant_id"],
                        "target_id": target.get("target_id", target.get("response_id")),
                        "condition": variant["condition"],
                        "target_type": target.get("target_type", "self_response"),
                        "prompt_id": variant["prompt_id"],
                        "response_id": target.get("source_response_id", target.get("response_id")),
                        "source_model": target.get("source_model", "student_rollout"),
                        "teacher_model": args.model,
                        "sections_used": variant.get("sections_used", {}),
                        "variant_spec": variant.get("variant_spec", {}),
                        **score_dict,
                        "reward": target.get("reward"),
                        "success": target["success"],
                        "difficulty_bucket": target.get("difficulty_bucket", variant.get("difficulty_bucket", "unknown")),
                    }
                )
                processed += 1
                if progress is not None:
                    progress.update(1)
                if args.max_examples is not None and processed >= args.max_examples:
                    write_jsonl(Path(args.output), rows_out)
                    print(f"Wrote teacher scores to {args.output}")
                    return 0
    finally:
        if progress is not None:
            progress.close()

    write_jsonl(Path(args.output), rows_out)
    print(f"Wrote teacher scores to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
