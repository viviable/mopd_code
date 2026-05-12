#!/usr/bin/env python3
import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL | re.IGNORECASE)
SUMMARY_FALLBACK_CHARS = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare fixed offline candidate/evidence datasets from rollout JSONL dumps."
    )
    parser.add_argument("--input", required=True, help="Input rollout JSONL file, directory, or glob pattern.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory. Writes candidate_responses.jsonl, evidence_items.jsonl, and dataset_summary.json.",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=1.0,
        help="Reward threshold used to derive the success label.",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Optional cap on prompts after bucketing/sampling.",
    )
    parser.add_argument(
        "--max-responses-per-prompt",
        type=int,
        default=8,
        help="Maximum number of responses kept for each prompt.",
    )
    parser.add_argument(
        "--min-responses-per-prompt",
        type=int,
        default=2,
        help="Minimum responses required to keep a prompt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed.",
    )
    return parser.parse_args()


def resolve_input_paths(input_arg: str) -> list[Path]:
    input_path = Path(input_arg)
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(p for p in input_path.rglob("*.jsonl") if p.is_file())
    matches = sorted(Path().glob(input_arg))
    return [p for p in matches if p.is_file()]


def extract_summary(output_text: str) -> str | None:
    match = SUMMARY_RE.search(output_text)
    return match.group(1).strip() if match else None


def stable_prompt_id(prompt_text: str) -> str:
    return hashlib.sha1(prompt_text.encode("utf-8")).hexdigest()[:16]


def difficulty_bucket(success_count: int, total_count: int) -> str:
    if total_count <= 0:
        return "unknown"
    if success_count >= 4:
        return "easy"
    if success_count >= 1:
        return "medium"
    return "hard"


def normalize_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def read_rollouts(paths: list[Path], success_threshold: float) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        with path.open() as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                prompt = normalize_text(record.get("input", ""))
                output = normalize_text(record.get("output", ""))
                reward_value = float(record.get("score", 0.0) or 0.0)
                prompt_id = stable_prompt_id(prompt)
                response_idx = len(grouped[prompt_id])
                response_id = f"{prompt_id}_r{response_idx}"
                feedback = normalize_text(record.get("feedback", "")).strip()
                summary_text = extract_summary(output)
                fallback_summary = output[:SUMMARY_FALLBACK_CHARS].strip()

                grouped[prompt_id].append(
                    {
                        "prompt_id": prompt_id,
                        "uid": prompt_id,
                        "prompt": prompt,
                        "response_id": response_id,
                        "response_text": output,
                        "reward": reward_value,
                        "success": reward_value >= success_threshold,
                        "summary_text": summary_text,
                        "summary_text_effective": summary_text if summary_text else fallback_summary,
                        "summary_has_tag": summary_text is not None,
                        "summary_is_fallback": summary_text is None,
                        "feedback_text": feedback,
                        "has_feedback": bool(feedback),
                        "ground_truth": record.get("gts"),
                        "pred": record.get("pred"),
                        "acc": record.get("acc"),
                        "incorrect_format": record.get("incorrect_format"),
                        "source_file": str(path),
                        "source_line": line_no,
                        "rollout_step": record.get("step"),
                    }
                )
    return grouped


def sample_groups(
    grouped: dict[str, list[dict[str, Any]]],
    max_prompts: int | None,
    max_responses_per_prompt: int,
    min_responses_per_prompt: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    kept_groups: list[tuple[str, list[dict[str, Any]], str]] = []

    for prompt_id, rows in grouped.items():
        if len(rows) < min_responses_per_prompt:
            continue
        sampled = list(rows)
        if len(sampled) > max_responses_per_prompt:
            sampled = rng.sample(sampled, max_responses_per_prompt)
            sampled.sort(key=lambda x: x["response_id"])
        success_count = sum(1 for row in sampled if row["success"])
        bucket = difficulty_bucket(success_count, len(sampled))
        for row in sampled:
            row["difficulty_bucket"] = bucket
            row["prompt_group_size"] = len(sampled)
            row["prompt_success_count"] = success_count
        kept_groups.append((prompt_id, sampled, bucket))

    if max_prompts is not None and len(kept_groups) > max_prompts:
        bucket_to_groups: dict[str, list[tuple[str, list[dict[str, Any]], str]]] = defaultdict(list)
        for item in kept_groups:
            bucket_to_groups[item[2]].append(item)

        selected: list[tuple[str, list[dict[str, Any]], str]] = []
        buckets = sorted(bucket_to_groups)
        per_bucket = max_prompts // len(buckets) if buckets else 0
        remainder = max_prompts % len(buckets) if buckets else 0
        for idx, bucket in enumerate(buckets):
            groups = bucket_to_groups[bucket]
            rng.shuffle(groups)
            take = per_bucket + (1 if idx < remainder else 0)
            selected.extend(groups[:take])
        kept_groups = selected[:max_prompts]

    rows_out = [row for _, rows, _ in kept_groups for row in rows]
    bucket_counts = defaultdict(int)
    for _, _, bucket in kept_groups:
        bucket_counts[bucket] += 1

    summary = {
        "prompt_count": len(kept_groups),
        "response_count": len(rows_out),
        "bucket_prompt_counts": dict(bucket_counts),
        "mean_responses_per_prompt": (len(rows_out) / len(kept_groups)) if kept_groups else 0.0,
        "mean_successes_per_prompt": (
            sum(rows[0]["prompt_success_count"] for _, rows, _ in kept_groups) / len(kept_groups)
        ) if kept_groups else 0.0,
        "summary_tag_rate": (
            sum(1 for row in rows_out if row["summary_has_tag"]) / len(rows_out)
        ) if rows_out else 0.0,
        "feedback_rate": (
            sum(1 for row in rows_out if row["has_feedback"]) / len(rows_out)
        ) if rows_out else 0.0,
    }
    return rows_out, summary


def build_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for row in rows:
        prompt_failure_count = row["prompt_group_size"] - row["prompt_success_count"]
        candidates.append(
            {
                "prompt_id": row["prompt_id"],
                "uid": row["uid"],
                "prompt": row["prompt"],
                "response_id": row["response_id"],
                "response_text": row["response_text"],
                "reward": row["reward"],
                "success": row["success"],
                "summary_text": row["summary_text"],
                "summary_text_effective": row["summary_text_effective"],
                "summary_has_tag": row["summary_has_tag"],
                "summary_is_fallback": row["summary_is_fallback"],
                "feedback_text": row["feedback_text"],
                "has_feedback": row["has_feedback"],
                "ground_truth": row["ground_truth"],
                "pred": row["pred"],
                "acc": row["acc"],
                "incorrect_format": row["incorrect_format"],
                "difficulty_bucket": row["difficulty_bucket"],
                "prompt_group_size": row["prompt_group_size"],
                "prompt_success_count": row["prompt_success_count"],
                "prompt_failure_count": prompt_failure_count,
                "prompt_has_primary_success": row["prompt_success_count"] >= 1,
                "prompt_has_another_success": row["prompt_success_count"] >= 2,
                "prompt_has_failure_peer": prompt_failure_count >= 1,
                "source_file": row["source_file"],
                "source_line": row["source_line"],
                "rollout_step": row["rollout_step"],
            }
        )
    return candidates


def build_evidence_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence_rows = []
    for row in rows:
        response_role = "success" if row["success"] else "failure"
        evidence_rows.append(
            {
                "evidence_id": f"{row['response_id']}::response_full",
                "prompt_id": row["prompt_id"],
                "uid": row["uid"],
                "source_response_id": row["response_id"],
                "evidence_type": "response_full",
                "role": response_role,
                "text": row["response_text"],
                "summary_text": row["summary_text"],
                "has_summary_tag": row["summary_has_tag"],
                "is_fallback_summary": False,
                "reward": row["reward"],
                "success": row["success"],
                "difficulty_bucket": row["difficulty_bucket"],
                "source_file": row["source_file"],
                "source_line": row["source_line"],
                "rollout_step": row["rollout_step"],
            }
        )
        evidence_rows.append(
            {
                "evidence_id": f"{row['response_id']}::summary",
                "prompt_id": row["prompt_id"],
                "uid": row["uid"],
                "source_response_id": row["response_id"],
                "evidence_type": "summary",
                "role": response_role,
                "text": row["summary_text_effective"],
                "summary_text": row["summary_text"],
                "has_summary_tag": row["summary_has_tag"],
                "is_fallback_summary": row["summary_is_fallback"],
                "reward": row["reward"],
                "success": row["success"],
                "difficulty_bucket": row["difficulty_bucket"],
                "source_file": row["source_file"],
                "source_line": row["source_line"],
                "rollout_step": row["rollout_step"],
            }
        )
        if row["has_feedback"]:
            evidence_rows.append(
                {
                    "evidence_id": f"{row['response_id']}::feedback",
                    "prompt_id": row["prompt_id"],
                    "uid": row["uid"],
                    "source_response_id": row["response_id"],
                    "evidence_type": "environment_feedback",
                    "role": response_role,
                    "text": row["feedback_text"],
                    "summary_text": None,
                    "has_summary_tag": False,
                    "is_fallback_summary": False,
                    "reward": row["reward"],
                    "success": row["success"],
                    "difficulty_bucket": row["difficulty_bucket"],
                    "source_file": row["source_file"],
                    "source_line": row["source_line"],
                    "rollout_step": row["rollout_step"],
                }
            )
    return evidence_rows


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
    paths = resolve_input_paths(args.input)
    if not paths:
        raise SystemExit(f"No input JSONL files found for: {args.input}")

    grouped = read_rollouts(paths, success_threshold=args.success_threshold)
    rows, summary = sample_groups(
        grouped=grouped,
        max_prompts=args.max_prompts,
        max_responses_per_prompt=args.max_responses_per_prompt,
        min_responses_per_prompt=args.min_responses_per_prompt,
        seed=args.seed,
    )

    candidates = build_candidate_rows(rows)
    evidence_items = build_evidence_rows(rows)

    output_dir = Path(args.output_dir)
    candidate_output = output_dir / "candidate_responses.jsonl"
    evidence_output = output_dir / "evidence_items.jsonl"
    summary_output = output_dir / "dataset_summary.json"

    write_jsonl(candidate_output, candidates)
    write_jsonl(evidence_output, evidence_items)
    write_json(
        summary_output,
        {
            "input_paths": [str(p) for p in paths],
            "success_threshold": args.success_threshold,
            "max_prompts": args.max_prompts,
            "max_responses_per_prompt": args.max_responses_per_prompt,
            "min_responses_per_prompt": args.min_responses_per_prompt,
            "seed": args.seed,
            "aggregate": {
                **summary,
                "candidate_count": len(candidates),
                "evidence_count": len(evidence_items),
                "evidence_type_counts": {
                    evidence_type: sum(1 for row in evidence_items if row["evidence_type"] == evidence_type)
                    for evidence_type in sorted({row["evidence_type"] for row in evidence_items})
                },
            },
        },
    )
    print(f"Wrote candidate responses to {candidate_output}")
    print(f"Wrote evidence items to {evidence_output}")
    print(f"Wrote dataset summary to {summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
