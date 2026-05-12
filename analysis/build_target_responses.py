#!/usr/bin/env python3
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build target responses for teacher scoring.")
    parser.add_argument("--candidates", required=True, help="Path to candidate_responses.jsonl.")
    parser.add_argument("--output", required=True, help="Output target_responses.jsonl path.")
    parser.add_argument(
        "--strong-success",
        help="Optional JSONL with stronger-model successful responses. Expected fields: prompt_id, response_text.",
    )
    parser.add_argument(
        "--student-success-per-prompt",
        type=int,
        default=1,
        help="Maximum number of student success targets per prompt.",
    )
    parser.add_argument(
        "--student-failure-per-prompt",
        type=int,
        default=1,
        help="Maximum number of student failure targets per prompt.",
    )
    parser.add_argument(
        "--strong-success-per-prompt",
        type=int,
        default=1,
        help="Maximum number of stronger-model success targets per prompt.",
    )
    parser.add_argument(
        "--strong-source-model",
        default="strong_model",
        help="Source model label used when strong-success rows do not provide one.",
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


def stable_target_id(target_type: str, prompt_id: str, source_response_id: str | None, response_text: str) -> str:
    raw = f"{target_type}|{prompt_id}|{source_response_id or 'none'}|{response_text}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def build_student_targets(
    candidates: list[dict[str, Any]],
    student_success_per_prompt: int,
    student_failure_per_prompt: int,
) -> list[dict[str, Any]]:
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_prompt[row["prompt_id"]].append(row)

    targets: list[dict[str, Any]] = []
    for prompt_id, rows in by_prompt.items():
        rows = sorted(rows, key=lambda x: x["response_id"])
        successes = [row for row in rows if row["success"]][:student_success_per_prompt]
        failures = [row for row in rows if not row["success"]][:student_failure_per_prompt]

        for row in successes:
            targets.append(
                {
                    "target_id": stable_target_id("student_success", prompt_id, row["response_id"], row["response_text"]),
                    "target_type": "student_success",
                    "prompt_id": prompt_id,
                    "source_response_id": row["response_id"],
                    "response_text": row["response_text"],
                    "success": True,
                    "reward": row["reward"],
                    "source_model": "student_rollout",
                    "difficulty_bucket": row["difficulty_bucket"],
                    "metadata": {
                        "rollout_step": row.get("rollout_step"),
                        "pred": row.get("pred"),
                        "ground_truth": row.get("ground_truth"),
                    },
                }
            )

        for row in failures:
            targets.append(
                {
                    "target_id": stable_target_id("student_failure", prompt_id, row["response_id"], row["response_text"]),
                    "target_type": "student_failure",
                    "prompt_id": prompt_id,
                    "source_response_id": row["response_id"],
                    "response_text": row["response_text"],
                    "success": False,
                    "reward": row["reward"],
                    "source_model": "student_rollout",
                    "difficulty_bucket": row["difficulty_bucket"],
                    "metadata": {
                        "rollout_step": row.get("rollout_step"),
                        "pred": row.get("pred"),
                        "ground_truth": row.get("ground_truth"),
                    },
                }
            )
    return targets


def build_strong_targets(
    strong_rows: list[dict[str, Any]],
    strong_success_per_prompt: int,
    default_source_model: str,
) -> list[dict[str, Any]]:
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in strong_rows:
        validated_success = row.get("validated_success", row.get("success", True))
        if not validated_success:
            continue
        prompt_id = row["prompt_id"]
        by_prompt[prompt_id].append(row)

    targets: list[dict[str, Any]] = []
    for prompt_id, rows in by_prompt.items():
        for row in rows[:strong_success_per_prompt]:
            response_text = row["response_text"]
            targets.append(
                {
                    "target_id": stable_target_id("strong_success", prompt_id, row.get("source_response_id"), response_text),
                    "target_type": "strong_success",
                    "prompt_id": prompt_id,
                    "source_response_id": row.get("source_response_id"),
                    "response_text": response_text,
                    "success": True,
                    "reward": row.get("reward"),
                    "source_model": row.get("source_model", row.get("generator_model", default_source_model)),
                    "difficulty_bucket": row.get("difficulty_bucket", "unknown"),
                    "metadata": {
                        "generator_model": row.get("generator_model", default_source_model),
                        "judge_info": row.get("judge_info"),
                    },
                }
            )
    return targets


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    candidates = read_jsonl(Path(args.candidates))
    targets = build_student_targets(
        candidates=candidates,
        student_success_per_prompt=args.student_success_per_prompt,
        student_failure_per_prompt=args.student_failure_per_prompt,
    )

    if args.strong_success:
        strong_rows = read_jsonl(Path(args.strong_success))
        targets.extend(
            build_strong_targets(
                strong_rows=strong_rows,
                strong_success_per_prompt=args.strong_success_per_prompt,
                default_source_model=args.strong_source_model,
            )
        )

    write_jsonl(Path(args.output), targets)
    print(f"Wrote target responses to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
