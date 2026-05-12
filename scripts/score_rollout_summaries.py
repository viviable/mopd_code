#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - runtime dependency check
    OpenAI = None


SYSTEM_PROMPT = """You are a strict evaluator for rollout summaries.

Your job is to judge whether a summary is a faithful, concise, step-ordered compression of the original solution.
Score conservatively. Penalize vague summaries, summaries that omit key reasoning steps, summaries that add unsupported claims,
and summaries that fail the requested format.

Return only valid JSON.
"""


USER_PROMPT_TEMPLATE = """Evaluate the summary against the original prompt and full model output.

Scoring rubric:
- format_compliance: 0-2
  2 = fully follows the requested summary format
  1 = partially follows the format but has minor issues
  0 = missing or badly violates the format
- correctness: 0-3
  3 = fully faithful and correct relative to the full output
  2 = mostly correct with minor imprecision
  1 = partially correct but contains a notable mistake or distortion
  0 = mostly incorrect, misleading, or unsupported
- critical_step_coverage: 0-3
  3 = preserves the key reasoning steps needed for correctness
  2 = covers most key steps but misses one important step
  1 = only captures superficial or incomplete steps
  0 = fails to capture the crucial reasoning
- conciseness_and_focus: 0-2
  2 = concise and focused on necessary reasoning only
  1 = somewhat verbose or includes mild fluff/result-only phrasing
  0 = mostly vague, padded, or unfocused

Extra boolean judgments:
- has_key_steps: true if it contains actual reasoning steps rather than generic high-level summary
- introduces_hallucination: true if it adds claims or reasoning not supported by the full output
- missing_essential_step: true if an essential reasoning step is omitted

Summary format target:
{summary_style_description}

Instructions:
- Judge relative to the full output, not whether the final answer itself is globally correct on the task.
- Use the prompt only for context.
- Be strict about "key steps" versus vague summary.
- Keep short_rationale under 40 words.

Return JSON with exactly these fields:
{{
  "format_compliance": <int>,
  "correctness": <int>,
  "critical_step_coverage": <int>,
  "conciseness_and_focus": <int>,
  "has_key_steps": <bool>,
  "introduces_hallucination": <bool>,
  "missing_essential_step": <bool>,
  "short_rationale": <string>
}}

Prompt:
{prompt_text}

Full output:
{full_output}

Extracted summary:
{summary_text}
"""


SUMMARY_STYLE_DESCRIPTIONS = {
    "summary": "A brief summary inside <summary> tags.",
    "key_step": "Inside <summary> tags, provide 2-4 short ordered key steps that preserve only the reasoning necessary for correctness.",
}


SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL | re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score rollout summaries with the OpenAI API.")
    parser.add_argument("--input", required=True, help="Input JSONL file, directory, or glob pattern.")
    parser.add_argument("--output", required=True, help="Output JSONL path for per-example scores.")
    parser.add_argument(
        "--aggregate-output",
        help="Optional aggregate JSON output path. Defaults to <output>.summary.json.",
    )
    parser.add_argument("--model", required=True, help="OpenAI model name used for judging.")
    parser.add_argument(
        "--summary-style",
        choices=sorted(SUMMARY_STYLE_DESCRIPTIONS),
        default="summary",
        help="Expected summary style when judging format compliance.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable that stores the OpenAI API key.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional OpenAI-compatible base URL.",
    )
    parser.add_argument("--max-examples", type=int, default=None, help="Optional cap on scored examples.")
    parser.add_argument("--max-retries", type=int, default=5, help="Retries for transient API failures.")
    parser.add_argument("--sleep-seconds", type=float, default=1.0, help="Base backoff sleep in seconds.")
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


def load_examples(paths: list[Path], max_examples: int | None) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for path in paths:
        with path.open() as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record["_source_file"] = str(path)
                record["_source_line"] = line_no
                examples.append(record)
                if max_examples is not None and len(examples) >= max_examples:
                    return examples
    return examples


def make_client(api_key: str, base_url: str | None) -> Any:
    if OpenAI is None:
        raise RuntimeError(
            "The 'openai' package is not installed. Install it with `python3 -m pip install openai`."
        )
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def judge_one(
    client: Any,
    model: str,
    prompt_text: str,
    full_output: str,
    summary_text: str,
    summary_style: str,
    max_retries: int,
    sleep_seconds: float,
) -> dict[str, Any]:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        summary_style_description=SUMMARY_STYLE_DESCRIPTIONS[summary_style],
        prompt_text=prompt_text,
        full_output=full_output,
        summary_text=summary_text,
    )

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response content from judge model.")
            parsed = json.loads(content)
            return normalize_judgment(parsed)
        except Exception as exc:  # pragma: no cover - depends on API/runtime
            last_error = exc
            if attempt + 1 == max_retries:
                break
            time.sleep(sleep_seconds * (2**attempt))
    raise RuntimeError(f"Judge request failed after {max_retries} attempts: {last_error}")


def clamp_int(value: Any, minimum: int, maximum: int) -> int:
    try:
        value = int(value)
    except Exception:
        value = minimum
    return max(minimum, min(maximum, value))


def normalize_judgment(data: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "format_compliance": clamp_int(data.get("format_compliance"), 0, 2),
        "correctness": clamp_int(data.get("correctness"), 0, 3),
        "critical_step_coverage": clamp_int(data.get("critical_step_coverage"), 0, 3),
        "conciseness_and_focus": clamp_int(data.get("conciseness_and_focus"), 0, 2),
        "has_key_steps": bool(data.get("has_key_steps")),
        "introduces_hallucination": bool(data.get("introduces_hallucination")),
        "missing_essential_step": bool(data.get("missing_essential_step")),
        "short_rationale": str(data.get("short_rationale", "")).strip(),
    }
    normalized["total_score"] = (
        normalized["format_compliance"]
        + normalized["correctness"]
        + normalized["critical_step_coverage"]
        + normalized["conciseness_and_focus"]
    )
    return normalized


def local_missing_summary_judgment() -> dict[str, Any]:
    return {
        "format_compliance": 0,
        "correctness": 0,
        "critical_step_coverage": 0,
        "conciseness_and_focus": 0,
        "has_key_steps": False,
        "introduces_hallucination": False,
        "missing_essential_step": True,
        "short_rationale": "Missing <summary> tag or empty summary.",
        "total_score": 0,
    }


def build_result(record: dict[str, Any], judgment: dict[str, Any], summary_text: str | None) -> dict[str, Any]:
    return {
        "source_file": record["_source_file"],
        "source_line": record["_source_line"],
        "step": record.get("step"),
        "reward_score": record.get("score"),
        "input": record.get("input"),
        "output": record.get("output"),
        "summary": summary_text,
        "judgment": judgment,
    }


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "count": 0,
            "mean_total_score": 0.0,
            "mean_format_compliance": 0.0,
            "mean_correctness": 0.0,
            "mean_critical_step_coverage": 0.0,
            "mean_conciseness_and_focus": 0.0,
            "has_key_steps_rate": 0.0,
            "introduces_hallucination_rate": 0.0,
            "missing_essential_step_rate": 0.0,
            "missing_summary_count": 0,
        }

    def mean(field: str) -> float:
        return sum(item["judgment"][field] for item in results) / len(results)

    missing_summary_count = sum(1 for item in results if not item["summary"])
    return {
        "count": len(results),
        "mean_total_score": mean("total_score"),
        "mean_format_compliance": mean("format_compliance"),
        "mean_correctness": mean("correctness"),
        "mean_critical_step_coverage": mean("critical_step_coverage"),
        "mean_conciseness_and_focus": mean("conciseness_and_focus"),
        "has_key_steps_rate": sum(item["judgment"]["has_key_steps"] for item in results) / len(results),
        "introduces_hallucination_rate": (
            sum(item["judgment"]["introduces_hallucination"] for item in results) / len(results)
        ),
        "missing_essential_step_rate": (
            sum(item["judgment"]["missing_essential_step"] for item in results) / len(results)
        ),
        "missing_summary_count": missing_summary_count,
    }


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
        print(f"No input JSONL files found for: {args.input}", file=sys.stderr)
        return 1

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"Missing API key in environment variable: {args.api_key_env}", file=sys.stderr)
        return 1

    examples = load_examples(paths, args.max_examples)
    client = make_client(api_key=api_key, base_url=args.base_url)

    results: list[dict[str, Any]] = []
    for idx, record in enumerate(examples, start=1):
        prompt_text = record.get("input", "")
        full_output = record.get("output", "")
        summary_text = extract_summary(full_output) if isinstance(full_output, str) else None

        if not isinstance(prompt_text, str):
            prompt_text = json.dumps(prompt_text, ensure_ascii=False)
        if not isinstance(full_output, str):
            full_output = json.dumps(full_output, ensure_ascii=False)

        if not summary_text:
            judgment = local_missing_summary_judgment()
        else:
            judgment = judge_one(
                client=client,
                model=args.model,
                prompt_text=prompt_text,
                full_output=full_output,
                summary_text=summary_text,
                summary_style=args.summary_style,
                max_retries=args.max_retries,
                sleep_seconds=args.sleep_seconds,
            )

        results.append(build_result(record, judgment, summary_text))
        if idx % 10 == 0:
            print(f"Scored {idx}/{len(examples)} examples", file=sys.stderr)

    output_path = Path(args.output)
    aggregate_output = Path(args.aggregate_output) if args.aggregate_output else output_path.with_suffix(".summary.json")

    write_jsonl(output_path, results)
    write_json(
        aggregate_output,
        {
            "input_paths": [str(p) for p in paths],
            "model": args.model,
            "summary_style": args.summary_style,
            "aggregate": aggregate_results(results),
        },
    )

    print(f"Wrote per-example scores to {output_path}")
    print(f"Wrote aggregate summary to {aggregate_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
