#!/usr/bin/env python3
import argparse
import ast
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from tqdm import tqdm


CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\s]")


@dataclass
class ResponseRecord:
    prompt_id: str
    response_id: str
    response_text: str
    success: bool
    reward: float
    difficulty_bucket: str | None
    source_file: str | None
    source_line: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute rollout diversity metrics from candidate_responses.jsonl.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Method input in the form NAME=/path/to/candidate_responses.jsonl. Repeat for multiple methods.",
    )
    parser.add_argument("--task", choices=["generic", "code"], default="generic")
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--max-responses-per-prompt", type=int, default=None)
    parser.add_argument("--min-responses-per-prompt", type=int, default=2)
    parser.add_argument("--intersect-prompts", action="store_true")
    parser.add_argument("--case-study-top-n", type=int, default=20)
    return parser.parse_args()


def parse_run_arg(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid --run value: {spec}")
    name, path = spec.split("=", 1)
    if not name:
        raise ValueError(f"Missing run name in --run value: {spec}")
    return name, Path(path)


def stable_prompt_id(prompt_text: str) -> str:
    return hashlib.sha1(prompt_text.encode("utf-8")).hexdigest()[:16]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_code(text: str) -> str:
    matches = CODE_BLOCK_RE.findall(text)
    if matches:
        return matches[-1].strip()
    return text.strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu_score(candidate: list[str], references: list[list[str]], max_n: int = 4) -> float:
    if not candidate:
        return 0.0
    precisions: list[float] = []
    for n in range(1, max_n + 1):
        cand_ngrams = ngrams(candidate, n)
        total = sum(cand_ngrams.values())
        if total == 0:
            precisions.append(1.0)
            continue
        ref_max: Counter[tuple[str, ...]] = Counter()
        for ref in references:
            ref_counts = ngrams(ref, n)
            for gram, count in ref_counts.items():
                if count > ref_max[gram]:
                    ref_max[gram] = count
        overlap = sum(min(count, ref_max[gram]) for gram, count in cand_ngrams.items())
        precisions.append((overlap + 1.0) / (total + 1.0))

    cand_len = len(candidate)
    ref_lens = [len(ref) for ref in references if ref]
    if not ref_lens:
        return 0.0
    closest_ref_len = min(ref_lens, key=lambda x: (abs(x - cand_len), x))
    brevity_penalty = 1.0 if cand_len > closest_ref_len else math.exp(1.0 - (closest_ref_len / max(cand_len, 1)))
    return brevity_penalty * math.exp(sum(math.log(p) for p in precisions) / max_n)


def self_bleu(texts: list[str]) -> float | None:
    if len(texts) < 2:
        return None
    tokenized = [tokenize(text) for text in texts]
    scores = []
    for idx, cand in enumerate(tokenized):
        refs = [ref for j, ref in enumerate(tokenized) if j != idx]
        scores.append(bleu_score(cand, refs))
    return sum(scores) / len(scores)


def distinct_n(texts: list[str], n: int) -> float | None:
    if not texts:
        return None
    all_tokens = [token for text in texts for token in tokenize(text)]
    grams = ngrams(all_tokens, n)
    total = sum(grams.values())
    if total == 0:
        return None
    return len(grams) / total


def pairwise_mean_distance(texts: list[str]) -> float | None:
    if len(texts) < 2:
        return None
    distances = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            ratio = SequenceMatcher(None, texts[i], texts[j]).ratio()
            distances.append(1.0 - ratio)
    return sum(distances) / len(distances) if distances else None


def ast_node_counter(code: str) -> Counter[str] | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    counts = Counter(type(node).__name__ for node in ast.walk(tree))
    return counts if counts else None


def ast_jaccard_distance(texts: list[str]) -> float | None:
    if len(texts) < 2:
        return None
    counters = [ast_node_counter(text) for text in texts]
    counters = [c for c in counters if c is not None]
    if len(counters) < 2:
        return None
    distances = []
    for i in range(len(counters)):
        for j in range(i + 1, len(counters)):
            keys = set(counters[i]) | set(counters[j])
            intersection = sum(min(counters[i].get(k, 0), counters[j].get(k, 0)) for k in keys)
            union = sum(max(counters[i].get(k, 0), counters[j].get(k, 0)) for k in keys)
            if union == 0:
                continue
            distances.append(1.0 - (intersection / union))
    return sum(distances) / len(distances) if distances else None


def unique_ratio(texts: list[str]) -> float | None:
    if not texts:
        return None
    return len({normalize_whitespace(text) for text in texts}) / len(texts)


def pass_at_k(total: int, successes: int, k: int) -> float | None:
    if total <= 0 or k <= 0:
        return None
    kk = min(k, total)
    if successes <= 0:
        return 0.0
    if kk > total - successes:
        return 1.0
    numerator = math.comb(total - successes, kk)
    denominator = math.comb(total, kk)
    return 1.0 - (numerator / denominator)


def maybe_mean(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def maybe_std(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return None
    return statistics.pstdev(clean)


def maybe_corr(xs: list[float | None], ys: list[float | None]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    xvals = [x for x, _ in pairs]
    yvals = [y for _, y in pairs]
    mean_x = sum(xvals) / len(xvals)
    mean_y = sum(yvals) / len(yvals)
    num = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xvals))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in yvals))
    if den_x == 0.0 or den_y == 0.0:
        return None
    return num / (den_x * den_y)


def prepare_records(rows: list[dict[str, Any]]) -> dict[str, list[ResponseRecord]]:
    grouped: dict[str, list[ResponseRecord]] = defaultdict(list)
    prompt_counts: dict[str, int] = defaultdict(int)
    for line_idx, row in enumerate(rows):
        if "prompt_id" in row:
            prompt_id = row["prompt_id"]
            response_id = row["response_id"]
            response_text = str(row.get("response_text", ""))
            success = bool(row.get("success", False))
            reward = float(row.get("reward", row.get("score", 0.0)) or 0.0)
            difficulty_bucket = row.get("difficulty_bucket")
            source_file = row.get("source_file")
            source_line = row.get("source_line")
        else:
            prompt_text = str(row.get("input", ""))
            prompt_id = stable_prompt_id(prompt_text)
            response_idx = prompt_counts[prompt_id]
            prompt_counts[prompt_id] += 1
            response_id = f"{prompt_id}_r{response_idx}"
            response_text = str(row.get("output", ""))
            reward = float(row.get("reward", row.get("score", 0.0)) or 0.0)
            success = reward >= 1.0
            difficulty_bucket = None
            source_file = None
            source_line = line_idx + 1
        grouped[prompt_id].append(
            ResponseRecord(
                prompt_id=prompt_id,
                response_id=response_id,
                response_text=response_text,
                success=success,
                reward=reward,
                difficulty_bucket=difficulty_bucket,
                source_file=source_file,
                source_line=source_line,
            )
        )
    return grouped


def truncate_groups(
    grouped: dict[str, list[ResponseRecord]],
    max_prompts: int | None,
    max_responses_per_prompt: int | None,
    min_responses_per_prompt: int,
) -> dict[str, list[ResponseRecord]]:
    prompt_ids = sorted(grouped)
    if max_prompts is not None:
        prompt_ids = prompt_ids[:max_prompts]
    out: dict[str, list[ResponseRecord]] = {}
    for prompt_id in prompt_ids:
        rows = sorted(grouped[prompt_id], key=lambda row: row.response_id)
        if max_responses_per_prompt is not None:
            rows = rows[:max_responses_per_prompt]
        if len(rows) >= min_responses_per_prompt:
            out[prompt_id] = rows
    return out


def response_view(rows: list[ResponseRecord], task: str) -> list[str]:
    if task == "code":
        return [extract_code(row.response_text) for row in rows]
    return [row.response_text for row in rows]


def build_prompt_metrics(method: str, prompt_id: str, rows: list[ResponseRecord], task: str, ks: list[int]) -> dict[str, Any]:
    views = response_view(rows, task)
    successes = [row for row in rows if row.success]
    success_views = response_view(successes, task)
    success_count = sum(1 for row in rows if row.success)
    result: dict[str, Any] = {
        "method": method,
        "prompt_id": prompt_id,
        "n_rollouts": len(rows),
        "success_count": success_count,
        "success_rate": success_count / len(rows),
        "mean_reward": sum(row.reward for row in rows) / len(rows),
        "difficulty_bucket": rows[0].difficulty_bucket,
        "self_bleu4": self_bleu(views),
        "pairwise_text_distance": pairwise_mean_distance(views),
        "distinct1": distinct_n(views, 1),
        "distinct2": distinct_n(views, 2),
        "unique_response_ratio": unique_ratio(views),
        "success_distinct2": distinct_n(success_views, 2),
        "success_self_bleu4": self_bleu(success_views),
        "success_pairwise_text_distance": pairwise_mean_distance(success_views),
        "success_unique_response_ratio": unique_ratio(success_views),
    }
    if task == "code":
        result["unique_code_ratio"] = unique_ratio(views)
        result["ast_node_jaccard_distance"] = ast_jaccard_distance(views)
        result["success_ast_node_jaccard_distance"] = ast_jaccard_distance(success_views)
    for k in ks:
        result[f"pass@{k}"] = pass_at_k(len(rows), success_count, k)
    return result


def build_summary(per_prompt_rows: list[dict[str, Any]], ks: list[int], task: str) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_prompt_rows:
        by_method[row["method"]].append(row)

    summary_rows: list[dict[str, Any]] = []
    for method, rows in sorted(by_method.items()):
        summary: dict[str, Any] = {
            "method": method,
            "prompt_count": len(rows),
        }
        metric_names = [
            "success_rate",
            "mean_reward",
            "self_bleu4",
            "pairwise_text_distance",
            "distinct1",
            "distinct2",
            "unique_response_ratio",
            "success_distinct2",
            "success_self_bleu4",
            "success_pairwise_text_distance",
            "success_unique_response_ratio",
        ]
        if task == "code":
            metric_names.extend(
                [
                    "unique_code_ratio",
                    "ast_node_jaccard_distance",
                    "success_ast_node_jaccard_distance",
                ]
            )
        metric_names.extend([f"pass@{k}" for k in ks])

        for metric in metric_names:
            values = [row.get(metric) for row in rows]
            summary[f"{metric}_mean"] = maybe_mean(values)
            summary[f"{metric}_std"] = maybe_std(values)

        if ks:
            scatter_metric = f"pass@{max(ks)}"
            summary[f"corr_pairwise_text_distance_vs_{scatter_metric}"] = maybe_corr(
                [row.get("pairwise_text_distance") for row in rows],
                [row.get(scatter_metric) for row in rows],
            )
        summary_rows.append(summary)
    return summary_rows


def build_case_studies(per_prompt_rows: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_prompt_rows:
        by_prompt[row["prompt_id"]].append(row)

    candidates: list[dict[str, Any]] = []
    for prompt_id, rows in by_prompt.items():
        if len(rows) < 2:
            continue
        diversities = [row.get("pairwise_text_distance") for row in rows if row.get("pairwise_text_distance") is not None]
        if len(diversities) < 2:
            continue
        ordered = sorted(rows, key=lambda row: row.get("pairwise_text_distance") or -1.0, reverse=True)
        high = ordered[0]
        low = ordered[-1]
        candidates.append(
            {
                "prompt_id": prompt_id,
                "high_method": high["method"],
                "high_diversity": high.get("pairwise_text_distance"),
                "high_pass": max(v for k, v in high.items() if k.startswith("pass@") and v is not None),
                "low_method": low["method"],
                "low_diversity": low.get("pairwise_text_distance"),
                "low_pass": max(v for k, v in low.items() if k.startswith("pass@") and v is not None),
                "diversity_gap": (high.get("pairwise_text_distance") or 0.0) - (low.get("pairwise_text_distance") or 0.0),
            }
        )
    candidates.sort(key=lambda row: row["diversity_gap"], reverse=True)
    return candidates[:top_n]


def write_json(path: Path, payload: Any) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = [parse_run_arg(spec) for spec in args.run]
    grouped_by_method: dict[str, dict[str, list[ResponseRecord]]] = {}
    for method, path in tqdm(runs, desc="loading runs", unit="run"):
        rows = read_jsonl(path)
        grouped = prepare_records(rows)
        grouped = truncate_groups(
            grouped,
            max_prompts=args.max_prompts,
            max_responses_per_prompt=args.max_responses_per_prompt,
            min_responses_per_prompt=args.min_responses_per_prompt,
        )
        grouped_by_method[method] = grouped

    if args.intersect_prompts:
        shared_prompt_ids = set.intersection(*(set(grouped) for grouped in grouped_by_method.values()))
        grouped_by_method = {
            method: {prompt_id: grouped[prompt_id] for prompt_id in sorted(shared_prompt_ids)}
            for method, grouped in grouped_by_method.items()
        }

    per_prompt_rows: list[dict[str, Any]] = []
    for method, grouped in sorted(grouped_by_method.items()):
        for prompt_id, rows in tqdm(sorted(grouped.items()), desc=method, unit="prompt"):
            per_prompt_rows.append(build_prompt_metrics(method, prompt_id, rows, args.task, args.ks))

    summary_rows = build_summary(per_prompt_rows, args.ks, args.task)
    case_studies = build_case_studies(per_prompt_rows, args.case_study_top_n)

    payload = {
        "config": {
            "task": args.task,
            "ks": args.ks,
            "intersect_prompts": args.intersect_prompts,
            "max_prompts": args.max_prompts,
            "max_responses_per_prompt": args.max_responses_per_prompt,
            "min_responses_per_prompt": args.min_responses_per_prompt,
            "runs": [{method: str(path)} for method, path in runs],
        },
        "summary": summary_rows,
    }
    write_json(output_dir / "summary.json", payload)
    write_csv(output_dir / "summary.csv", summary_rows)
    write_jsonl(output_dir / "per_prompt.jsonl", per_prompt_rows)
    write_jsonl(output_dir / "case_studies.jsonl", case_studies)
    print(f"Wrote summary to {output_dir / 'summary.json'}")
    print(f"Wrote per-prompt metrics to {output_dir / 'per_prompt.jsonl'}")
    print(f"Wrote case studies to {output_dir / 'case_studies.jsonl'}")


if __name__ == "__main__":
    main()
