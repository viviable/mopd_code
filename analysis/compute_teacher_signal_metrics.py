#!/usr/bin/env python3
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


CONDITION_TO_REQUIRED_SECTIONS = {
    "solution": {"solution"},
    "another_solution": {"another_solution"},
    "solution+another_solution": {"solution", "another_solution"},
    "all_solutions": {"solution", "another_solution"},
    "failure_solution": {"failure_solution"},
    "solution+failure_solution": {"solution", "failure_solution"},
    "solution+another_solution+failure_solution": {
        "solution",
        "another_solution",
        "failure_solution",
    },
    "summary_success_k2": {"summary"},
    "solution+summary_success_k2": {"solution", "summary"},
    "solution+feedback": {"solution", "feedback"},
    "solution+feedback+summary_all_k2": {"solution", "summary"},
    "feedback": {"feedback"},
    "random_summary_control": {"summary"},
    "base": set(),
    "base_raw": set(),
    "base_reprompt": set(),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute offline teacher-signal ranking metrics.")
    parser.add_argument("--input", required=True, help="Teacher-scored JSONL file.")
    parser.add_argument("--output", required=True, help="Aggregate JSON output path.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def rankdata(values: list[float]) -> list[float]:
    sorted_pairs = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(sorted_pairs):
        j = i
        while j + 1 < len(sorted_pairs) and sorted_pairs[j + 1][1] == sorted_pairs[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[sorted_pairs[k][0]] = avg_rank
        i = j + 1
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    mean_x = average(x)
    mean_y = average(y)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    den_x = math.sqrt(sum((a - mean_x) ** 2 for a in x))
    den_y = math.sqrt(sum((b - mean_y) ** 2 for b in y))
    if den_x == 0.0 or den_y == 0.0:
        return 0.0
    return num / (den_x * den_y)


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    return pearson(rankdata(x), rankdata(y))


def kendall_tau(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    concordant = 0
    discordant = 0
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 or dy == 0:
                continue
            if dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    denom = concordant + discordant
    if denom == 0:
        return 0.0
    return (concordant - discordant) / denom


def pairwise_accuracy(scores: list[float], rewards: list[float]) -> float:
    correct = 0
    total = 0
    n = len(scores)
    for i in range(n):
        for j in range(i + 1, n):
            if rewards[i] == rewards[j]:
                continue
            total += 1
            if (scores[i] - scores[j]) * (rewards[i] - rewards[j]) > 0:
                correct += 1
    return correct / total if total else 0.0


def auc_from_scores(scores: list[float], labels: list[int]) -> float:
    positives = [(s, l) for s, l in zip(scores, labels) if l == 1]
    negatives = [(s, l) for s, l in zip(scores, labels) if l == 0]
    if not positives or not negatives:
        return 0.0
    better = 0.0
    total = 0
    for ps, _ in positives:
        for ns, _ in negatives:
            total += 1
            if ps > ns:
                better += 1.0
            elif ps == ns:
                better += 0.5
    return better / total if total else 0.0


def top1_hit_rate(scores: list[float], rewards: list[float], successes: list[int]) -> float:
    if not scores:
        return 0.0
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    max_reward = max(rewards)
    return 1.0 if rewards[best_idx] == max_reward or successes[best_idx] == 1 else 0.0


def point_biserial(scores: list[float], labels: list[int]) -> float:
    if len(scores) != len(labels) or len(scores) < 2:
        return 0.0
    return pearson(scores, [float(label) for label in labels])


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def minmax_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    min_value = min(values)
    max_value = max(values)
    if max_value == min_value:
        return [0.5 for _ in values]
    scale = max_value - min_value
    return [(value - min_value) / scale for value in values]


def brier_score(probabilities: list[float], labels: list[int]) -> float:
    if len(probabilities) != len(labels) or not probabilities:
        return 0.0
    return average(
        [(prob - float(label)) ** 2 for prob, label in zip(probabilities, labels)]
    )


def group_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        condition = row["condition"]
        prompt_id = row["prompt_id"]
        grouped[condition][prompt_id].append(row)
    return grouped


def group_rows_by_target_type(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, list[dict[str, Any]]]]]:
    grouped: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        target_type = row.get("target_type", "unknown")
        condition = row["condition"]
        prompt_id = row["prompt_id"]
        grouped[target_type][condition][prompt_id].append(row)
    return grouped


def compute_condition_metrics(rows_by_prompt: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    per_prompt_metrics = []
    prompt_lengths = []
    trunc_flags = []
    for prompt_id, rows in rows_by_prompt.items():
        if len(rows) < 2:
            continue
        scores = [float(row["teacher_score"]) for row in rows]
        rewards = [float(row["reward"]) for row in rows]
        successes = [1 if row["success"] else 0 for row in rows]
        per_prompt_metrics.append(
            {
                "prompt_id": prompt_id,
                "difficulty_bucket": rows[0].get("difficulty_bucket", "unknown"),
                "spearman": spearman(scores, rewards),
                "kendall_tau": kendall_tau(scores, rewards),
                "pairwise_accuracy": pairwise_accuracy(scores, rewards),
                "success_auc": auc_from_scores(scores, successes),
                "success_point_biserial": point_biserial(scores, successes),
                "success_brier_sigmoid": brier_score([sigmoid(score) for score in scores], successes),
                "success_brier_minmax": brier_score(minmax_normalize(scores), successes),
                "top1_hit_rate": top1_hit_rate(scores, rewards, successes),
            }
        )
        prompt_lengths.extend(
            float(row["teacher_prompt_len"])
            for row in rows
            if row.get("teacher_prompt_len") is not None
        )
        trunc_flags.extend(
            1.0 if row.get("teacher_prompt_truncated") else 0.0
            for row in rows
            if row.get("teacher_prompt_truncated") is not None
        )

    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in per_prompt_metrics:
        by_bucket[item["difficulty_bucket"]].append(item)

    return {
        "count_prompts": len(per_prompt_metrics),
        "mean_spearman": average([x["spearman"] for x in per_prompt_metrics]),
        "mean_kendall_tau": average([x["kendall_tau"] for x in per_prompt_metrics]),
        "pairwise_accuracy": average([x["pairwise_accuracy"] for x in per_prompt_metrics]),
        "success_auc": average([x["success_auc"] for x in per_prompt_metrics]),
        "success_point_biserial": average([x["success_point_biserial"] for x in per_prompt_metrics]),
        "success_brier_sigmoid": average([x["success_brier_sigmoid"] for x in per_prompt_metrics]),
        "success_brier_minmax": average([x["success_brier_minmax"] for x in per_prompt_metrics]),
        "top1_hit_rate": average([x["top1_hit_rate"] for x in per_prompt_metrics]),
        "mean_prompt_length": average(prompt_lengths),
        "truncation_rate": average(trunc_flags),
        "by_bucket": {
            bucket: {
                "count_prompts": len(items),
                "mean_spearman": average([x["spearman"] for x in items]),
                "mean_kendall_tau": average([x["kendall_tau"] for x in items]),
                "pairwise_accuracy": average([x["pairwise_accuracy"] for x in items]),
                "success_auc": average([x["success_auc"] for x in items]),
                "success_point_biserial": average([x["success_point_biserial"] for x in items]),
                "success_brier_sigmoid": average([x["success_brier_sigmoid"] for x in items]),
                "success_brier_minmax": average([x["success_brier_minmax"] for x in items]),
                "top1_hit_rate": average([x["top1_hit_rate"] for x in items]),
            }
            for bucket, items in sorted(by_bucket.items())
        },
    }


def required_sections_for_condition(condition: str) -> set[str]:
    return CONDITION_TO_REQUIRED_SECTIONS.get(condition, set())


def is_effective_row(row: dict[str, Any]) -> bool:
    required_sections = required_sections_for_condition(row["condition"])
    if not required_sections:
        return True
    sections_used = row.get("sections_used", {})
    return all(bool(sections_used.get(section, False)) for section in required_sections)


def validate_rows(rows: list[dict[str, Any]]) -> None:
    required = {"condition", "prompt_id", "target_id", "teacher_score", "reward", "success"}
    for idx, row in enumerate(rows, start=1):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"Row {idx} is missing required fields: {missing}")


def add_effective_flags(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        new_row = dict(row)
        new_row["effective_for_condition"] = is_effective_row(row)
        out.append(new_row)
    return out


def compute_split(grouped_rows: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    return {
        condition: compute_condition_metrics(rows_by_prompt)
        for condition, rows_by_prompt in sorted(grouped_rows.items())
    }


def build_paired_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = group_rows_by_target_type(rows)
    out: dict[str, Any] = {}
    for target_type, by_condition in sorted(grouped.items()):
        prompt_sets = [set(rows_by_prompt) for rows_by_prompt in by_condition.values()]
        if not prompt_sets:
            common_prompts: set[str] = set()
        else:
            common_prompts = set.intersection(*prompt_sets)

        paired_grouped = {
            condition: {
                prompt_id: rows_by_prompt[prompt_id]
                for prompt_id in sorted(common_prompts)
                if prompt_id in rows_by_prompt
            }
            for condition, rows_by_prompt in sorted(by_condition.items())
        }

        out[target_type] = {
            "paired_prompt_count": len(common_prompts),
            "conditions": compute_split(paired_grouped),
        }
    return out


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> int:
    args = parse_args()
    rows = read_jsonl(Path(args.input))
    validate_rows(rows)
    rows = add_effective_flags(rows)

    all_grouped = group_rows(rows)
    effective_rows = [row for row in rows if row["effective_for_condition"]]
    effective_grouped = group_rows(effective_rows)
    reward_zero_rows = [row for row in rows if float(row["reward"]) == 0.0]
    reward_zero_grouped = group_rows(reward_zero_rows)
    reward_zero_effective_rows = [row for row in reward_zero_rows if row["effective_for_condition"]]
    reward_zero_effective_grouped = group_rows(reward_zero_effective_rows)

    all_counts = defaultdict(int)
    effective_counts = defaultdict(int)
    reward_zero_counts = defaultdict(int)
    reward_zero_effective_counts = defaultdict(int)
    for row in rows:
        all_counts[row["condition"]] += 1
        if row["effective_for_condition"]:
            effective_counts[row["condition"]] += 1
        if float(row["reward"]) == 0.0:
            reward_zero_counts[row["condition"]] += 1
            if row["effective_for_condition"]:
                reward_zero_effective_counts[row["condition"]] += 1

    result = {
        "input": args.input,
        "sample_sets": {
            "all_samples": {
                "description": "All scored samples, including rows where the requested context did not activate.",
                "conditions": compute_split(all_grouped),
                "paired_by_target_type": build_paired_subset(rows),
            },
            "effective_only": {
                "description": "Only scored samples where the requested context sections were actually present.",
                "conditions": compute_split(effective_grouped),
                "paired_by_target_type": build_paired_subset(effective_rows),
            },
            "reward_zero_only": {
                "description": "Only scored samples with reward == 0. These metrics are mainly diagnostic because rank-based metrics can degenerate when all retained labels are failures.",
                "conditions": compute_split(reward_zero_grouped),
                "paired_by_target_type": build_paired_subset(reward_zero_rows),
            },
            "reward_zero_effective_only": {
                "description": "Only reward == 0 samples where the requested context sections were actually present. Mainly diagnostic for failure-only slices.",
                "conditions": compute_split(reward_zero_effective_grouped),
                "paired_by_target_type": build_paired_subset(reward_zero_effective_rows),
            },
        },
        "condition_sample_counts": {
            condition: {
                "all_samples": all_counts[condition],
                "effective_only": effective_counts[condition],
                "reward_zero_only": reward_zero_counts[condition],
                "reward_zero_effective_only": reward_zero_effective_counts[condition],
            }
            for condition in sorted(
                set(all_counts)
                | set(effective_counts)
                | set(reward_zero_counts)
                | set(reward_zero_effective_counts)
            )
        },
    }
    write_json(Path(args.output), result)
    print(f"Wrote teacher signal metrics to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
