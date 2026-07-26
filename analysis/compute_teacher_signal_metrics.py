#!/usr/bin/env python3
import argparse
import json
import math
import random as _random
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
    "random_peer": {"solution"},
    "verifier_score_ordered": {"solution"},
    "base": set(),
    "base_raw": set(),
    "base_reprompt": set(),
}

# Aggregate metric name -> per-prompt metric key it averages over (used for bootstrap CIs).
AGG_TO_PROMPT_KEY = {
    "mean_spearman": "spearman",
    "mean_kendall_tau": "kendall_tau",
    "pairwise_accuracy": "pairwise_accuracy",
    "success_auc": "success_auc",
    "success_point_biserial": "success_point_biserial",
    "success_brier_sigmoid": "success_brier_sigmoid",
    "success_brier_minmax": "success_brier_minmax",
    "top1_hit_rate": "top1_hit_rate",
}

# Bootstrap settings for prompt-level 95% CIs. Overridable from main() via CLI.
BOOTSTRAP_NUM = 2000
BOOTSTRAP_SEED = 12345
BOOTSTRAP_ALPHA = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute offline teacher-signal ranking metrics.")
    parser.add_argument("--input", required=True, help="Teacher-scored JSONL file.")
    parser.add_argument("--output", required=True, help="Aggregate JSON output path.")
    parser.add_argument(
        "--num-bootstrap",
        type=int,
        default=BOOTSTRAP_NUM,
        help="Number of prompt-level bootstrap resamples for 95%% CIs.",
    )
    parser.add_argument(
        "--table-output",
        help="Optional path for a markdown Table 7 (main metrics + CIs). Defaults to <output stem>_table7.md; a .csv sibling is also written.",
    )
    parser.add_argument(
        "--table-sample-set",
        default="both_present_effective_only",
        help="Which sample set to render into Table 7.",
    )
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


def percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation percentile of an already-sorted list; q in [0, 1]."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def bootstrap_cis(
    per_prompt_metrics: list[dict[str, Any]],
    agg_to_key: dict[str, str],
    num_boot: int = BOOTSTRAP_NUM,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = BOOTSTRAP_ALPHA,
) -> dict[str, list[float]]:
    """Prompt-level bootstrap: resample prompts with replacement, recompute each metric mean.

    Returns {agg_metric_name: [ci_low, ci_high]}. All metrics share the same resampled prompt
    indices per iteration so their CIs are mutually consistent.
    """
    n = len(per_prompt_metrics)
    if n == 0:
        return {agg: [0.0, 0.0] for agg in agg_to_key}
    # Pre-extract per-prompt value arrays once.
    columns = {agg: [float(m[key]) for m in per_prompt_metrics] for agg, key in agg_to_key.items()}
    rng = _random.Random(seed)
    boot_means: dict[str, list[float]] = {agg: [] for agg in agg_to_key}
    for _ in range(num_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        for agg, col in columns.items():
            boot_means[agg].append(sum(col[i] for i in idx) / n)
    lo_q = alpha / 2.0
    hi_q = 1.0 - alpha / 2.0
    out: dict[str, list[float]] = {}
    for agg, means in boot_means.items():
        means.sort()
        out[agg] = [percentile(means, lo_q), percentile(means, hi_q)]
    return out


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

    cis = bootstrap_cis(per_prompt_metrics, AGG_TO_PROMPT_KEY)

    result = {
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
    for agg, ci in cis.items():
        result[f"{agg}_ci"] = ci
    return result


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


TABLE7_METRICS = ["mean_spearman", "mean_kendall_tau", "pairwise_accuracy", "success_auc"]
TABLE7_CONDITION_ORDER = [
    "base",
    "solution",
    "another_solution",
    "failure_solution",
    "solution+failure_solution",
    "all_solutions",
    "random_peer",
    "verifier_score_ordered",
    "solution+another_solution+failure_solution",
]


def _fmt_ci(cond_metrics: dict[str, Any], metric: str) -> str:
    value = cond_metrics.get(metric)
    ci = cond_metrics.get(f"{metric}_ci")
    if value is None:
        return "n/a"
    if not ci:
        return f"{value:.3f}"
    return f"{value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def write_table7(result: dict[str, Any], sample_set: str, md_path: Path, csv_path: Path) -> None:
    """Emit Table 7: main ranking metrics + 95% CIs, one row per condition, for one sample set."""
    split = result["sample_sets"].get(sample_set)
    if split is None:
        print(f"[table7] sample set {sample_set!r} not found; skipping table.")
        return
    conditions_view = split["conditions"]
    ordered = [c for c in TABLE7_CONDITION_ORDER if c in conditions_view]
    ordered += [c for c in sorted(conditions_view) if c not in ordered]

    header = ["condition", "count_prompts", *TABLE7_METRICS]
    md_lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    csv_lines = [",".join(header)]
    for cond in ordered:
        cm = conditions_view[cond]
        n = cm.get("count_prompts", 0)
        md_cells = [cond, str(n)] + [_fmt_ci(cm, m) for m in TABLE7_METRICS]
        md_lines.append("| " + " | ".join(md_cells) + " |")
        csv_cells = [cond, str(n)]
        for m in TABLE7_METRICS:
            value = cm.get(m)
            ci = cm.get(f"{m}_ci") or [None, None]
            csv_cells += [
                "" if value is None else f"{value:.6f}",
                "" if ci[0] is None else f"{ci[0]:.6f}",
                "" if ci[1] is None else f"{ci[1]:.6f}",
            ]
        csv_lines.append(",".join(csv_cells))
    # CSV header needs per-metric ci columns; rebuild to match rows.
    csv_header = ["condition", "count_prompts"]
    for m in TABLE7_METRICS:
        csv_header += [m, f"{m}_ci_low", f"{m}_ci_high"]
    csv_lines[0] = ",".join(csv_header)

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(
        f"# Table 7: teacher-signal ranking metrics ({sample_set})\n\n"
        + "Values are prompt-mean with prompt-level bootstrap 95% CI in brackets.\n\n"
        + "\n".join(md_lines)
        + "\n"
    )
    csv_path.write_text("\n".join(csv_lines) + "\n")
    print(f"Wrote Table 7 to {md_path} and {csv_path}")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> int:
    args = parse_args()
    global BOOTSTRAP_NUM
    BOOTSTRAP_NUM = int(args.num_bootstrap)
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

    # Both-present subset: prompts whose candidate group contains BOTH >=1 success and >=1 failure.
    labels_by_prompt: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        labels_by_prompt[row["prompt_id"]].add(1 if row["success"] else 0)
    both_present_prompts = {p for p, labels in labels_by_prompt.items() if 0 in labels and 1 in labels}
    both_present_rows = [row for row in rows if row["prompt_id"] in both_present_prompts]
    both_present_grouped = group_rows(both_present_rows)
    both_present_effective_rows = [row for row in both_present_rows if row["effective_for_condition"]]
    both_present_effective_grouped = group_rows(both_present_effective_rows)

    all_counts = defaultdict(int)
    effective_counts = defaultdict(int)
    reward_zero_counts = defaultdict(int)
    reward_zero_effective_counts = defaultdict(int)
    both_present_counts = defaultdict(int)
    both_present_effective_counts = defaultdict(int)
    for row in rows:
        all_counts[row["condition"]] += 1
        if row["effective_for_condition"]:
            effective_counts[row["condition"]] += 1
        if float(row["reward"]) == 0.0:
            reward_zero_counts[row["condition"]] += 1
            if row["effective_for_condition"]:
                reward_zero_effective_counts[row["condition"]] += 1
        if row["prompt_id"] in both_present_prompts:
            both_present_counts[row["condition"]] += 1
            if row["effective_for_condition"]:
                both_present_effective_counts[row["condition"]] += 1

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
            "both_present_all": {
                "description": "Only prompts whose candidate group has BOTH >=1 success and >=1 failure. AUC / pairwise accuracy are only well-defined here.",
                "conditions": compute_split(both_present_grouped),
                "paired_by_target_type": build_paired_subset(both_present_rows),
            },
            "both_present_effective_only": {
                "description": "Both-present prompts, restricted to samples where the requested context sections were actually present. Primary subset for the 2S1F-vs-controls claim.",
                "conditions": compute_split(both_present_effective_grouped),
                "paired_by_target_type": build_paired_subset(both_present_effective_rows),
            },
        },
        "condition_sample_counts": {
            condition: {
                "all_samples": all_counts[condition],
                "effective_only": effective_counts[condition],
                "reward_zero_only": reward_zero_counts[condition],
                "reward_zero_effective_only": reward_zero_effective_counts[condition],
                "both_present_all": both_present_counts[condition],
                "both_present_effective_only": both_present_effective_counts[condition],
            }
            for condition in sorted(
                set(all_counts)
                | set(effective_counts)
                | set(reward_zero_counts)
                | set(reward_zero_effective_counts)
            )
        },
        "both_present_prompt_count": len(both_present_prompts),
    }
    write_json(Path(args.output), result)
    print(f"Wrote teacher signal metrics to {args.output}")

    out_path = Path(args.output)
    if args.table_output:
        md_path = Path(args.table_output)
    else:
        md_path = out_path.with_name(f"{out_path.stem}_table7.md")
    csv_path = md_path.with_suffix(".csv")
    write_table7(result, args.table_sample_set, md_path, csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
