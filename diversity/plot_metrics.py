#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


METHOD_LABELS = {
    "MOPD_V1": "MOPD",
    "MOPD-V1": "MOPD",
    "MOPD_V2": "base",
    "MOPD-V2": "base",
    "BASE": "base",
}


def display_method_name(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot diversity analysis outputs.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--per-prompt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scatter-k", type=int, default=8)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(f"matplotlib is required for plotting: {exc}") from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = read_json(Path(args.summary))
    summary_rows = summary_payload["summary"]
    per_prompt_rows = read_jsonl(Path(args.per_prompt))

    methods = [display_method_name(row["method"]) for row in summary_rows]
    ks = summary_payload["config"]["ks"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for row in summary_rows:
        xs = ks
        ys = [row.get(f"pass@{k}_mean") for k in ks]
        ax.plot(xs, ys, marker="o", label=display_method_name(row["method"]))
    ax.set_xlabel("k")
    ax.set_ylabel("pass@k")
    ax.set_title("pass@k Curve")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "passk_curve.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bar_keys = [
        ("success_distinct2_mean", "Success Distinct-2"),
        ("success_ast_node_jaccard_distance_mean", "Success AST Jaccard"),
        ("unique_code_ratio_mean", "Unique Code Ratio"),
    ]
    x = list(range(len(bar_keys)))
    width = 0.8 / max(1, len(summary_rows))
    for idx, row in enumerate(summary_rows):
        xs = [v - 0.4 + width / 2 + idx * width for v in x]
        ys = [0.0 if row.get(key) is None else row.get(key) for key, _ in bar_keys]
        ax.bar(xs, ys, width=width, label=display_method_name(row["method"]))
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in bar_keys], rotation=15, ha="right")
    ax.set_title("Method-Level Diversity Metrics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "method_diversity_bars.png", dpi=200)
    plt.close(fig)

    scatter_key = f"pass@{args.scatter_k}"
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_prompt_rows:
        by_method[display_method_name(row["method"])].append(row)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, rows in sorted(by_method.items()):
        filtered = [
            (row.get("pairwise_text_distance"), row.get(scatter_key))
            for row in rows
            if row.get("pairwise_text_distance") is not None and row.get(scatter_key) is not None
        ]
        if filtered:
            xs, ys = zip(*filtered)
            ax.scatter(xs, ys, s=12, alpha=0.55, label=method)
    ax.set_xlabel("Pairwise Text Distance")
    ax.set_ylabel(scatter_key)
    ax.set_title(f"Diversity vs {scatter_key}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "diversity_vs_passk.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, rows in sorted(by_method.items()):
        filtered = [
            (row.get("success_pairwise_text_distance"), row.get(scatter_key))
            for row in rows
            if row.get("success_pairwise_text_distance") is not None and row.get(scatter_key) is not None
        ]
        if filtered:
            xs, ys = zip(*filtered)
            ax.scatter(xs, ys, s=12, alpha=0.55, label=method)
    ax.set_xlabel("Success Pairwise Text Distance")
    ax.set_ylabel(scatter_key)
    ax.set_title(f"Success Diversity vs {scatter_key}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "success_diversity_vs_passk.png", dpi=200)
    plt.close(fig)

    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
