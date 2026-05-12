#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import wandb


DEFAULT_METRIC = "val-core/sciknoweval/acc/mean@8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch a W&B run metric history and draw a line plot."
    )
    parser.add_argument(
        "--run",
        required=True,
        help="W&B run URL or entity/project/run_id path.",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help=f"Metric name to plot. Default: {DEFAULT_METRIC}",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Defaults to analysis_outputs/<run_id>_<sanitized_metric>.png",
    )
    parser.add_argument(
        "--csv-output",
        default=None,
        help="Optional CSV path to save the fetched history.",
    )
    parser.add_argument(
        "--x-axis",
        choices=["step", "history_index"],
        default="step",
        help="Use W&B _step if present, otherwise fall back to sequential history index.",
    )
    return parser.parse_args()


def parse_run_path(run_value: str) -> tuple[str, str, str]:
    if run_value.startswith("http://") or run_value.startswith("https://"):
        parsed = urlparse(run_value)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 4 and parts[2] == "runs":
            return parts[0], parts[1], parts[3]
        raise ValueError(f"Unsupported W&B run URL format: {run_value}")

    parts = [part for part in run_value.split("/") if part]
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    raise ValueError("Run must be a W&B URL or entity/project/run_id.")


def sanitize_metric_name(metric: str) -> str:
    return metric.replace("/", "_").replace("@", "at")


def fetch_history(entity: str, project: str, run_id: str, metric: str) -> list[dict]:
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    rows = []
    for history_index, row in enumerate(run.scan_history(keys=["_step", metric])):
        if metric not in row or row[metric] is None:
            continue
        rows.append(
            {
                "history_index": history_index,
                "_step": row.get("_step"),
                metric: row[metric],
            }
        )
    return rows


def write_csv(rows: list[dict], metric: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["history_index", "_step", metric])
        writer.writeheader()
        writer.writerows(rows)


def plot_history(rows: list[dict], metric: str, run_label: str, x_axis: str, output_path: Path) -> None:
    if not rows:
        raise ValueError(f"No history found for metric: {metric}")

    if x_axis == "step" and any(row["_step"] is not None for row in rows):
        x_values = [row["_step"] if row["_step"] is not None else row["history_index"] for row in rows]
        x_label = "_step"
    else:
        x_values = [row["history_index"] for row in rows]
        x_label = "history_index"

    y_values = [row[metric] for row in rows]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(x_values, y_values, color="#2f6db3", linewidth=2)
    plt.scatter(x_values, y_values, color="#2f6db3", s=14)
    plt.title(f"{metric}\n{run_label}")
    plt.xlabel(x_label)
    plt.ylabel(metric)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> int:
    args = parse_args()
    entity, project, run_id = parse_run_path(args.run)
    rows = fetch_history(entity, project, run_id, args.metric)

    default_output = (
        Path("analysis_outputs")
        / f"{run_id}_{sanitize_metric_name(args.metric)}.png"
    )
    output_path = Path(args.output) if args.output else default_output

    csv_path = None
    if args.csv_output:
        csv_path = Path(args.csv_output)
    else:
        csv_path = output_path.with_suffix(".csv")

    write_csv(rows, args.metric, csv_path)
    plot_history(rows, args.metric, f"{entity}/{project}/{run_id}", args.x_axis, output_path)

    print(f"Wrote plot to {output_path}")
    print(f"Wrote history CSV to {csv_path}")
    print(f"Fetched {len(rows)} points for metric {args.metric}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
