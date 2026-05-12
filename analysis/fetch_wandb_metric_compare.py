#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch one metric for two W&B runs, save per-run JSON, and draw a comparison plot."
    )
    parser.add_argument("--entity", default="safety")
    parser.add_argument("--project", default="sdpo_base")
    parser.add_argument("--run-name", action="append", help="W&B display_name. Repeat twice.")
    parser.add_argument(
        "--run-path",
        action="append",
        help="Explicit W&B run path entity/project/run_id. Repeat twice. If set, overrides --run-name lookup.",
    )
    parser.add_argument("--metric", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--x-axis", choices=["step", "history_index"], default="step")
    parser.add_argument("--label", action="append", help="Display label for each run, in the same order as --run-name.")
    parser.add_argument("--max-step", type=int, default=None, help="If set, keep only points with _step <= this value.")
    parser.add_argument("--x-label", default=None, help="Override x-axis label.")
    parser.add_argument("--y-label", default=None, help="Override y-axis label.")
    parser.add_argument("--title", default=None, help="Plot title. Use empty string to disable.")
    parser.add_argument("--font-size", type=int, default=16, help="Base font size for axis labels and ticks.")
    parser.add_argument("--legend-font-size", type=int, default=14, help="Legend font size.")
    parser.add_argument("--legend-loc", default="best", help="Matplotlib legend location, e.g. lower right.")
    parser.add_argument(
        "--color",
        action="append",
        help="Line color for each run, in the same order as --run-name/--run-path.",
    )
    return parser.parse_args()


def sanitize(text: str) -> str:
    return (
        text.replace("/", "_")
        .replace("@", "at")
        .replace(" ", "_")
        .replace(":", "_")
    )


def ensure_wandb_login() -> None:
    if os.environ.get("WANDB_API_KEY"):
        return
    token_path = Path(".wandb.txt")
    if token_path.exists():
        token = token_path.read_text(encoding="utf-8").strip()
        if token:
            os.environ["WANDB_API_KEY"] = token


def resolve_run(api: wandb.Api, entity: str, project: str, run_name: str):
    runs = list(api.runs(f"{entity}/{project}", filters={"display_name": run_name}))
    if not runs:
        raise ValueError(f"No W&B run found for display_name={run_name!r} in {entity}/{project}")
    if len(runs) > 1:
        raise ValueError(f"Multiple W&B runs found for display_name={run_name!r} in {entity}/{project}")
    return runs[0]


def resolve_run_path(api: wandb.Api, run_path: str):
    return api.run(run_path)


def fetch_metric_history(run, metric: str) -> list[dict]:
    rows = []
    for history_index, row in enumerate(run.scan_history(keys=["_step", metric])):
        value = row.get(metric)
        if value is None:
            continue
        rows.append(
            {
                "history_index": history_index,
                "_step": row.get("_step"),
                metric: value,
            }
        )
    return rows


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def plot_runs(
    run_payloads: list[dict],
    metric: str,
    x_axis: str,
    output_path: Path,
    x_label: str | None,
    y_label: str | None,
    title: str | None,
    font_size: int,
    legend_font_size: int,
    legend_loc: str,
    colors: list[str] | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5.5))
    palette = colors or ["#2f6db3", "#d95f02", "#1b9e77", "#7570b3"]

    xlabel = "history_index"
    for idx, payload in enumerate(run_payloads):
        rows = payload["history"]
        if x_axis == "step" and any(row["_step"] is not None for row in rows):
            xs = [row["_step"] if row["_step"] is not None else row["history_index"] for row in rows]
            xlabel = "step"
        else:
            xs = [row["history_index"] for row in rows]
            xlabel = "history_index"
        ys = [row[metric] for row in rows]
        plt.plot(xs, ys, linewidth=2, color=palette[idx % len(palette)], label=payload["plot_label"])

    if title:
        plt.title(title, fontsize=font_size + 1)
    plt.xlabel(x_label or xlabel, fontsize=font_size)
    plt.ylabel(y_label or metric, fontsize=font_size)
    plt.xticks(fontsize=font_size - 1)
    plt.yticks(fontsize=font_size - 1)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(fontsize=legend_font_size, loc=legend_loc)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> int:
    args = parse_args()
    if args.run_path is not None:
        if len(args.run_path) != 2:
            raise ValueError("Please pass exactly two --run-path values.")
    elif len(args.run_name) != 2:
        raise ValueError("Please pass exactly two --run-name values.")
    expected_runs = len(args.run_path) if args.run_path is not None else len(args.run_name)
    if args.label is not None and len(args.label) != expected_runs:
        raise ValueError("If provided, --label must be passed once per --run-name.")
    if args.color is not None and len(args.color) != expected_runs:
        raise ValueError("If provided, --color must be passed once per run.")

    ensure_wandb_login()
    api = wandb.Api()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payloads = []
    run_refs = args.run_path if args.run_path is not None else args.run_name
    for idx, run_ref in enumerate(run_refs):
        run = (
            resolve_run_path(api, run_ref)
            if args.run_path is not None
            else resolve_run(api, args.entity, args.project, run_ref)
        )
        history = fetch_metric_history(run, args.metric)
        if args.max_step is not None:
            history = [row for row in history if row.get("_step") is not None and row["_step"] <= args.max_step]
        payload = {
            "entity": args.entity,
            "project": args.project,
            "run_id": run.id,
            "run_name": run.display_name or run.name,
            "plot_label": args.label[idx] if args.label is not None else (run.display_name or run.name),
            "metric": args.metric,
            "state": run.state,
            "history": history,
        }
        json_path = output_dir / f"{sanitize(run.display_name or run.name)}__{sanitize(args.metric)}.json"
        write_json(json_path, payload)
        payload["json_path"] = str(json_path)
        payloads.append(payload)

    plot_path = output_dir / f"compare__{sanitize(args.metric)}.png"
    plot_runs(
        payloads,
        args.metric,
        args.x_axis,
        plot_path,
        args.x_label,
        args.y_label,
        args.title,
        args.font_size,
        args.legend_font_size,
        args.legend_loc,
        args.color,
    )

    for payload in payloads:
        print(f"Wrote {payload['json_path']}")
        print(f"Fetched {len(payload['history'])} points from run {payload['run_id']}")
    print(f"Wrote plot to {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
