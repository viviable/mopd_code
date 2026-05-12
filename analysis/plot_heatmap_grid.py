#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw a paper-style heatmap with value annotations, highlighted cells, and optional dots."
    )
    parser.add_argument("--input", help="Path to a JSON spec file.")
    parser.add_argument("--output", required=True, help="Output image path, e.g. figure.png or figure.pdf.")
    parser.add_argument("--demo", action="store_true", help="Use a built-in demo spec similar to the provided figure.")
    return parser.parse_args()


def demo_spec() -> dict[str, Any]:
    return {
        "data": [
            [0.47, 0.37, 0.67, 0.57, 0.50, 0.50, 0.40, 0.30, 0.50, 0.50, 0.48],
            [0.30, 0.30, 0.73, 0.63, 0.57, 0.40, 0.47, 0.37, 0.43, 0.40, 0.46],
            [0.27, 0.30, 0.73, 0.50, 0.53, 0.60, 0.63, 0.40, 0.37, 0.50, 0.48],
            [0.43, 0.23, 0.73, 0.47, 0.40, 0.53, 0.50, 0.33, 0.47, 0.47, 0.46],
            [0.33, 0.30, 0.67, 0.50, 0.40, 0.50, 0.33, 0.47, 0.57, 0.53, 0.46],
            [0.67, 0.60, 0.90, 0.80, 0.80, 0.83, 0.67, 0.63, 0.73, 0.77, 0.74],
        ],
        "row_labels": ["(A)", "(U, A)", "(U, R, A)", "(R, C, R, A)", "(R, H, R, A)", "Oracle"],
        "col_labels": [
            "MechEng",
            "Music",
            "ArtTh.",
            "ClinMed",
            "Energy",
            "Acct.",
            "Econ.",
            "ArchEng",
            "Mater.",
            "Math",
            "Overall",
        ],
        "figsize": [11.8, 4.3],
        "cmap": "RdYlBu",
        "vmin": 0.0,
        "vmax": 1.0,
        "colorbar_label": "Accuracy",
        "annotation_fmt": "{:.0%}",
        "grid_linewidth": 1.2,
        "x_rotation": 30,
        "highlights": [
            {"row": 0, "col": 0},
            {"row": 0, "col": 1},
            {"row": 1, "col": 2},
            {"row": 1, "col": 3},
            {"row": 1, "col": 4},
            {"row": 2, "col": 2},
            {"row": 2, "col": 5},
            {"row": 2, "col": 6},
            {"row": 2, "col": 10},
            {"row": 3, "col": 2},
            {"row": 4, "col": 7},
            {"row": 4, "col": 8},
            {"row": 4, "col": 9},
        ],
        "dots": [
            {"row": 0, "col": 0, "color": "#86b6de"},
            {"row": 0, "col": 1, "color": "#86b6de"},
            {"row": 1, "col": 2, "color": "#f6c252"},
            {"row": 1, "col": 3, "color": "#f6c252"},
            {"row": 1, "col": 4, "color": "#f6c252"},
            {"row": 2, "col": 2, "color": "#f28b5b"},
            {"row": 2, "col": 5, "color": "#f28b5b"},
            {"row": 2, "col": 6, "color": "#f28b5b"},
            {"row": 2, "col": 10, "color": "#f28b5b"},
            {"row": 3, "col": 2, "color": "#e65f82"},
            {"row": 4, "col": 7, "color": "#c05b8e"},
            {"row": 4, "col": 8, "color": "#c05b8e"},
            {"row": 4, "col": 9, "color": "#c05b8e"},
        ],
        "top_bands": [
            {"col_start": 0, "col_end": 1, "color": "#8ec1e6"},
            {"col_start": 2, "col_end": 4, "color": "#f6c252"},
            {"col_start": 5, "col_end": 6, "color": "#ef8a62"},
            {"col_start": 7, "col_end": 9, "color": "#c05b8e"},
            {"col_start": 10, "col_end": 10, "color": "#f07f45"},
        ],
        "top_label": "Best",
    }


def load_spec(args: argparse.Namespace) -> dict[str, Any]:
    if args.demo:
        return demo_spec()
    if not args.input:
        raise ValueError("Pass --input SPEC.json or use --demo.")
    with Path(args.input).open("r", encoding="utf-8") as f:
        return json.load(f)


def draw_heatmap(spec: dict[str, Any], output_path: Path) -> None:
    data = np.asarray(spec["data"], dtype=float)
    row_labels = spec["row_labels"]
    col_labels = spec["col_labels"]
    if data.shape != (len(row_labels), len(col_labels)):
        raise ValueError("Shape of data must match row_labels x col_labels.")

    figsize = tuple(spec.get("figsize", [12, 4.5]))
    cmap = spec.get("cmap", "RdYlBu")
    vmin = spec.get("vmin", 0.0)
    vmax = spec.get("vmax", 1.0)
    annotation_fmt = spec.get("annotation_fmt", "{:.0%}")
    x_rotation = spec.get("x_rotation", 35)
    grid_linewidth = spec.get("grid_linewidth", 1.0)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", origin="upper")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=x_rotation, ha="right", fontsize=12)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=13)

    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=grid_linewidth, alpha=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    threshold = (vmin + vmax) / 2
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            color = "white" if value >= threshold * 0.9 else "#3d3d3d"
            ax.text(j, i, annotation_fmt.format(value), ha="center", va="center", fontsize=11, color=color, fontweight=600)

    for item in spec.get("highlights", []):
        rect = patches.Rectangle(
            (item["col"] - 0.5, item["row"] - 0.5),
            1,
            1,
            fill=False,
            edgecolor=item.get("color", "#3a3a3a"),
            linewidth=item.get("linewidth", 2.0),
        )
        ax.add_patch(rect)

    for item in spec.get("dots", []):
        dot = patches.Circle(
            (item["col"] + item.get("x_offset", 0.34), item["row"] + item.get("y_offset", -0.32)),
            radius=item.get("radius", 0.09),
            facecolor=item.get("color", "#f28b5b"),
            edgecolor=item.get("edgecolor", "#666666"),
            linewidth=item.get("linewidth", 1.0),
            zorder=4,
        )
        ax.add_patch(dot)

    for band in spec.get("top_bands", []):
        width = band["col_end"] - band["col_start"] + 1
        band_rect = patches.Rectangle(
            (band["col_start"] - 0.5, -0.66),
            width,
            0.12,
            facecolor=band["color"],
            edgecolor="none",
            clip_on=False,
        )
        ax.add_patch(band_rect)

    if "top_label" in spec:
        ax.text(-1.2, -0.58, spec["top_label"], ha="left", va="center", fontsize=13, color="#4b4b4b")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.012)
    cbar.set_label(spec.get("colorbar_label", ""), fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    for spine in ax.spines.values():
        spine.set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    spec = load_spec(args)
    draw_heatmap(spec, Path(args.output))
    print(f"Wrote figure to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
