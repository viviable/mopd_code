#!/usr/bin/env python3
"""Load parquet datasets from a Hugging Face dataset repo.

Expected repo layout (from upload_parquet_datasets_to_hf.py):
  lcb_v6/train.parquet
  lcb_v6/test.parquet
  sciknoweval/biology/train.parquet
  sciknoweval/biology/test.parquet
  ...

This script discovers such pairs, loads each dataset with `datasets.load_dataset`,
and optionally saves loaded Arrow datasets to disk.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load parquet datasets from HF repo.")
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HF dataset repo id, e.g. user_or_org/sdpo-parquet",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token. If omitted, uses HF_TOKEN env var or local login.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help=(
            "Dataset name to load (relative folder in repo), e.g. lcb_v6 or "
            "sciknoweval/biology. Can be passed multiple times. "
            "If omitted, loads all discovered datasets."
        ),
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help=(
            "Optional local output directory to save each loaded dataset with "
            "DatasetDict.save_to_disk()."
        ),
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list discovered datasets and their parquet files.",
    )
    return parser.parse_args()


def discover_dataset_files(repo_id: str, token: str | None) -> dict[str, dict[str, str]]:
    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

    grouped: dict[str, dict[str, str]] = defaultdict(dict)
    for f in files:
        if not f.endswith(".parquet"):
            continue
        p = Path(f)
        if p.name not in {"train.parquet", "test.parquet"}:
            continue
        dataset_name = p.parent.as_posix()
        split = p.stem  # train / test
        grouped[dataset_name][split] = f

    # Keep only datasets that have at least one split parquet.
    return dict(sorted(grouped.items(), key=lambda x: x[0]))


def main() -> None:
    args = parse_args()

    discovered = discover_dataset_files(args.repo_id, args.token)
    if not discovered:
        print(f"No train/test parquet files found in {args.repo_id}")
        return

    selected = set(args.dataset) if args.dataset else set(discovered.keys())
    missing = sorted(selected - set(discovered.keys()))
    if missing:
        raise ValueError(
            "Requested dataset(s) not found in repo: "
            + ", ".join(missing)
            + f"\nAvailable: {', '.join(discovered.keys())}"
        )

    print(f"Repo: {args.repo_id}")
    print("Discovered datasets:")
    for name, split_map in discovered.items():
        train_f = split_map.get("train", "<missing>")
        test_f = split_map.get("test", "<missing>")
        marker = "*" if name in selected else " "
        print(f" {marker} {name}: train={train_f}, test={test_f}")

    if args.list_only:
        return

    save_root = Path(args.save_dir).resolve() if args.save_dir else None
    if save_root:
        save_root.mkdir(parents=True, exist_ok=True)

    for name in sorted(selected):
        split_map = discovered[name]
        data_files = {k: v for k, v in split_map.items() if k in {"train", "test"}}
        if not data_files:
            print(f"Skipping {name}: no train/test parquet found.")
            continue

        print(f"\nLoading {name} from {args.repo_id} ...")
        ds = load_dataset(args.repo_id, data_files=data_files)
        for split_name, split_ds in ds.items():
            print(f"  - {split_name}: {split_ds.num_rows} rows")

        if save_root:
            out_dir = save_root / name
            out_dir.parent.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(out_dir))
            print(f"  saved to: {out_dir}")


if __name__ == "__main__":
    main()
