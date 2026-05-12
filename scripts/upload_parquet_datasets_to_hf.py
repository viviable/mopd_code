#!/usr/bin/env python3
"""Upload local parquet datasets to a Hugging Face dataset repository.

This script scans a root directory for train/test parquet files and uploads them
to a single HF dataset repo while preserving relative folder layout.

Example:
  python scripts/upload_parquet_datasets_to_hf.py \
    --repo-id your_name/sdpo-parquet \
    --datasets-root ./datasets
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload local parquet datasets to HF Hub.")
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target HF dataset repo id, e.g. user_or_org/repo_name",
    )
    parser.add_argument(
        "--datasets-root",
        default="./datasets",
        help="Local root folder that contains dataset subfolders.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Target branch/revision in the HF repo.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repo as private if it does not exist.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token. If omitted, uses HF_TOKEN env var or local login.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print upload plan without pushing files.",
    )
    return parser.parse_args()


def discover_parquet_pairs(datasets_root: Path) -> list[Path]:
    files: list[Path] = []
    for p in datasets_root.rglob("*.parquet"):
        if p.name in {"train.parquet", "test.parquet"}:
            files.append(p)
    return sorted(files)


def main() -> None:
    args = parse_args()
    root = Path(args.datasets_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"datasets root not found: {root}")

    parquet_files = discover_parquet_pairs(root)
    if not parquet_files:
        print(f"No train/test parquet files found under {root}")
        return

    print("Found parquet files:")
    for p in parquet_files:
        print(f"  - {p}")

    if args.dry_run:
        print("\nDry run enabled. No files uploaded.")
        return

    api = HfApi(token=args.token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    print(f"\nUploading to dataset repo: {args.repo_id} (revision={args.revision})")
    for local_path in parquet_files:
        rel = local_path.relative_to(root).as_posix()
        # Keep directory structure from datasets root, e.g.:
        # lcb_v6/train.parquet, sciknoweval/biology/test.parquet, ...
        path_in_repo = rel
        print(f"  -> {local_path}  ==>  {path_in_repo}")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            commit_message=f"Add {path_in_repo}",
        )

    print("\nUpload complete.")


if __name__ == "__main__":
    main()
