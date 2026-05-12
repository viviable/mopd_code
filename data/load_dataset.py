import argparse
from pathlib import Path
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from data.format.train import load_train
from data.format.gpqa import load_gpqa
from data.format.mmlu_pro import load_mmlu_pro
from data.format.math import load_math
from data.format.code import load_code
from data.format.sciknoweval import load_sciknoweval
from data.utils.data_handling import write_hf_to_json


def _add_embeddings(ds, embeddings_file = None):
    import numpy as np

    if not embeddings_file is None:
        embeddings = np.load(embeddings_file)
        ds = ds.add_column("embedding", [embeddings[i] for i in range(len(ds))])
    else:
        ds = ds.add_column("embedding", [np.array([])] * len(ds))
    return ds


def load_dataset_hf(
    dataset_name: str,
    output_path: str | None,
    start_idx: int = 0,
    num_el: int = None,
    category: str | None = None,
    embeddings_file: str | None = None,
) -> Dataset:

    final_columns = ["idx", "kind", "dataset", "answer", "elo", "prompt", "description", "tests", "embedding", "system"]

    if category == "false":
        category = None

    if dataset_name == "lasgroup/verifiable-corpus":
        ds = load_train(category)
    elif dataset_name == "Idavidrein/gpqa-D":
        ds = load_gpqa(category)
    elif dataset_name == "TIGER-Lab/MMLU-Pro":
        ds = load_mmlu_pro(category, implementation="evalchemy")
    elif dataset_name in ["math-ai/aime24", "math-ai/aime25", "math-ai/math500", "math-ai/amc23", "openai/gsm8k"]:
        ds = load_math(dataset_name)
    elif dataset_name in ["open-r1/codeforces", "Qwen/CodeElo", "livecodebench/code_generation_lite-v6", "evalplus/humanevalplus", "evalplus/mbppplus"]:
        ds = load_code(dataset_name)
    elif dataset_name == "tooluse":
        print("Tooluse dataset is already loaded. You can proceed to preprocess it.")
    elif dataset_name in ["Biology", "Chemistry", "Material", "Physics"]:
        ds = load_sciknoweval(
            domains=[dataset_name],
            levels=["L3"],
            types=["mcq-4-choices", "mcq-2-choices"],
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    ds = ds.add_column("idx", list(range(len(ds))))
    ds = _add_embeddings(ds, embeddings_file=embeddings_file)


    # Common shape across datasets
    print(len(ds))
    if not "tests" in ds.column_names:
        ds = ds.add_column("tests", ["-"] * len(ds))
    else:
        ds = ds.map(lambda ex: {"tests": "-" if ex["tests"] is None else ex["tests"]},  writer_batch_size=64)

    if not "answer" in ds.column_names:
        ds = ds.add_column("answer", ["-"] * len(ds))
    else:
        ds = ds.map(lambda ex: {"answer": "-" if ex["answer"] is None else ex["answer"]})

    if not "dataset" in ds.column_names:
        ds = ds.add_column("dataset", [dataset_name] * len(ds))
    else:
        ds = ds.map(lambda ex: {"dataset": "-" if ex["dataset"] is None else ex["dataset"]})

    if not "elo" in ds.column_names:
        ds = ds.add_column("elo", ["-"] * len(ds))
    else:
        ds = ds.map(lambda ex: {"elo": "-" if ex["elo"] is None else ex["elo"]})

    # Add correct suffix to each description
    ds = ds.map(lambda ex: {"description": ex["description"] + f" The solution will be evaluated in a {ex['kind']} environment."})

    final_columns = [c for c in final_columns if c in ds.column_names]
    ds = ds.select_columns(final_columns)


    # Save dataset
    if num_el is None:
        num_el = len(ds)
    last_el = min(start_idx + num_el, len(ds))

    ds_filtered = ds.select(range(start_idx, last_el))
    print(ds_filtered.column_names)
    print(f"Export to file {output_path}.")
    print(ds_filtered.features)
    print(f"Final number dataset samples: {len(ds_filtered)}")

    if output_path is None:
        return ds_filtered
    else:
        write_hf_to_json(
            ds=ds_filtered,
            output_path=output_path
        )


def push_dataset_to_hf(dataset_path: str, hf_dataset_name: str) -> None:
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    try:
        ds = load_from_disk(str(dataset_path))
        print(f"Loaded HF dataset directory from {dataset_path}.")
    except FileNotFoundError:
        parquet_files = {}
        for split in ["train", "validation", "val", "test"]:
            split_path = dataset_path / f"{split}.parquet"
            if split_path.exists():
                normalized_split = "validation" if split == "val" else split
                parquet_files[normalized_split] = str(split_path)

        if not parquet_files:
            parquet_files = {
                file_path.stem: str(file_path)
                for file_path in sorted(dataset_path.glob("*.parquet"))
            }

        if not parquet_files:
            raise FileNotFoundError(
                f"Directory {dataset_path} is neither a Hugging Face saved dataset nor a parquet dataset directory."
            )

        loaded_splits = {}
        all_columns = set()
        column_features = {}
        for split_name, split_file in parquet_files.items():
            split_ds = load_dataset("parquet", data_files={split_name: split_file})[split_name]
            loaded_splits[split_name] = split_ds
            all_columns.update(split_ds.column_names)
            for column_name, feature in split_ds.features.items():
                column_features.setdefault(column_name, feature)

        def _default_value_for_feature(feature):
            dtype = getattr(feature, "dtype", None)
            if dtype == "string":
                return ""
            if dtype and dtype.startswith("int"):
                return 0
            if dtype and dtype.startswith("float"):
                return 0.0
            if dtype == "bool":
                return False
            if isinstance(feature, dict):
                return {
                    key: _default_value_for_feature(sub_feature)
                    for key, sub_feature in feature.items()
                }
            if hasattr(feature, "feature"):
                return []
            return None

        aligned_splits = {}
        ordered_columns = sorted(all_columns)
        for split_name, split_ds in loaded_splits.items():
            missing_columns = [column for column in ordered_columns if column not in split_ds.column_names]
            for column in missing_columns:
                if column == "id":
                    default_value = "DeepMath"
                else:
                    default_value = _default_value_for_feature(column_features[column])
                split_ds = split_ds.add_column(column, [default_value] * len(split_ds))
            aligned_splits[split_name] = split_ds.select_columns(ordered_columns)

        ds = DatasetDict(aligned_splits)
        print(f"Loaded parquet dataset from {dataset_path} with splits: {list(parquet_files.keys())}.")

    print(f"Pushing dataset to hub: {hf_dataset_name}")
    if isinstance(ds, (Dataset, DatasetDict)):
        ds.push_to_hub(hf_dataset_name)
    else:
        raise TypeError(f"Unsupported dataset object type: {type(ds)}")
    print("Push complete.")


def download_dataset_from_hf(hf_dataset_name: str, output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(hf_dataset_name)
    print(f"Loaded dataset from hub: {hf_dataset_name}")
    for split_name, split_ds in ds.items():
        normalized_split = "validation" if split_name == "val" else split_name
        split_output_path = output_path / f"{normalized_split}.parquet"
        split_ds.to_parquet(str(split_output_path))
        print(f"Saved {normalized_split} split to {split_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Produce sorted dataset that can be used for training on most relevant questions."
    )
    parser.add_argument(
        "--dataset_name", type=str,
        help="HF dataset name."
    )
    parser.add_argument(
        "--output_path", type=str,
        help="File where the dataset will be saved."
    )
    parser.add_argument(
        "--start_idx", type=int, default=0,
        help="Start index used from the final dataset."
    )
    parser.add_argument(
        "--num_el", type=int, default=None,
        help="End index used from the final dataset."
    )
    parser.add_argument(
        "--category", type=str, default=None,
        help="File where the dataset will be saved."
    )
    parser.add_argument(
        "--embeddings_file", type=str, default=None,
        help="Where the embeddings for the dataset lie."
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.0,
        help="Test ratio for the dataset."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for the dataset."
    )
    parser.add_argument(
        "--push_dataset_path", type=str, default=None,
        help="Path to a dataset saved on disk with save_to_disk."
    )
    parser.add_argument(
        "--push_hf_dataset_name", type=str, default=None,
        help="HF dataset name to push the on-disk dataset to."
    )
    parser.add_argument(
        "--download_hf_dataset_name", type=str, default=None,
        help="HF dataset name to download from the hub."
    )
    parser.add_argument(
        "--download_output_dir", type=str, default=None,
        help="Local directory to save downloaded splits as <split>.parquet."
    )
    args = parser.parse_args()
    if args.download_hf_dataset_name is not None or args.download_output_dir is not None:
        if args.download_hf_dataset_name is None or args.download_output_dir is None:
            raise ValueError("Both --download_hf_dataset_name and --download_output_dir must be provided together.")
        download_dataset_from_hf(
            hf_dataset_name=args.download_hf_dataset_name,
            output_dir=args.download_output_dir,
        )
    elif args.push_dataset_path is not None or args.push_hf_dataset_name is not None:
        if args.push_dataset_path is None or args.push_hf_dataset_name is None:
            raise ValueError("Both --push_dataset_path and --push_hf_dataset_name must be provided together.")
        push_dataset_to_hf(
            dataset_path=args.push_dataset_path,
            hf_dataset_name=args.push_hf_dataset_name,
        )
    else:
        load_dataset_hf(
            dataset_name=args.dataset_name,
            output_path=args.output_path,
            start_idx=args.start_idx,
            num_el=args.num_el,
            category=args.category,
            embeddings_file=args.embeddings_file,
        )
