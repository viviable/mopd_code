# Data Generation

## Setup

Add the repository directory to `PYTHONPATH`:

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```

## Data Formats

This repo currently uses two data-preparation paths:

1. Raw JSON datasets with `train.json` and `test.json`, then converted to parquet with `data/preprocess.py`.
2. Hugging Face datasets or parquet-first datasets that already provide `train` / `test` splits and can be saved directly as `train.parquet` / `test.parquet`.

`run_local_sdpo.sh` and the training configs expect dataset directories that look like:

```text
datasets/<name>/train.parquet
datasets/<name>/test.parquet
```

## Benchmark Datasets

### LiveCodeBench v6

Export the benchmark to JSON:

```bash
python data/load_dataset.py \
    --dataset_name livecodebench/code_generation_lite-v6 \
    --output_path datasets/lcb_v6.json
```

Then create train/test splits:

```bash
python data/split_tests.py \
    --json_path datasets/lcb_v6.json \
    --output_dir datasets/lcb_v6
```

Finally preprocess to parquet:

```bash
python data/preprocess.py \
    --data_source datasets/lcb_v6
```

### SciKnowEval

Export the domain-specific JSON files:

```bash
python data/load_dataset.py --dataset_name Biology --output_path datasets/sciknoweval/biology/biology.json
python data/load_dataset.py --dataset_name Chemistry --output_path datasets/sciknoweval/chemistry/chemistry.json
python data/load_dataset.py --dataset_name Material --output_path datasets/sciknoweval/material/material.json
python data/load_dataset.py --dataset_name Physics --output_path datasets/sciknoweval/physics/physics.json
```

Create train/test splits:

```bash
python data/split_tasks.py \
    --json_path datasets/sciknoweval/biology/biology.json \
    --output_dir datasets/sciknoweval/biology

python data/split_tasks.py \
    --json_path datasets/sciknoweval/chemistry/chemistry.json \
    --output_dir datasets/sciknoweval/chemistry

python data/split_tasks.py \
    --json_path datasets/sciknoweval/material/material.json \
    --output_dir datasets/sciknoweval/material

python data/split_tasks.py \
    --json_path datasets/sciknoweval/physics/physics.json \
    --output_dir datasets/sciknoweval/physics
```

Then preprocess each directory to parquet:

```bash
python data/preprocess.py --data_source datasets/sciknoweval/biology
python data/preprocess.py --data_source datasets/sciknoweval/chemistry
python data/preprocess.py --data_source datasets/sciknoweval/material
python data/preprocess.py --data_source datasets/sciknoweval/physics
```

### ToolUse

The raw JSON files are already provided:

```text
datasets/tooluse/train.json
datasets/tooluse/test.json
```

Convert them to parquet:

```bash
python data/preprocess.py --data_source datasets/tooluse
```

## EvalPlus Conversion

`data/convert_evalplus_to_parquet.py` converts EvalPlus JSONL files directly to OPD-compatible parquet.

Examples:

```bash
python data/convert_evalplus_to_parquet.py \
    --dataset humaneval \
    --input_path /path/to/HumanEvalPlus.jsonl \
    --output_path datasets/humanevalplus/humanevalplus/test.parquet

python data/convert_evalplus_to_parquet.py \
    --dataset mbpp \
    --input_path /path/to/MbppPlus.jsonl \
    --output_path datasets/mbppplus/test.parquet
```

These datasets are test-only in the current workflow, so only `test.parquet` is required when they are used as validation datasets.

## Hugging Face Parquet Datasets

For datasets that already exist on Hugging Face with split structure, save them locally as parquet and use them directly. `data/load_dataset.py` provides a helper for this:

```bash
python data/load_dataset.py \
    --download_hf_dataset_name <hf_dataset_name> \
    --download_output_dir <local_output_dir>
```

This writes each split as:

```text
<local_output_dir>/train.parquet
<local_output_dir>/test.parquet
```

### Eurus

If your training setup uses:

```bash
DATA_PATH="datasets/G-OPD-Training-Data/Eurus"
```

download the Hugging Face dataset repo to the matching local directory:

```bash
python data/load_dataset.py \
    --download_hf_dataset_name G-OPD-Training-Data/Eurus \
    --download_output_dir datasets/G-OPD-Training-Data/Eurus
```

After that, training can read:

```text
datasets/G-OPD-Training-Data/Eurus/train.parquet
datasets/G-OPD-Training-Data/Eurus/test.parquet
```

### DeepMath

For the local path used in `run_local_sdpo.sh`:

```bash
DATA_PATH="datasets/G-OPD-Training-Data/DeepMath-103K_test_aime2024"
```

the processed Hugging Face dataset repo is:

```text
vivi-yu/opd-math
```

Download it into the matching local directory:

```bash
python data/load_dataset.py \
    --download_hf_dataset_name vivi-yu/opd-math \
    --download_output_dir datasets/G-OPD-Training-Data/DeepMath-103K_test_aime2024
```

This produces:

```text
datasets/G-OPD-Training-Data/DeepMath-103K_test_aime2024/train.parquet
datasets/G-OPD-Training-Data/DeepMath-103K_test_aime2024/test.parquet
```

If you use a different local directory name, keep it aligned with the value passed to `DATA_PATH` in `run_local_sdpo.sh`.

## Generic JSON-to-Parquet Preprocessing

For datasets that already contain:

```text
DATASET_PATH/train.json
DATASET_PATH/test.json
```

run:

```bash
python data/preprocess.py \
    --data_source DATASET_PATH
```

This generates:

```text
DATASET_PATH/train.parquet
DATASET_PATH/test.parquet
```

## Notes

`data/prepare_data_splits.sh` is a convenience wrapper for JSON datasets that need duplicated train/test files, split generation, and parquet conversion.

`run_local_sdpo.sh` expects parquet files. If a dataset is already parquet-first, do not run `data/preprocess.py` on it again.
