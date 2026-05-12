import argparse
import json
from pathlib import Path
import sys

from datasets import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.format.prompts import CODE_PROMPT

TIME_LIMIT = 5


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _build_chat_prompt(problem: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": CODE_PROMPT.format(problem=problem)}]


def _parse_humaneval_description(problem: str, fn_name: str) -> str:
    text = problem.split(f"def {fn_name}")[1]
    if '"""' in text:
        text = text.split('"""')[1]
    elif "'''" in text:
        text = text.split("'''")[1]
    else:
        raise ValueError(f"Unexpected HumanEval prompt format for function {fn_name}")
    text = text.split(">>>")[0].strip()
    return text


def _build_humaneval_problem(row: dict) -> str:
    return (
        "Your task is to complete the following function. "
        "You are not allowed to modify the given code and should do the completion only. "
        "Here is the given code to complete: ```python\n"
        f"{row['prompt']}\n```"
    )


def _build_humaneval_tests(row: dict) -> str:
    tests = {
        "inputs": [row["test"] + "\n" + f"check({row['entry_point']})"],
        "outputs": [""],
        "testtype": "code",
        "fn_name": "",
        "time_limit": TIME_LIMIT,
    }
    return json.dumps(tests, ensure_ascii=False)


def _parse_mbpp_fn_name(code: str) -> str:
    return code.split("def ")[1].split("(")[0].strip()


def _build_mbpp_problem(row: dict) -> str:
    fn_name = _parse_mbpp_fn_name(row["canonical_solution"])
    return f"{row['prompt']} The function should be called `{fn_name}`."


def _build_mbpp_tests(row: dict) -> str:
    tests = {
        "inputs": [row["assertion"]],
        "outputs": [""],
        "testtype": "code",
        "fn_name": "",
        "time_limit": TIME_LIMIT,
    }
    return json.dumps(tests, ensure_ascii=False)


def _convert_humaneval(rows: list[dict]) -> list[dict]:
    converted = []
    for idx, row in enumerate(rows):
        tests = _build_humaneval_tests(row)
        converted.append(
            {
                "id": row["task_id"],
                "ability": "code",
                "data_source": "humanevalplus",
                "extra_info": {"index": idx, "split": "test"},
                "reward_model": {"ground_truth": tests, "style": "rule"},
                "prompt": _build_chat_prompt(_build_humaneval_problem(row)),
                "kind": "code",
                "dataset": "humanevalplus",
                "description": _parse_humaneval_description(row["prompt"], row["entry_point"]),
                "problem": row["prompt"],
                "tests": tests,
            }
        )
    return converted


def _convert_mbpp(rows: list[dict]) -> list[dict]:
    converted = []
    for idx, row in enumerate(rows):
        tests = _build_mbpp_tests(row)
        converted.append(
            {
                "id": row["task_id"],
                "ability": "code",
                "data_source": "mbppplus",
                "extra_info": {"index": idx, "split": "test"},
                "reward_model": {"ground_truth": tests, "style": "rule"},
                "prompt": _build_chat_prompt(_build_mbpp_problem(row)),
                "kind": "code",
                "dataset": "mbppplus",
                "description": row["prompt"],
                "problem": row["prompt"],
                "tests": tests,
            }
        )
    return converted


def _build_math_prompt(problem: str) -> list[dict[str, str]]:
    instruction = "Let's think step by step and output the final answer within \\boxed{}."
    return [{"role": "user", "content": f"{problem} {instruction}"}]


def _convert_math(rows: list[dict], data_source: str) -> list[dict]:
    converted = []
    for idx, row in enumerate(rows):
        converted.append(
            {
                "id": str(row["id"]),
                "ability": "math",
                "data_source": data_source,
                "extra_info": {
                    "index": idx,
                    "split": "test",
                    "answer": row["answer"],
                    "question": row["problem"],
                },
                "reward_model": {"ground_truth": row["answer"], "style": "rule"},
                "prompt": _build_math_prompt(row["problem"]),
            }
        )
    return converted


def convert_evalplus_to_parquet(dataset_name: str, input_path: str, output_path: str) -> None:
    raw_rows = _read_jsonl(Path(input_path))
    if dataset_name == "humaneval":
        rows = _convert_humaneval(raw_rows)
    elif dataset_name == "mbpp":
        rows = _convert_mbpp(raw_rows)
    elif dataset_name in {"hmmt25_feb", "hmmt25_nov"}:
        rows = _convert_math(raw_rows, data_source=dataset_name)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ds = Dataset.from_list(rows)
    ds.to_parquet(str(output))
    print(f"Converted {len(ds)} rows to {output}")
    print(ds.features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert EvalPlus jsonl benchmark files to OPD-compatible parquet.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["humaneval", "mbpp", "hmmt25_feb", "hmmt25_nov"],
        help="Dataset name.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to HumanEvalPlus.jsonl or MbppPlus.jsonl.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output parquet path, e.g. datasets/humanevalplus/test.parquet.",
    )
    args = parser.parse_args()
    convert_evalplus_to_parquet(
        dataset_name=args.dataset,
        input_path=args.input_path,
        output_path=args.output_path,
    )
