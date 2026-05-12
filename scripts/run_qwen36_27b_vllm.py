import argparse

from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--prompt", default="/no_think Explain tensor parallelism in one sentence.")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
    )
    outputs = llm.generate([args.prompt], sampling_params)
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
