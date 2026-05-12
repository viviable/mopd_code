from verl.utils.reward_score.feedback import math
from verl.utils.reward_score.feedback import code
from verl.utils.reward_score.feedback import gpqa
from verl.utils.reward_score.feedback import mcq
from verl.utils.reward_score.feedback import tooluse


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
) -> dict:
    if extra_info is None:
        extra_info = {}

    if data_source in ["code", "livecodebench", "humanevalplus", "mbppplus"]:
        results = code.compute_score(solution_str, ground_truth, extra_info, sparse_rewards=True, max_test_cases=None)
    elif data_source in ["math", "math500", "dapo_math", "gsm8k"]:
        results = math.compute_score(solution_str, ground_truth, extra_info)
    elif data_source.startswith("aime") or data_source.startswith("AIME") or data_source.startswith("hmmt"):
        results = math.compute_score(solution_str, ground_truth, extra_info)
        if not results.get("acc", 0.0):
            from verl.utils.reward_score import math_verify

            verify_score = math_verify.compute_score(solution_str, ground_truth)
            if verify_score:
                results["score"] = 1.0
                results["acc"] = 1.0
    elif data_source in ["gpqa"]:
        results = gpqa.compute_score(solution_str, ground_truth)
    elif data_source in ["sciknoweval"]:
        results = mcq.compute_score(solution_str, ground_truth)
    elif data_source in ["tooluse"]:
        results = tooluse.compute_score(solution_str, ground_truth)
    else:
        from verl.utils.reward_score import default_compute_score

        results = default_compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
    return results
