#!/usr/bin/env python3
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_REPROMPT_TEMPLATE = "{prompt}{solution}{solution_summaries}{another_solution}{failure}{feedback}\n\nCorrectly solve the original question.\n"
DEFAULT_SOLUTION_TEMPLATE = "\nCorrect solution:\n\n{successful_previous_attempt}\n\n"
DEFAULT_ANOTHER_SOLUTION_TEMPLATE = "\nAnother successful solution:\n\n{another_successful_attempt}\n\n"
DEFAULT_FAILURE_TEMPLATE = "\nA known failed solution (avoid repeating this mistake):\n\n{failed_attempt}\n\n"
DEFAULT_FEEDBACK_TEMPLATE = "\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}\n\n"
DEFAULT_SUMMARY_TEMPLATE = "\nSummary of another successful solution:\n\n{summary_text}\n\n"
DEFAULT_SUMMARY_FAILED_TEMPLATE = "\nSummary of another unsuccessful solution:\n\n{summary_text}\n\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build teacher-context variants from candidate/evidence datasets.")
    parser.add_argument("--candidates", required=True, help="Path to candidate_responses.jsonl.")
    parser.add_argument("--evidence", required=True, help="Path to evidence_items.jsonl.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--variant-config",
        help="Optional JSON config describing context variants. Uses built-in defaults if omitted.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_variant_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {
            "global": {
                "dont_reprompt_on_self_success": True,
                "feedback_only_without_solution": False,
                "summary_k": 2,
                "summary_sampling_policy": "first_k",
            },
            "variants": [
                {"name": "base"},
                {"name": "base_raw"},
                {"name": "base_reprompt", "force_reprompt": True},
                {"name": "solution", "include_solution": True},
                {"name": "another_solution", "include_another_solution": True},
                {"name": "solution+another_solution", "include_solution": True, "include_another_solution": True},
                {"name": "all_solutions", "include_solution": True, "include_another_solution": True},
                {"name": "failure_solution", "include_failure_solution": True},
                {"name": "solution+failure_solution", "include_solution": True, "include_failure_solution": True},
                {
                    "name": "solution+another_solution+failure_solution",
                    "include_solution": True,
                    "include_another_solution": True,
                    "include_failure_solution": True,
                },
                {"name": "feedback", "include_feedback": True},
                {"name": "summary_success_k2", "include_summary": True, "summary_k": 2},
                {"name": "solution+feedback", "include_solution": True, "include_feedback": True},
                {"name": "solution+summary_success_k2", "include_solution": True, "include_summary": True, "summary_k": 2},
                {
                    "name": "solution+feedback+summary_all_k2",
                    "include_solution": True,
                    "include_feedback": True,
                    "include_summary": True,
                    "summary_from_all": True,
                    "summary_k": 2,
                },
                {"name": "failure_only", "include_failure_solution": True},
                {"name": "random_summary_control", "include_summary": True, "summary_control": "random_other_prompt", "summary_k": 2},
                # Matched-budget controls (mirror the training-side peer controls).
                # random_peer: keep the 2S1F slot structure (2 success slots + 1 failure slot)
                #   but fill each slot with a random same-prompt peer, ignoring its true label.
                # verifier_score_ordered: take the top-N peers by verifier score (reward), N = the
                #   same 2S1F block budget, rendered best->worst (top-N are usually all successes).
                {"name": "random_peer", "peer_control": "random"},
                {"name": "verifier_score_ordered", "peer_control": "score_ordered"},
            ],
        }
    with Path(path).open() as f:
        return json.load(f)


def stable_variant_id(condition: str, prompt_id: str, response_id: str) -> str:
    raw = f"{condition}|{prompt_id}|{response_id}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def index_evidence(evidence_rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    index: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in evidence_rows:
        index[row["source_response_id"]][row["evidence_type"]] = row
    return index


def group_candidates(candidates: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[row["prompt_id"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda x: x["response_id"])
    return grouped


def build_other_prompt_summary_pool(candidates_by_prompt: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    pool = []
    for prompt_id, rows in sorted(candidates_by_prompt.items()):
        for row in rows:
            pool.append(
                {
                    "prompt_id": prompt_id,
                    "response_id": row["response_id"],
                    "summary_text_effective": row["summary_text_effective"],
                    "summary_has_tag": row["summary_has_tag"],
                    "summary_is_fallback": row["summary_is_fallback"],
                    "success": row["success"],
                }
            )
    return pool


def select_primary_solution(rows: list[dict[str, Any]], self_response_id: str, dont_reprompt_on_self_success: bool) -> dict[str, Any] | None:
    for row in rows:
        if dont_reprompt_on_self_success and row["response_id"] == self_response_id:
            continue
        if row["success"]:
            return row
    return None


def select_another_solution(
    rows: list[dict[str, Any]],
    self_response_id: str,
    primary_response_id: str | None,
    dont_reprompt_on_self_success: bool,
) -> dict[str, Any] | None:
    for row in rows:
        if dont_reprompt_on_self_success and row["response_id"] == self_response_id:
            continue
        if primary_response_id is not None and row["response_id"] == primary_response_id:
            continue
        if row["success"]:
            return row
    return None


def select_failure_solution(rows: list[dict[str, Any]], self_response_id: str) -> dict[str, Any] | None:
    for row in rows:
        if row["response_id"] == self_response_id:
            continue
        if not row["success"]:
            return row
    return None


def select_summary_rows(
    rows: list[dict[str, Any]],
    self_response_id: str,
    primary_response_id: str | None,
    include_all: bool,
    summary_k: int,
    dont_reprompt_on_self_success: bool,
) -> list[dict[str, Any]]:
    pool = []
    for row in rows:
        if dont_reprompt_on_self_success and row["response_id"] == self_response_id:
            continue
        if primary_response_id is not None and row["response_id"] == primary_response_id:
            continue
        if include_all or row["success"]:
            pool.append(row)
    return pool[:summary_k]


def select_random_control_summaries(
    other_prompt_summary_pool: list[dict[str, Any]],
    current_prompt_id: str,
    summary_k: int,
) -> list[dict[str, Any]]:
    selected = []
    for item in other_prompt_summary_pool:
        if item["prompt_id"] == current_prompt_id:
            continue
        selected.append(item)
        if len(selected) >= summary_k:
            break
    return selected


def select_random_peers(
    rows: list[dict[str, Any]],
    self_response_id: str,
    n_peers: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Pick n random same-prompt peers (excluding self), ignoring their success label."""
    others = [row for row in rows if row["response_id"] != self_response_id]
    if n_peers <= 0 or not others:
        return []
    order = list(range(len(others)))
    random.Random(seed).shuffle(order)
    return [others[i] for i in order[:n_peers]]


def select_score_ordered_peers(
    rows: list[dict[str, Any]],
    self_response_id: str,
    n_peers: int,
) -> list[dict[str, Any]]:
    """Take the top-n same-prompt peers (excluding self) by verifier score (reward), best first."""
    others = [row for row in rows if row["response_id"] != self_response_id]
    if n_peers <= 0 or not others:
        return []
    others_sorted = sorted(others, key=lambda row: (-float(row["reward"]), str(row["response_id"])))
    return others_sorted[:n_peers]


def render_peer_sections(
    success_slot_peers: list[dict[str, Any]],
    failure_slot_peers: list[dict[str, Any]],
) -> tuple[str, str, str, list[str], dict[str, bool]]:
    """Render selected peers into the same (solution / another_solution / failure) slots used by 2S1F."""
    evidence_ids: list[str] = []
    sections_used = {
        "solution": False,
        "another_solution": False,
        "failure_solution": False,
        "feedback": False,
        "summary": False,
    }
    solution_section = ""
    another_solution_section = ""
    failure_section = ""

    if success_slot_peers:
        solution_section = DEFAULT_SOLUTION_TEMPLATE.format(
            successful_previous_attempt=success_slot_peers[0]["response_text"]
        )
        evidence_ids.append(f"{success_slot_peers[0]['response_id']}::response_full")
        sections_used["solution"] = True
        rest = success_slot_peers[1:]
        if rest:
            another_solution_section = "".join(
                DEFAULT_ANOTHER_SOLUTION_TEMPLATE.format(another_successful_attempt=row["response_text"])
                for row in rest
            )
            evidence_ids.extend(f"{row['response_id']}::response_full" for row in rest)
            sections_used["another_solution"] = True

    if failure_slot_peers:
        failure_section = "".join(
            DEFAULT_FAILURE_TEMPLATE.format(failed_attempt=row["response_text"])
            for row in failure_slot_peers
        )
        evidence_ids.extend(f"{row['response_id']}::response_full" for row in failure_slot_peers)
        sections_used["failure_solution"] = True

    return solution_section, another_solution_section, failure_section, evidence_ids, sections_used


def build_summary_block(
    selected_rows: list[dict[str, Any]],
    variant: dict[str, Any],
) -> tuple[str, list[str], list[dict[str, Any]]]:
    blocks = []
    evidence_ids: list[str] = []
    summary_sources: list[dict[str, Any]] = []
    for row in selected_rows:
        is_failure = not bool(row["success"])
        template = DEFAULT_SUMMARY_FAILED_TEMPLATE if is_failure and variant.get("summary_from_all", False) else DEFAULT_SUMMARY_TEMPLATE
        blocks.append(template.format(summary_text=row["summary_text_effective"]))
        evidence_ids.append(f"{row['response_id']}::summary")
        summary_sources.append(
            {
                "source_response_id": row["response_id"],
                "success": row["success"],
                "summary_has_tag": row["summary_has_tag"],
                "summary_is_fallback": row["summary_is_fallback"],
            }
        )
    return "".join(blocks), evidence_ids, summary_sources


def assemble_reprompt(
    prompt: str,
    solution_section: str,
    solution_summaries_section: str,
    another_solution_section: str,
    failure_section: str,
    feedback_section: str,
) -> str:
    return DEFAULT_REPROMPT_TEMPLATE.format(
        prompt=prompt,
        solution=solution_section,
        solution_summaries=solution_summaries_section,
        another_solution=another_solution_section,
        failure=failure_section,
        feedback=feedback_section,
    )


def should_use_reprompt(variant: dict[str, Any], sections_used: dict[str, bool]) -> bool:
    if variant.get("force_raw_prompt", False):
        return False
    if variant.get("force_reprompt", False):
        return True
    return any(sections_used.values())


def build_variant_rows(
    candidates: list[dict[str, Any]],
    evidence_index: dict[str, dict[str, dict[str, Any]]],
    variant_config: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates_by_prompt = group_candidates(candidates)
    other_prompt_summary_pool = build_other_prompt_summary_pool(candidates_by_prompt)
    global_cfg = variant_config.get("global", {})
    variants = variant_config.get("variants", [])

    rows_out = []
    for variant in variants:
        condition = variant["name"]
        dont_reprompt_on_self_success = variant.get(
            "dont_reprompt_on_self_success",
            global_cfg.get("dont_reprompt_on_self_success", False),
        )
        feedback_only_without_solution = variant.get(
            "feedback_only_without_solution",
            global_cfg.get("feedback_only_without_solution", False),
        )
        summary_k = int(variant.get("summary_k", global_cfg.get("summary_k", 1)))

        for candidate in candidates:
            prompt_rows = candidates_by_prompt[candidate["prompt_id"]]
            self_response_id = candidate["response_id"]
            primary = select_primary_solution(prompt_rows, self_response_id, dont_reprompt_on_self_success)
            primary_response_id = primary["response_id"] if primary else None

            evidence_ids: list[str] = []
            sections_used = {
                "solution": False,
                "another_solution": False,
                "failure_solution": False,
                "feedback": False,
                "summary": False,
            }

            solution_section = ""
            if variant.get("include_solution", False) and primary is not None:
                solution_section = DEFAULT_SOLUTION_TEMPLATE.format(
                    successful_previous_attempt=primary["response_text"]
                )
                evidence_ids.append(f"{primary['response_id']}::response_full")
                sections_used["solution"] = True

            another_solution_section = ""
            if variant.get("include_another_solution", False):
                another = select_another_solution(
                    prompt_rows,
                    self_response_id=self_response_id,
                    primary_response_id=primary_response_id,
                    dont_reprompt_on_self_success=dont_reprompt_on_self_success,
                )
                if another is not None:
                    another_solution_section = DEFAULT_ANOTHER_SOLUTION_TEMPLATE.format(
                        another_successful_attempt=another["response_text"]
                    )
                    evidence_ids.append(f"{another['response_id']}::response_full")
                    sections_used["another_solution"] = True

            failure_section = ""
            if variant.get("include_failure_solution", False):
                failure = select_failure_solution(prompt_rows, self_response_id=self_response_id)
                if failure is not None:
                    failure_section = DEFAULT_FAILURE_TEMPLATE.format(
                        failed_attempt=failure["response_text"]
                    )
                    evidence_ids.append(f"{failure['response_id']}::response_full")
                    sections_used["failure_solution"] = True

            feedback_section = ""
            if variant.get("include_feedback", False):
                use_feedback = candidate["has_feedback"] and (
                    not feedback_only_without_solution or primary is None
                )
                if use_feedback:
                    feedback_section = DEFAULT_FEEDBACK_TEMPLATE.format(
                        feedback_raw=candidate["feedback_text"]
                    )
                    evidence_ids.append(f"{candidate['response_id']}::feedback")
                    sections_used["feedback"] = True

            summary_sources: list[dict[str, Any]] = []
            solution_summaries_section = ""
            if variant.get("include_summary", False) and primary is not None:
                if variant.get("summary_control") == "random_other_prompt":
                    selected = select_random_control_summaries(
                        other_prompt_summary_pool,
                        current_prompt_id=candidate["prompt_id"],
                        summary_k=summary_k,
                    )
                else:
                    selected = select_summary_rows(
                        prompt_rows,
                        self_response_id=self_response_id,
                        primary_response_id=primary_response_id,
                        include_all=variant.get("summary_from_all", False),
                        summary_k=summary_k,
                        dont_reprompt_on_self_success=dont_reprompt_on_self_success,
                    )
                solution_summaries_section, summary_evidence_ids, summary_sources = build_summary_block(selected, variant)
                if solution_summaries_section:
                    evidence_ids.extend(summary_evidence_ids)
                    sections_used["summary"] = True

            # Matched-budget peer controls: override the solution/failure slots. The block budget
            # is derived from what 2S1F would occupy for THIS candidate, so length/#blocks match.
            peer_control = variant.get("peer_control")
            if peer_control:
                another = select_another_solution(
                    prompt_rows,
                    self_response_id=self_response_id,
                    primary_response_id=primary_response_id,
                    dont_reprompt_on_self_success=dont_reprompt_on_self_success,
                )
                failure = select_failure_solution(prompt_rows, self_response_id=self_response_id)
                n_success_slots = (1 if primary is not None else 0) + (1 if another is not None else 0)
                n_failure_slots = 1 if failure is not None else 0
                if peer_control == "random":
                    seed = int(
                        hashlib.sha1(f"{candidate['prompt_id']}|{self_response_id}".encode("utf-8")).hexdigest(),
                        16,
                    ) % (2**32)
                    picked = select_random_peers(
                        prompt_rows, self_response_id, n_success_slots + n_failure_slots, seed
                    )
                    success_slot_peers = picked[:n_success_slots]
                    failure_slot_peers = picked[n_success_slots : n_success_slots + n_failure_slots]
                else:  # score_ordered
                    picked = select_score_ordered_peers(
                        prompt_rows, self_response_id, n_success_slots + n_failure_slots
                    )
                    success_slot_peers = [row for row in picked if row["success"]]
                    failure_slot_peers = [row for row in picked if not row["success"]]
                (
                    solution_section,
                    another_solution_section,
                    failure_section,
                    evidence_ids,
                    sections_used,
                ) = render_peer_sections(success_slot_peers, failure_slot_peers)
                feedback_section = ""
                solution_summaries_section = ""
                summary_sources = []

            if should_use_reprompt(variant, sections_used):
                assembled_context_text = assemble_reprompt(
                    prompt=candidate["prompt"],
                    solution_section=solution_section,
                    solution_summaries_section=solution_summaries_section,
                    another_solution_section=another_solution_section,
                    failure_section=failure_section,
                    feedback_section=feedback_section,
                )
            else:
                assembled_context_text = candidate["prompt"]

            rows_out.append(
                {
                    "variant_id": stable_variant_id(condition, candidate["prompt_id"], candidate["response_id"]),
                    "condition": condition,
                    "prompt_id": candidate["prompt_id"],
                    "response_id": candidate["response_id"],
                    "uid": candidate["uid"],
                    "reward": candidate["reward"],
                    "success": candidate["success"],
                    "difficulty_bucket": candidate["difficulty_bucket"],
                    "variant_spec": {
                        "force_reprompt": bool(variant.get("force_reprompt", False)),
                        "force_raw_prompt": bool(variant.get("force_raw_prompt", False)),
                        "include_solution": bool(variant.get("include_solution", False)),
                        "include_another_solution": bool(variant.get("include_another_solution", False)),
                        "include_failure_solution": bool(variant.get("include_failure_solution", False)),
                        "include_feedback": bool(variant.get("include_feedback", False)),
                        "include_summary": bool(variant.get("include_summary", False)),
                        "summary_from_all": bool(variant.get("summary_from_all", False)),
                        "summary_k": summary_k,
                        "summary_control": variant.get("summary_control"),
                        "peer_control": variant.get("peer_control"),
                        "dont_reprompt_on_self_success": dont_reprompt_on_self_success,
                        "feedback_only_without_solution": feedback_only_without_solution,
                    },
                    "sections_used": sections_used,
                    "evidence_ids": evidence_ids,
                    "primary_solution_response_id": primary_response_id,
                    "summary_sources": summary_sources,
                    "assembled_context_text": assembled_context_text,
                    "assembled_context_length_chars": len(assembled_context_text),
                }
            )
    return rows_out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    candidates = read_jsonl(Path(args.candidates))
    evidence_rows = read_jsonl(Path(args.evidence))
    variant_config = read_variant_config(args.variant_config)

    variant_rows = build_variant_rows(
        candidates=candidates,
        evidence_index=index_evidence(evidence_rows),
        variant_config=variant_config,
    )
    write_jsonl(Path(args.output), variant_rows)
    print(f"Wrote context variants to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
