# Diversity Analysis

This directory provides an offline workflow for measuring rollout diversity and
linking it to accuracy metrics such as `pass@k`.

The intended use case is method comparison on a fixed prompt set, for example:

- `Base`
- `GRPO`
- `SDPO`
- `MOPD-V1`
- `MOPD-V2`
- `MOPD-V3`

The scripts here operate directly on `candidate_responses.jsonl` produced by
[`analysis/prepare_rollout_dataset.py`](../analysis/prepare_rollout_dataset.py).

## Main Questions

1. Are MOPD rollouts more diverse than baselines?
2. Does higher diversity correlate with higher `pass@k`?
3. Does V1 increase diversity inside successful rollouts?

## Implemented Metrics

For each `(method, prompt_id)` group, the pipeline computes:

- `self_bleu4`
  - Lower is more diverse.
- `pairwise_text_distance`
  - Mean `1 - difflib.SequenceMatcher.ratio(...)`.
- `distinct1`, `distinct2`
  - Distinct token ratio inside the rollout set.
- `unique_response_ratio`
  - Fraction of unique normalized responses.
- `ast_node_jaccard_distance`
  - Code-only structural diversity using Python AST node types.
- `success_self_bleu4`
  - Same as `self_bleu4`, restricted to successful rollouts.
- `success_pairwise_text_distance`
  - Same as pairwise text distance, restricted to successful rollouts.
- `success_ast_node_jaccard_distance`
  - Same as AST distance, restricted to successful rollouts.
- `pass@k`
  - For any requested `k` list.

## Outputs

`compute_metrics.py` writes:

- `summary.json`
  - Aggregate metrics by method.
- `summary.csv`
  - Flat table for paper tables / spreadsheet use.
- `per_prompt.jsonl`
  - Per-problem diversity and accuracy metrics.
- `case_studies.jsonl`
  - Top prompts with the largest diversity gap between methods.

`plot_metrics.py` writes:

- `passk_curve.png`
- `diversity_vs_passk.png`
- `method_diversity_bars.png`

## Minimal Workflow

### 1. Build candidate datasets for each method

Example:

```bash
python3 analysis/prepare_rollout_dataset.py \
  --input "/path/to/rollout_dir_or_jsonl" \
  --output-dir analysis_outputs/lcbv6_sdpo_candidates \
  --max-prompts 300 \
  --max-responses-per-prompt 8
```

Repeat this for every method checkpoint you want to compare.

### 2. Compute diversity metrics

```bash
python3 diversity/compute_metrics.py \
  --run SDPO=analysis_outputs/lcbv6_sdpo_candidates/candidate_responses.jsonl \
  --run MOPD_V1=analysis_outputs/lcbv6_mopd_v1_candidates/candidate_responses.jsonl \
  --run MOPD_V2=analysis_outputs/lcbv6_mopd_v2_candidates/candidate_responses.jsonl \
  --task code \
  --ks 1 2 4 8 \
  --output-dir analysis_outputs/diversity_lcbv6 \
  --intersect-prompts \
  --max-responses-per-prompt 8
```

### 3. Plot

```bash
python3 diversity/plot_metrics.py \
  --summary analysis_outputs/diversity_lcbv6/summary.json \
  --per-prompt analysis_outputs/diversity_lcbv6/per_prompt.jsonl \
  --output-dir analysis_outputs/diversity_lcbv6/plots \
  --scatter-k 8
```

## Notes

- `--task code` enables code extraction and AST-based metrics.
- `--intersect-prompts` is recommended for fair multi-method comparison.
- `success_*` metrics directly support the paper claim about diversity inside
  `Y+`.
- If you later want algorithm-label entropy from an LLM classifier, the current
  outputs already provide the grouped per-prompt samples needed for that extra
  stage.
