<div align="center">

# Multi-Rollout On-Policy Distillation (MOPD)

[![Code](https://img.shields.io/badge/Code-Anonymous-000000?style=for-the-badge&logo=github&logoColor=white)](https://anonymous.4open.science/r/mopd_code-C7BC)

</div>

<div align="center" style="font-family: Arial, sans-serif;">
  <p>
    <a href="#-introduction" style="text-decoration: none; font-weight: bold;">📖 Introduction</a> •
    <a href="#-main-results" style="text-decoration: none; font-weight: bold;">📊 Main Results</a> •
    <a href="#-getting-started" style="text-decoration: none; font-weight: bold;">🚀 Getting Started</a>
  </p>
  <p>
    <a href="#-configuration-reference" style="text-decoration: none; font-weight: bold;">⚙️ Configuration Reference</a> •
    <a href="#-additional-baselines--ablations" style="text-decoration: none; font-weight: bold;">🧩 Baselines &amp; Ablations</a> •
    <a href="#-teacher-signal-analysis" style="text-decoration: none; font-weight: bold;">🔬 Teacher Signal Analysis</a>
  </p>
</div>

## 📖 Introduction

Large language models are often post-trained with sparse verifier rewards, which indicate whether a sampled trajectory succeeds but provide limited guidance about *where* reasoning succeeds or fails. On-policy distillation (OPD) offers denser token-level supervision by training on student-generated trajectories, yet existing methods typically distill each rollout independently and ignore the other attempts sampled for the same prompt.

**We propose Multi-Rollout On-Policy Distillation (MOPD)**, a peer-conditioned distillation framework that uses the student's local rollout group to construct more informative teacher signals. MOPD conditions the teacher on both successful and failed peer rollouts from the same problem instance:

- **Successes** provide positive evidence for valid reasoning patterns.
- **Failures** provide structured negative evidence about plausible mistakes to avoid.

This turns OPD from independent imitation of individual trajectories into a comparative learning process: the teacher is no longer only a global expert, but also a local diagnostic model that can recognize instance-specific errors and distinguish superficially plausible failures from correct solutions.

We study two peer-context constructions:

| Variant | Teacher context |
|---|---|
| **Positive peer imitation** | Condition on successful peer rollouts only |
| **Contrastive success–failure** | Condition on both successful and failed peer rollouts |

Experiments on competitive programming, mathematical reasoning, scientific question answering, and tool-use benchmarks show that MOPD consistently improves over standard on-policy baselines.

---

## 📊 Main Results

### LiveCodeBench v6

| Method | Qwen3-4B mean@8 | Qwen3-4B pass@8 | Qwen3-8B mean@8 | Qwen3-8B pass@8 |
|---|---|---|---|---|
| Base | 28.80 | 49.36 | 30.97 | 53.02 |
| GRPO | 40.75 | 55.43 | 43.65 | 58.72 |
| SDPO | 48.84 | 63.23 | 49.49 | 64.34 |
| **MOPD (ours)** | **57.01** | **65.48** | **61.82** | **67.23** |

### Math Reasoning (Qwen3-4B)

| Method | AIME2025 mean@8 | AIME2024 mean@8 | HMMT25 Feb mean@8 | HMMT25 Nov mean@8 |
|---|---|---|---|---|
| Qwen3-4B | 17.92 | 16.89 | 5.06 | 9.58 |
| GRPO | 25.80 | 17.09 | 16.92 | 13.16 |
| SDPO | 7.81 | 7.29 | 8.59 | 12.01 |
| **MOPD (ours)** | **25.41** | **28.54** | **18.50** | **15.83** |

### Science QA (Qwen3-8B)

| Method | Biology | Chemistry | Physics | Materials |
|---|---|---|---|---|
| Qwen3-8B | 30.89 | 42.98 | 58.44 | 65.59 |
| GRPO | 47.32 | 64.24 | 64.12 | 72.09 |
| SDPO | 50.60 | 62.91 | 67.36 | 72.34 |
| **MOPD (ours)** | **55.69** | **74.29** | **76.25** | **78.59** |

### Tool Use (Qwen3-8B)

| Method | Accuracy |
|---|---|
| Qwen3-8B | 59.11 |
| GRPO | 60.32 |
| SDPO | 63.45 |
| **MOPD (ours)** | **66.73** |

---

## 🚀 Getting Started

### System Requirements

- **Operating System:** Linux (tested on Ubuntu 22.04)
- **Hardware:** NVIDIA GPUs (CUDA compatible); experiments used 8×H100 or 8×A100
- **Python:** 3.10+
- **CUDA Driver:** Compatible with PyTorch version installed

---

### Installation

#### Option 1: Docker (Recommended)

```bash
# Build the image
docker build . -f Dockerfile -t mopd

# Or for GH200 clusters
podman build . -f Dockerfile.gh200 -t mopd-gh200
enroot import -x mount -o mopd-gh200.sqsh podman://localhost/mopd-gh200:latest
```

#### Option 2: Local Installation

1. **Install PyTorch** (Ampere/Hopper — H100, A100):
```bash
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

2. **Install MOPD and dependencies:**
```bash
pip install -r requirements.txt
pip install -e .
pip install flash-attn --no-build-isolation
```

3. **Optional — SGLang/vLLM for high-throughput rollouts:**
```bash
pip install -r requirements_sglang.txt
```

---

### Data Preparation

Pre-processed datasets for the four benchmarks are in the `datasets/` directory.

To reproduce the data splits from scratch:

```bash
# Science QA (Chemistry example)
python data/load_dataset.py \
    --dataset_name Chemistry \
    --output_path datasets/sciknoweval/chemistry.json

python data/split_tasks.py \
    --json_path datasets/sciknoweval/chemistry.json \
    --output_dir datasets/sciknoweval/chemistry \
    --test_ratio 0.1 \
    --seed 42

# LiveCodeBench v6
python data/split_tests.py \
    --json_path datasets/lcb_v6.json \
    --output_dir datasets/lcb_v6
```

Convert to parquet (required for training):
```bash
python data/preprocess.py --data_source DATASET_PATH
```

**Math training data:** Download the standard DeepMath-103K corpus from its public release and preprocess with `data/preprocess.py`.

---

### Training

#### MOPD (default: 2 successful + 1 failed peer context)

```bash
# LiveCodeBench v6
DATA_PATH=datasets/lcb_v6 bash run_local_sdpo.sh

# Science QA — Chemistry
DATA_PATH=datasets/sciknoweval/chemistry bash run_local_sdpo.sh

# Tool Use
DATA_PATH=datasets/tooluse bash run_local_sdpo.sh
```

Key flags that control the peer context (already set as defaults in `run_local_sdpo.sh`):

```bash
INCLUDE_PRIMARY_SOLUTION=True      # first successful peer
INCLUDE_ANOTHER_SOLUTION=True      # second successful peer
INCLUDE_FAILURE_SOLUTION=True      # one failed peer
FAILURE_SOLUTION_CONDITION=always  # include failure even when a success exists
```

To run **positive peer imitation only** (no failure peer):
```bash
INCLUDE_FAILURE_SOLUTION=False bash run_local_sdpo.sh
```

To run the **GRPO baseline**:
```bash
DATA_PATH=datasets/lcb_v6 bash run_local_grpo.sh
```

To run **teacher–student distillation** (Qwen3-14B → Qwen3-4B):
```bash
DATA_PATH=datasets/lcb_v6 \
TEACHER_MODEL_PATH=Qwen/Qwen3-14B-Instruct \
bash run_ts_sdpo.sh
```

---

### Reproducing Experiment Suites

```bash
# All generalization experiments (Science QA, Tool Use)
bash experiments/generalization/run_sdpo_all.sh
bash experiments/generalization/run_baseline_grpo_all.sh

# Rich-feedback experiments (LiveCodeBench v6)
bash experiments/rich_feedback/run_sdpo.sh
bash experiments/rich_feedback/run_baseline_grpo.sh
```

---

## ⚙️ Configuration Reference

MOPD extends the base `self_distillation` block in `trainer/config/actor/actor.yaml`.

### Peer-Context Settings

Located at `actor_rollout_ref.actor.self_distillation`:

| Parameter | Default | Description |
|---|---|---|
| `include_primary_solution` | `True` | Include the primary successful peer rollout |
| `include_another_solution` | `False` | Include a second successful peer rollout |
| `include_failure_solution` | `False` | Include a failed peer rollout |
| `failure_solution_condition` | `when_no_solution` | When to attach the failure peer: `when_no_solution` or `always` |

Set `include_another_solution=True`, `include_failure_solution=True`, `failure_solution_condition=always` to replicate the **2 Suc. + 1 Failure** context from the paper (best configuration per ablations).

### Core Distillation Settings

| Parameter | Default | Description |
|---|---|---|
| `alpha` | `0.5` | KL interpolation: `0.0` = forward KL, `1.0` = reverse KL, `0.5` = JSD |
| `distillation_topk` | `100` | Top-k logits for distillation (`null` = full distribution) |
| `success_reward_threshold` | `1.0` | Minimum sequence reward to qualify as a successful peer |
| `dont_reprompt_on_self_success` | `True` | Exclude a trajectory's own successful response from its peer context |

### Peer Context Templates

The templates injected into the teacher prompt are configurable:

```yaml
# Successful peer template (actor.yaml)
solution_template: |-
  Correct solution:

  {successful_previous_attempt}

another_solution_template: |-
  Another successful solution:

  {another_successful_attempt}

failure_solution_template: |-
  A known failed solution (avoid repeating this mistake):

  {failed_attempt}
```

---

## 🧩 Additional Baselines & Ablations

This release adds two capabilities on top of the core MOPD trainer: (1) a family of
**on-policy preference / filtered-SFT baselines** that reuse the exact same rollout
budget as GRPO/MOPD, and (2) **matched-budget peer-control ablations** that isolate
*what* in the peer context drives the gains (the same-prompt success–failure contrast)
from confounds like context length and demonstration count. All additions are
opt-in — with the default config, GRPO/MOPD batches are byte-for-byte unchanged.

### 1. On-policy DPO / SimPO / verified-success SFT baselines

**Purpose.** Provide apples-to-apples baselines under an identical rollout
budget. Preference pairs (DPO/SimPO) are formed *within* each rollout group (shared
`uid`): any higher-verified-reward sample is "chosen" over a lower one; verified-success
SFT (a.k.a. rejection-sampling / STaR) simply does NLL on the self-generated rollouts
that pass verification. Selected via `actor.policy_loss.loss_mode ∈ {dpo, simpo, sft}`.

| Loss mode | What it does | Reference policy |
|---|---|---|
| `dpo` | Within-group DPO on verified reward; cDPO label smoothing supported | required (auto-enabled) |
| `simpo` | Reference-free, length-normalized preference loss with a target margin | not needed |
| `sft` / `verified_sft` | NLL on verified-success rollouts only (filter-then-SFT) | not needed |

**Code changed:**

| File | Change |
|---|---|
| `verl/trainer/ppo/core_algos.py` | New `compute_preference_loss` (DPO/SimPO) and `compute_verified_sft_loss`, plus a `_sequence_log_probs` helper |
| `verl/workers/actor/dp_actor.py` | Dispatch the new `dpo`/`simpo`/`sft` loss modes; select `sequence_reward` (and `ref_log_prob`/`uid` where needed) into the micro-batch |
| `verl/trainer/ppo/ray_trainer.py` | Attach a per-sample scalar `sequence_reward` (summed verified reward) to the batch, only for these loss modes |
| `verl/trainer/ppo/utils.py` | `need_reference_policy` now returns `True` for `loss_mode == dpo` (DPO needs reference log-probs even without a KL term) |
| `verl/workers/config/actor.py`, `verl/trainer/config/actor/actor.yaml` | New `policy_loss` fields: `dpo_beta`, `simpo_beta`, `simpo_gamma`, `pref_label_smoothing`, `sft_success_threshold`, `sft_length_normalize` |
| `verl/trainer/config/{dpo,simpo,verified_sft}.yaml` *(new)* | Ready-to-use configs for each baseline |
| `run_local_{dpo,simpo,sft}.sh` *(new)* | Launch scripts mirroring `run_local_sdpo.sh` |

```bash
DATA_PATH=datasets/lcb_v6 bash run_local_dpo.sh    # on-policy DPO
DATA_PATH=datasets/lcb_v6 bash run_local_simpo.sh  # SimPO
DATA_PATH=datasets/lcb_v6 bash run_local_sft.sh    # verified-success SFT
```

### 2. Matched-budget peer-control ablations

**Purpose.** The 2-success-1-failure (2S1F) MOPD context differs from a bare GRPO
prompt in three ways at once: it adds tokens, adds demonstrations, and adds the
same-prompt success/failure structure. These controls hold the **block count / token
budget fixed** and vary only *what fills the slots*, so the same-instance contrast can
be credited (or not) on its own.

Set via `actor_rollout_ref.actor.self_distillation.peer_control_mode` (with
`peer_control_match ∈ {block, token}`):

| `peer_control_mode` | Fills the 2S1F slots with… |
|---|---|
| `none` (default) | Standard MOPD (primary + another + failure) |
| `random_peer` | Same #success/#failure slots, but each slot is a peer sampled at random from the group, ignoring its verifier label |
| `unrelated_prompt` | Successful rollouts drawn from *other* prompts (no same-instance structure) |
| `success_only_matched` | Successes from the *same* prompt only (failure slot replaced by an extra success), isolating the failure-as-negative-evidence term |

**Code changed:**

| File | Change |
|---|---|
| `verl/trainer/ppo/ray_trainer.py` | New `_build_control_peers` (re-derives each sample's 2S1F budget, then fills it with control peers); control peers take over the solution/failure slots in `_maybe_build_self_distillation_batch`, with `self_distillation/peer_control_*` diagnostics logged |
| `verl/workers/config/actor.py`, `verl/trainer/config/actor/actor.yaml` | New `self_distillation.peer_control_mode` / `peer_control_match` with validation |
| `run_local_sdpo.sh` | `PEER_CONTROL_MODE` / `PEER_CONTROL_MATCH` env plumbing; when active, pins the 2S1F context so the baseline row and controls share one context; run tag distinguishes the four rows |

The offline **teacher-signal** analysis gets matching controls plus statistics:

| File | Change |
|---|---|
| `analysis/build_context_variants.py` | New `random_peer` and `verifier_score_ordered` (top-N by reward) control variants, matched to the 2S1F slot budget |
| `analysis/compute_teacher_signal_metrics.py` | Prompt-level bootstrap 95% CIs; a "both-present" sample set (prompts with ≥1 success *and* ≥1 failure, where AUC/pairwise-accuracy are well-defined); a Table-7 markdown/CSV export (`--num-bootstrap`, `--table-sample-set`) |
| `analysis/plot_teacher_metrics.py` | CI error bars, extra palette colors, and the two new conditions / sample sets |
| `analysis/run_ana.sh` | Wires the new conditions, bootstrap CIs, and the both-present plot into the pipeline |

### 3. vLLM 0.8.x compatibility fix

`verl/workers/rollout/vllm_rollout/vllm_rollout.py` — `_load_model` now falls back to
`execute_method("load_model", …)` when the engine does not expose a callable
`load_model` attribute (vLLM 0.8.x keeps it on the wrapped worker), so rollout startup
works across vLLM versions.

---

## 🔬 Teacher Signal Analysis

The `analysis/` directory contains scripts to reproduce the self-teacher signal quality evaluation from Section 5.3 of the paper.

**Step 1 — Prepare rollout dataset from a training run:**
```bash
python analysis/prepare_rollout_dataset.py \
  --input ./rollouts/<exp_name> \
  --output-dir analysis_outputs/lcbv6_rollout_dataset \
  --max-prompts 300 \
  --max-responses-per-prompt 8
```

**Step 2 — Build context variants:**
```bash
python analysis/build_context_variants.py \
  --candidates analysis_outputs/lcbv6_rollout_dataset/candidate_responses.jsonl \
  --evidence  analysis_outputs/lcbv6_rollout_dataset/evidence_items.jsonl \
  --output    analysis_outputs/lcbv6_context_variants.jsonl
```

**Step 3 — Score teacher contexts:**
```bash
python analysis/score_teacher_contexts.py \
  --variants  analysis_outputs/lcbv6_context_variants.jsonl \
  --targets   analysis_outputs/lcbv6_rollout_dataset/candidate_responses.jsonl \
  --output    analysis_outputs/lcbv6_teacher_scores.jsonl \
  --model     Qwen/Qwen3-4B \
  --self-target-only \
  --condition-filter base solution another_solution failure_solution \
                     solution+failure_solution solution+another_solution+failure_solution all_solutions \
  --max-model-len 8192
```

**Step 4 — Compute metrics and plot:**
```bash
python analysis/compute_teacher_signal_metrics.py \
  --input  analysis_outputs/lcbv6_teacher_scores.jsonl \
  --output analysis_outputs/lcbv6_teacher_metrics.json

python analysis/plot_teacher_metrics.py \
  --input      analysis_outputs/lcbv6_teacher_metrics.json \
  --output-dir analysis_outputs/plots \
  --sample-set effective_only
```

### Solution Diversity Analysis

```bash
# After generating rollouts from MOPD and SDPO checkpoints:
SDPO_CANDIDATES=./rollouts/sdpo_step50/candidate_responses.jsonl \
MOPD_CANDIDATES=./rollouts/mopd_step50/candidate_responses.jsonl \
BASE_CANDIDATES=./rollouts/base/candidate_responses.jsonl \
bash diversity/run_diversity.sh
```

---

## Attribution

Our implementation builds on prior open-source on-policy distillation and RL training frameworks. Specific upstream projects will be acknowledged in a future release.
