# Handoff: DMS-Conditioned Protein Diffusion

Date: 2026-06-26

## One-Sentence State

We pivoted the project from "PhaseSMC / survival / phase calibration" to a field-readable ICML story: **protein diffusion models should use target-specific experimental mutation maps as context for multi-mutant design**.

## Core Contribution Being Implemented

The new method is **DMS-conditioned protein diffusion**:

```text
wildtype protein sequence
+ single-mutant DMS map for that target protein
+ edit budget
-> frozen DPLM-650M backbone
+ trained DMS-context adapter
-> multi-mutant designs
```

The old PhaseSMC work should now be treated as an ablation or appendix theory, not the main paper language.

Use this language:

- DMS-conditioned generation
- experimental-context protein diffusion
- multi-mutant design
- calibrated edit risk
- same-task baselines
- held-out measured multi-mutants

Avoid leading with:

- phase
- survival
- cliffs
- Cramer / Bahadur-Rao
- private large-deviation jargon

## Files Added Or Changed

New implementation files:

- `src/phaseagent/mutation_map_context.py`
  - Defines `MutationMapContext`
  - Converts single-mutant DMS rows into a leakage-safe context tensor
  - Context shape is `[L, 20]` scores plus `[L, 20]` observation mask plus per-position features
  - Strict mode rejects multi-mutant rows to prevent target-label leakage

- `src/phaseagent/dms_context_adapter.py`
  - Defines `DMSContextAdapter`
  - Frozen-DPLM logit residual adapter
  - Inputs: DPLM hidden state, mutation-map context vector, edit budget
  - Outputs: 20 amino-acid logit correction
  - Loss: weighted masked-token NLL plus optional pairwise ranking loss

- `src/phaseagent/context_generation.py`
  - Open generation with DPLM plus DMS-context adapter
  - Same-task baseline evaluation
  - Baselines include:
    - `dplm_unguided`
    - `dplm_best_of_k`
    - `dplm_additive_context_rerank`
    - `dplm_dms_prior_rerank`
    - `dms_context_adapter`
    - `measured_pool_additive_oracle`
    - `context_adapter_pool_rerank`

- `tests/test_mutation_map_context.py`
  - Context tensor construction
  - Missing values
  - Leakage guard
  - Measured-pool additive oracle task constraints

- `tests/test_dms_context_adapter.py`
  - Variant parsing
  - Multi-mutant training-frame filtering
  - Adapter forward shape
  - Toy loss decreases under training

Updated files:

- `modal_app_v2.py`
  - Added `train_dms_context_adapter`
  - Added `run_dms_context_generation`
  - Added `build_context_generation_figures`
  - Added `run_context_generation_benchmark` marker entrypoint

- `src/phaseagent/embedding_baselines.py`
  - Rounded RF predictions to 12 decimals to remove last-bit nondeterminism in tests

New paper draft:

- `paper/editguard_context_2026.tex`
- `paper/editguard_context_2026.pdf`

The new paper draft compiles with `tectonic`.

## Modal Runs Completed

### 1. Smoke Adapter Training

Command:

```bash
modal run modal_app_v2.py::train_dms_context_adapter \
  --max-train-variants 256 \
  --epochs 1 \
  --batch-size 4 \
  --out-model outputs/context_generation/smoke_dms_context_adapter.pt \
  --out-trace outputs/context_generation/smoke_train_trace.csv
```

Result:

- Completed successfully on Modal A10G
- Mean epoch loss: `3.3130`

Local artifacts:

- `outputs/context_generation/smoke_dms_context_adapter.pt`
- `outputs/context_generation/smoke_train_trace.csv`

### 2. Smoke Generation Benchmark

Command:

```bash
modal run modal_app_v2.py::run_dms_context_generation \
  --adapter-path outputs/context_generation/smoke_dms_context_adapter.pt \
  --out-metrics outputs/context_generation/smoke_context_generation_metrics.csv \
  --out-selections outputs/context_generation/smoke_context_generation_selections.parquet \
  --max-tasks 2 \
  --seeds 1 \
  --n-mask-patterns 4 \
  --samples-per-pattern 4 \
  --top-k 10
```

Result:

- Completed successfully
- Exercised DPLM loading, context building, adapter generation, baselines, and output writing
- Open DPLM generation had very low measured DMS overlap, as expected for tiny sample size

Local artifacts:

- `outputs/context_generation/smoke_context_generation_metrics.csv`
- `outputs/context_generation/smoke_context_generation_selections.parquet`
- `outputs/context_generation/smoke_figures/`

### 3. Real 5k Adapter Training

Command:

```bash
modal run -d modal_app_v2.py::train_dms_context_adapter \
  --max-train-variants 5000 \
  --epochs 1 \
  --batch-size 8 \
  --out-model outputs/context_generation/dms_context_adapter_5k_e1.pt \
  --out-trace outputs/context_generation/dms_context_adapter_5k_e1_train_trace.csv
```

Result:

- Completed successfully on Modal A10G
- Mean epoch loss: `1.7900`
- This is the first real trained DMS-context adapter checkpoint

Local artifacts:

- `outputs/context_generation/dms_context_adapter_5k_e1.pt`
- `outputs/context_generation/dms_context_adapter_5k_e1_train_trace.csv`

### 4. Initial 10-Task Benchmark

Command:

```bash
modal run modal_app_v2.py::run_dms_context_generation \
  --adapter-path outputs/context_generation/dms_context_adapter_5k_e1.pt \
  --out-metrics outputs/context_generation/context_generation_5k_e1_metrics.csv \
  --out-selections outputs/context_generation/context_generation_5k_e1_selections.parquet \
  --max-tasks 10 \
  --seeds 1 \
  --n-mask-patterns 8 \
  --samples-per-pattern 8 \
  --top-k 25
```

Result:

- Completed successfully
- Found an important issue:
  - Open generation had low measured-library overlap
  - `dms_context_adapter` open samples had zero labeled overlap in this small run
- This is not yet a paper-ready open-generation result

Local artifacts:

- `outputs/context_generation/context_generation_5k_e1_metrics.csv`
- `outputs/context_generation/context_generation_5k_e1_selections.parquet`
- `outputs/context_generation/figures_5k_e1/`

### 5. Updated 10-Task Benchmark With Held-Out Pool Reranking

Added `context_adapter_pool_rerank` because open generation overlap is too sparse to evaluate the trained adapter cleanly.

Command:

```bash
modal run modal_app_v2.py::run_dms_context_generation \
  --adapter-path outputs/context_generation/dms_context_adapter_5k_e1.pt \
  --out-metrics outputs/context_generation/context_generation_5k_e1_metrics_v2.csv \
  --out-selections outputs/context_generation/context_generation_5k_e1_selections_v2.parquet \
  --max-tasks 10 \
  --seeds 1 \
  --n-mask-patterns 8 \
  --samples-per-pattern 8 \
  --top-k 25
```

Result:

- Completed successfully
- 70 metric rows
- 10 tasks
- Main useful first result:

```text
context_adapter_pool_rerank:
  mean functional hit rate: 0.904
  mean fitness: 0.911871
  mean best fitness: 0.995770
  mean edit distance: 2.392

measured_pool_additive_oracle:
  mean functional hit rate: 0.928
  mean fitness: 0.927709
  mean best fitness: 0.993355
  mean edit distance: 2.600
```

Interpretation:

- The trained adapter is real and competitive as a held-out measured-pool ranker.
- It does not yet beat the additive oracle/reference.
- Open generation still needs work because generated variants rarely overlap measured DMS libraries in this small benchmark.

Local artifacts:

- `outputs/context_generation/context_generation_5k_e1_metrics_v2.csv`
- `outputs/context_generation/context_generation_5k_e1_selections_v2.parquet`
- `outputs/context_generation/figures_5k_e1_v2/`

Figure copied into paper path:

- `paper/figures_context/fig3_context_generation_leaderboard.pdf`
- `paper/figures_context/fig3_context_generation_leaderboard.png`

## Test Status

Full local test suite:

```text
102 passed, 4 skipped
```

Command used:

```bash
pytest -q
```

Targeted new tests:

```text
7 passed
```

Command used:

```bash
pytest -q tests/test_mutation_map_context.py tests/test_dms_context_adapter.py
```

Paper compile:

```bash
tectonic paper/editguard_context_2026.tex
```

Result:

- PDF generated successfully
- Only formatting warnings

## Current Scientific Interpretation

This is not yet a finished ICML result.

What we have proven with code and real runs:

1. A single-mutant DMS map can be represented as safe experimental context.
2. A frozen DPLM-650M adapter can be trained on real multi-mutant measurements.
3. The trained adapter can rank held-out measured multi-mutants competitively.
4. The simple additive DMS reference is still very strong.
5. Open generation evaluation is currently bottlenecked by low measured-library overlap.

The paper should not claim that open DMS-conditioned generation beats baselines yet.

The honest current claim is:

> We implemented the first DMS-conditioned DPLM adapter and found that it learns useful held-out multi-mutant ranking signal, but additive single-mutant reranking remains a hard baseline and open generation requires better evaluation or stronger proposal mechanisms.

## Recommended Next Steps

### Highest Priority

Run a broader held-out pool-ranking benchmark:

```bash
modal run modal_app_v2.py::run_dms_context_generation \
  --adapter-path outputs/context_generation/dms_context_adapter_5k_e1.pt \
  --out-metrics outputs/context_generation/context_generation_5k_e1_metrics_broad.csv \
  --out-selections outputs/context_generation/context_generation_5k_e1_selections_broad.parquet \
  --max-tasks 0 \
  --seeds 3 \
  --n-mask-patterns 8 \
  --samples-per-pattern 8 \
  --top-k 25
```

This will be slower because `context_adapter_pool_rerank` scores measured pools with DPLM hidden states.

### Next Model Improvements

1. Train longer:

```bash
modal run modal_app_v2.py::train_dms_context_adapter \
  --max-train-variants 50000 \
  --epochs 2 \
  --batch-size 8 \
  --out-model outputs/context_generation/dms_context_adapter_50k_e2.pt \
  --out-trace outputs/context_generation/dms_context_adapter_50k_e2_train_trace.csv
```

2. Add explicit ablations:

- shuffled DMS context
- no DMS context
- additive-only logit correction
- adapter without pairwise ranking loss
- adapter with stronger pairwise weight

3. Improve open generation:

- generate from DMS-evaluable mutation tokens rather than all amino-acid substitutions
- increase samples per task
- add iterative denoising rather than one-shot masked LM
- use additive top positions as proposal masks
- report both open-generation and measured-pool ranking as separate tasks

## Important Caveats For The Next Agent

- Do not describe the current open-generation results as a success.
- The useful result is currently the held-out measured-pool ranking.
- The additive oracle/reference is still slightly stronger than the trained adapter on the 10-task v2 run.
- The benchmark currently overrepresents `F7YBW8_MESOW_Aakre_2015` because `max_tasks=10` takes the first task rows. A broad benchmark must cover more proteins.
- `PhaseSMC` should remain an ablation, not the main paper contribution.
- The new paper draft is intentionally a skeleton. It is not submission-ready, but the framing is much clearer than the old phase manuscript.

