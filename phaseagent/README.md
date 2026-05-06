# PhaseAgent-lite

PhaseAgent-lite estimates functional phase boundaries in protein fitness landscapes from public DMS data. It treats mutation distance as a design radius and estimates the critical radius at which functional viability collapses. It also simulates an active boundary-discovery agent and a phase-aware search policy.

All compute runs on **Modal** — a persistent volume (`phaseagent-data`) holds raw DMS CSVs, the processed parquet, result tables, and figures. Real ProteinGym v1.3 substitution data is the only input; there is no synthetic-data fallback.

## Project status (May 2026)

The active scientific story is **EditGuard** — a DMS-conditioned protein
editing benchmark and guidance framework. The PhaseAgent boundary atlas
described above remains in the codebase but is not the current submission;
its neural-Spectral-Transformer and structure-aware GNN scaffolds have been
moved to `src/phaseagent/_experimental/` and are not wired into any
EditGuard claim. See `RUNBOOK.md` (forthcoming) for the EditGuard pipeline
and `/Users/vincentyip/.claude/plans/sparkling-watching-beaver.md` for the
implementation plan.

## Setup

```bash
pip install -e .
modal token new        # one-time, if you haven't authenticated
```

## Main commands (Modal)

End-to-end:
```bash
modal run modal_app.py
```
Pulls the ProteinGym v1.3 substitution archive (~1 GB) into the `phaseagent-data` volume, builds the canonical parquet, fits phase boundaries, runs the active-query simulation, runs the phase-aware search benchmark, scores PLM naturalness on an A10G GPU, and renders all figures.

Step by step:
```bash
modal run modal_app.py::download_proteingym
modal run modal_app.py::build_dataset --n-datasets 10        # smoke first; drop the flag for full
modal run modal_app.py::run_phase_atlas
modal run modal_app.py::run_phaseagent
modal run modal_app.py::run_phase_aware_search
modal run modal_app.py::run_plm_scoring                      # GPU step (A10G)
modal run modal_app.py::make_figures
```

Pull outputs back locally:
```bash
modal volume get phaseagent-data outputs ./outputs
```

## Local commands (no Modal)

The same logic is exposed as plain Python scripts — useful for unit tests and ad-hoc inspection of a small subset:

```bash
python scripts/download_proteingym.py --out-dir data/raw/proteingym_v1_3
python scripts/build_dataset.py --raw-dir data/raw/proteingym_v1_3 --out data/processed/all_dms.parquet --n-datasets 5
python scripts/run_phase_atlas.py --data data/processed/all_dms.parquet --out outputs/tables/phase_boundaries.csv
python scripts/run_phaseagent.py --data data/processed/all_dms.parquet --out outputs/tables/phaseagent_simulation.csv
python scripts/run_phase_aware_search.py --data data/processed/all_dms.parquet --boundaries outputs/tables/phase_boundaries.csv --out outputs/tables/search_benchmark.csv
python scripts/run_plm_scoring.py --data data/processed/all_dms.parquet --out outputs/tables/plm_scores.csv  # needs torch + fair-esm + GPU
python scripts/make_all_figures.py --tables outputs/tables --out outputs/figures --data data/processed/all_dms.parquet
```

## Tests

```bash
pytest -q
```

Tests use deterministic inline V(d) tables and small DataFrames as fixtures — no synthetic landscapes leak into the data pipeline.

## Layout

```
phaseagent/
  modal_app.py              Modal app: CPU image + GPU image, persistent volume
  configs/
    datasets.yaml           Manifest (used only if you bypass ProteinGym)
    default.yaml            Pipeline hyperparameters
  src/phaseagent/
    io.py                   Column inference + DMS CSV loader
    mutations.py            Mutation notation parsing + Hamming distance
    preprocessing.py        Canonical schema (mutation_distance, fitness_norm, viable)
    proteingym.py           ProteinGym substitution archive ingestion
    thresholds.py           Quantile / absolute viability thresholds
    phase.py                V(d), sigmoid fit, susceptibility, bootstrap CI, regime classifier
    metrics.py              Additive / ruggedness epistasis proxies + search eval
    agents.py               Query policies + simulator (Random / UniformShell / UncertaintyShell / BoundaryGreedy / PhaseAgent)
    search.py               Random / novelty / fitness-proxy / phase-aware search baselines
    spectrum.py             Single-mutant spectrum tokens and summary features
    large_deviation.py      Additive Monte Carlo / FFT / large-deviation survival baselines
    survival_model.py       Optional PyTorch survival-curve and residual models
    flow_posterior.py       Bootstrap/Gaussian phase posterior interface
    active_phaseagent.py    Posterior-aware active querying policies
    survival_search.py      Survival-aware search frontier utilities
    structure.py            Optional PDB/contact-map helpers
    equivariant_model.py    Optional invariant structure-aware neural model
    eval_survival.py        Survival-curve metrics
    plots_advanced.py       Advanced spectral/LD/frontier figures
    plm.py                  ESM-2 mean-log-prob naturalness scoring (GPU)
    plots.py                All figure generators + make_all_figures()
  scripts/                  Local CLI mirrors of the Modal entrypoints
  tests/                    Mutation, phase-fit, and policy unit tests
  outputs/{tables,figures,logs}/   Result CSVs and figures (mirrored on Modal volume)
```

## Outputs

After `make_figures` you'll have:

| Figure | File | Notes |
| --- | --- | --- |
| 1 | `fig1_phase_transition_curves.{png,pdf}` | V(d) curves for first 6 datasets |
| 2 | `fig2_phase_atlas_heatmap.{png,pdf}` | Datasets × distance, sorted by `dc` |
| 3 | `fig3_boundary_error_vs_queries.{png,pdf}` | Active-query comparison |
| 4 | `fig4_epistasis_vs_sharpness.{png,pdf}` | Ruggedness proxy vs `alpha` |
| 5 | `fig5_phase_aware_frontier.{png,pdf}` | Hit rate vs novelty |
| 6 | `fig6_dc_alpha_distribution.{png,pdf}` | `dc`/`alpha` histograms |
| 7 | `fig7_boundary_taxonomy_examples.{png,pdf}` | One example per regime |
| 8 | `fig8_failure_examples.{png,pdf}` | Variants beyond `dc` with low fitness |
| 9 | `fig9_plm_vs_fitness.{png,pdf}` | ESM-2 score vs DMS score (GPU step) |

Tables (`outputs/tables/`):
- `dataset_summary.csv` — per-assay row counts and ingestion status
- `phase_boundaries.csv` — `dc`, `alpha`, `r2`, regime per dataset
- `viability_by_distance.csv` — V(d) per dataset
- `bootstrap_boundaries.csv` — bootstrap CI samples
- `phaseagent_simulation.csv` — query-efficiency rows
- `search_benchmark.csv` — search policy metrics
- `plm_scores.csv` — ESM-2 naturalness scores

## Cost discipline

- A10G GPU (~$1.10/hr on Modal) is used only for `run_plm_scoring`.
- The full pipeline on the v1.3 substitution benchmark costs roughly **$1–3** end-to-end (download + CPU work + a few minutes of A10G).
- Always smoke-test with `--n-datasets 5` first.
