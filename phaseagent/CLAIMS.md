# CLAIMS — every quantitative claim in the paper → its committed artifact

Each row maps a number that appears in `paper/epistasis_icml.tex` to the exact
committed file (and column / key) it is pulled from. A claim with no committed
artifact is NOT in the paper body; it is tracked as pending in `PROGRESS.md`.

## Decomposition (Section 6 / Table "decomp") — GROUNDED

Source: `outputs/epistasis/decomposition_summary.json` (derived from the committed
`paper/figures_epistasis/three_layer_atlas.parquet` by
`scripts/build_decomposition_artifact.py`; guarded by
`tests/test_decomposition_artifact.py`).

| Claim in paper | Value | Artifact key / column |
|---|---|---|
| Proteins | 149 | `decomposition_summary.json:n_proteins` |
| Double mutants | 131,062 | `decomposition_summary.json:total_double_mutants` |
| Median additive $R^2$ | $-0.91$ | `decomposition_summary.json:median_r2_additive` |
| Median additive+global $R^2$ | $0.72$ | `decomposition_summary.json:median_r2_global` |
| Proteins with additive $R^2<0$ | 72% | `decomposition_summary.json:frac_proteins_additive_r2_negative` |
| Median specific-epistasis std (kcal/mol) | $0.42$ | `decomposition_summary.json:median_spec_std_kcal` |
| Median specific variance share | 28% | `decomposition_summary.json:median_spec_var_share` |
| Proteins with specific share > 10% | 98% | `decomposition_summary.json:frac_proteins_spec_var_share_gt_0p1` |
| Proteins with specific std > 0.3 kcal/mol | 85% | `decomposition_summary.json:frac_proteins_spec_std_gt_0p3` |

## Figures

| Figure | File | Regen script | Committed data source |
|---|---|---|---|
| fig2 three-layer $R^2$ | `paper/figures_epistasis/fig2_three_layer_r2.pdf` | `scripts/build_epistasis_figures.py` | `three_layer_atlas.parquet` |
| fig3 specific epistasis | `paper/figures_epistasis/fig3_specific_epistasis.pdf` | `scripts/build_epistasis_figures.py` | `three_layer_atlas.parquet` |

## PENDING — not yet in the paper body (need committed run artifacts)

These numbers from `HANDOFF_EPISTASIS_2026.md` are NOT yet committed as artifacts
under `outputs/epistasis/` and therefore do NOT appear as numbers in the paper.
They are folded in (with a row above) only when their artifact is committed.

- Zero-shot DPLM PLL epistasis Spearman, STABILITY (Megascale doubles): **RUNNING** — `run_headroom_gate` (`ap-a7hQxHVfIOYVuoil7e6Go7`); will commit `outputs/epistasis/headroom_summary.json` (keys `overall_spearman`, `median_per_protein_spearman`, `frac_proteins_pp_spearman_gt_0p2`) + `headroom_dplm.parquet`. Folds into §7.2 (stability side of the zero-shot contrast) when committed.
- Zero-shot DPLM PLL epistasis Spearman, FUNCTION (GB1, GFP): pending the function feature/scoring pipeline (`extract_function_features` / function PLL scoring) — needs committed scoring CSV.
- Model 1 held-out doubles / held-out positions Spearman (GB1, GFP) — needs committed `train_model1.py` output.
- Model 2 gain-of-function recovery + matched-additive within-bin Spearman — needs committed `model2_design.py` output.
- e2e cross-protein median held-out-protein Spearman — needs committed `outputs/epistasis/e2e_results.json` (run `ap-ayiJz1iYjfF0M5qUt0Rhaj`).
- Function decomposition (GFP, GB1-Olson, GRB2, PABP global-link $R^2$) — needs committed function-assay decomposition table.
