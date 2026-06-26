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

## Zero-shot DPLM PLL baseline, STABILITY (Section 7.2 / Table "zeroshot") — GROUNDED

Source: `outputs/epistasis/headroom_summary.json` (per-double zero-shot DPLM PLL
epistasis `LL(dd)-LL(s1)-LL(s2)+LL(wt)` vs the measured specific-epistasis residual
on Megascale STABILITY doubles, run `ap-a7hQxHVfIOYVuoil7e6Go7`). Per-double values
in `outputs/epistasis/headroom_dplm.parquet` (columns `eps_specific`,
`dplm_epistasis`); the summary's per-protein stats reproduce from the parquet.

| Claim in paper | Value | Artifact key / column |
|---|---|---|
| Proteins scored | 30 | `headroom_summary.json:n_proteins` |
| Doubles scored | 15,000 | `headroom_summary.json:n_doubles_scored` |
| Median per-protein Spearman | $0.25$ | `headroom_summary.json:median_per_protein_spearman` |
| Overall Spearman | $0.23$ | `headroom_summary.json:overall_spearman` |
| Proteins with per-protein $\rho>0.2$ | 60% | `headroom_summary.json:frac_proteins_pp_spearman_gt_0p2` |

## Figures

| Figure | File | Regen script | Committed data source |
|---|---|---|---|
| fig2 three-layer $R^2$ | `paper/figures_epistasis/fig2_three_layer_r2.pdf` | `scripts/build_epistasis_figures.py` | `three_layer_atlas.parquet` |
| fig3 specific epistasis | `paper/figures_epistasis/fig3_specific_epistasis.pdf` | `scripts/build_epistasis_figures.py` | `three_layer_atlas.parquet` |

## PENDING — not yet in the paper body (need committed run artifacts)

These numbers from `HANDOFF_EPISTASIS_2026.md` are NOT yet committed as artifacts
under `outputs/epistasis/` and therefore do NOT appear as numbers in the paper.
They are folded in (with a row above) only when their artifact is committed.

- Zero-shot DPLM PLL epistasis Spearman, FUNCTION (GB1, GFP): **MEASURED iter8 directly from the landed `feat_*.npz` (`meta[:,0]` measured eps vs `meta[:,1]` PLL epistasis): GFP 0.005, GB1_Olson 0.021 — essentially zero, the other half of the §7.2 contrast (stability 0.25 already committed above).** NOT yet in the paper: the `.npz` is volume-only, so the number is paper-eligible only once `train_model1.py` writes it to the committed `model1_<assay>_summary.json:zeroshot_spearman` (the identical quantity, OOF-CSV-traceable).
- Model 1 held-out doubles / held-out positions Spearman (GB1, GFP): **all four `feat_/shift_{GB1_Olson,GFP}.npz` DONE on volume (iter8) — UNBLOCKED.** `train_model1.py` already writes `model1_oof_<assay>.csv` (per-double OOF for both the held-out-DOUBLES KFold and held-out-POSITION GroupKFold splits) + `model1_<assay>_summary.json` (the cited Spearmans). Guard `tests/test_model1_committed.py` in place (skips until committed). Pending only the CPU run-to-completion + commit.
- Model 2 gain-of-function recovery + matched-additive within-bin Spearman: **`feat_GB1_Olson.npz` DONE on volume (iter8); GB1 assay CSV pre-staged at `/tmp/SPG1_STRSG_Olson_2014.csv` (iter11).** **`model2_design.py` now writes the committed artifacts (iter12):** `model2_oof_GB1.csv` (raw per-double `glob, measured, true_eps, pred_eps`) + `model2_GB1_summary.json` (keys `pool_mean_binding`, `gof_top10pct_model1`, `gof_lift_model1`, `matched_additive_spearman`, `n_matched_bins`), derived purely from the CSV via `summarize_model2`. Guard `tests/test_model2_committed.py` in place (synthetic drift-guard passes; committed-artifact guard skips until landed). Pending only the CPU run-to-completion (`python scripts/train_model1.py`-style: `python scripts/model2_design.py --feat /tmp/feat_GB1_Olson.npz --assay-csv /tmp/SPG1_STRSG_Olson_2014.csv --assay GB1 --out outputs/epistasis --figure`, ~9 min) + commit, then fold into a paper Model 2 table + a row here.
- e2e cross-protein median held-out-protein Spearman — needs committed `outputs/epistasis/e2e_results.json` (run `ap-ayiJz1iYjfF0M5qUt0Rhaj`).
- Function decomposition (GFP, GB1-Olson, GRB2, PABP global-link $R^2$) — needs committed function-assay decomposition table.
