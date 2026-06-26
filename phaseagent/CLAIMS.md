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

## Model 1 + function zero-shot (Section 7.2 / 7.3, Table "model1") — GROUNDED

Source: `outputs/epistasis/model1_oof_{GB1_Olson,GFP}.csv` (raw out-of-fold per-double
predictions for both the held-out-DOUBLES KFold split and the held-out-POSITION
GroupKFold split) + the matching `model1_{GB1_Olson,GFP}_summary.json` (Spearmans
derived purely from the CSV). Features are DPLM-650M hidden states + representation
shifts; no residue indices. Guarded by `tests/test_model1_committed.py` (asserts the
summary reproduces from the OOF CSV and Model 1 beats the zero-shot floor on doubles).

| Claim in paper | Value | Artifact key / column |
|---|---|---|
| GB1 zero-shot DPLM PLL Spearman (function floor) | $0.021$ | `model1_GB1_Olson_summary.json:zeroshot_spearman` |
| GFP zero-shot DPLM PLL Spearman (function floor) | $0.005$ | `model1_GFP_summary.json:zeroshot_spearman` |
| GB1 Model 1 held-out-doubles Spearman | $0.36$ | `model1_GB1_Olson_summary.json:model1_doubles_spearman` |
| GB1 Model 1 held-out-position Spearman | $0.12$ | `model1_GB1_Olson_summary.json:model1_position_spearman` |
| GFP Model 1 held-out-doubles Spearman | $0.14$ | `model1_GFP_summary.json:model1_doubles_spearman` |
| GFP Model 1 held-out-position Spearman | $0.08$ | `model1_GFP_summary.json:model1_position_spearman` |

## Cross-protein generalization, e2e LoRA fine-tune (Section 7.5 / Table "e2e") — GROUNDED

Source: `outputs/epistasis/e2e_results.json` (end-to-end LoRA fine-tune of DPLM-650M
+ pairwise epistasis head over Megascale STABILITY doubles, trained on 124 proteins
and evaluated on the 25 held-out proteins that contribute no training examples; run
`ap-ayiJz1iYjfF0M5qUt0Rhaj`). Guarded by `tests/test_e2e_artifact.py`.

| Claim in paper | Value | Artifact key / column |
|---|---|---|
| Held-out proteins (no training examples) | 25 | `e2e_results.json:n_holdout_proteins` |
| Training proteins | 124 | `e2e_results.json:n_proteins` − `n_holdout_proteins` |
| Trainable params (M) | 12.9 | `e2e_results.json:trainable_params_M` |
| Median per-held-out-protein Spearman | $0.33$ | `e2e_results.json:heldout_protein_median_spearman` |
| Mean per-held-out-protein Spearman | $0.33$ | `e2e_results.json:heldout_protein_mean_spearman` |
| Fraction of held-out proteins with $\rho>0.2$ | 0.80 | `e2e_results.json:frac_heldout_prot_gt_0p2` |
| Zero-shot DPLM PLL reference | $0.25$ | `e2e_results.json:zero_shot_reference` |

## Figures

| Figure | File | Regen script | Committed data source |
|---|---|---|---|
| fig2 three-layer $R^2$ | `paper/figures_epistasis/fig2_three_layer_r2.pdf` | `scripts/build_epistasis_figures.py` | `three_layer_atlas.parquet` |
| fig3 specific epistasis | `paper/figures_epistasis/fig3_specific_epistasis.pdf` | `scripts/build_epistasis_figures.py` | `three_layer_atlas.parquet` |

## PENDING — not yet in the paper body (need committed run artifacts)

These numbers from `HANDOFF_EPISTASIS_2026.md` are NOT yet committed as artifacts
under `outputs/epistasis/` and therefore do NOT appear as numbers in the paper.
They are folded in (with a row above) only when their artifact is committed.

- Model 2 gain-of-function recovery + matched-additive within-bin Spearman: **`feat_GB1_Olson.npz` DONE on volume (iter8); GB1 assay CSV pre-staged at `/tmp/SPG1_STRSG_Olson_2014.csv` (iter11).** **`model2_design.py` now writes the committed artifacts (iter12):** `model2_oof_GB1.csv` (raw per-double `glob, measured, true_eps, pred_eps`) + `model2_GB1_summary.json` (keys `pool_mean_binding`, `gof_top10pct_model1`, `gof_lift_model1`, `matched_additive_spearman`, `n_matched_bins`), derived purely from the CSV via `summarize_model2`. Guard `tests/test_model2_committed.py` in place (synthetic drift-guard passes; committed-artifact guard skips until landed). Pending only the CPU run-to-completion (`python scripts/train_model1.py`-style: `python scripts/model2_design.py --feat /tmp/feat_GB1_Olson.npz --assay-csv /tmp/SPG1_STRSG_Olson_2014.csv --assay GB1 --out outputs/epistasis --figure`, ~9 min) + commit, then fold into a paper Model 2 table + a row here.
- Function decomposition (GFP, GB1-Olson, GRB2, PABP global-link $R^2$) — needs committed function-assay decomposition table.
