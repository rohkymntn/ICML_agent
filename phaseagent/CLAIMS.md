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

### Decomposition bootstrap CIs (Section 7.1) — GROUNDED

Source: `outputs/epistasis/decomposition_ci.json` — 2,000-resample percentile
bootstrap over the 149 per-protein rows of the committed atlas
(`scripts/bootstrap_decomposition_ci.py`, seed 0; each resample recomputes the
exact paper statistic `build_decomposition_artifact.summarize`, so point estimates
equal the committed `decomposition_summary.json`). Guarded by
`tests/test_decomposition_ci.py` (no-drift + the three significance CIs exclude
their null).

| Claim in paper | Value | Artifact key / column |
|---|---|---|
| Median additive $R^2$ 95% CI (entirely $<0$) | $[-1.39, -0.58]$ | `decomposition_ci.json:median_r2_additive_ci95` |
| Median +global $R^2$ 95% CI (entirely $>0$) | $[0.67, 0.74]$ | `decomposition_ci.json:median_r2_global_ci95` |
| Median specific std 95% CI (kcal/mol) | $[0.39, 0.45]$ | `decomposition_ci.json:median_spec_std_kcal_ci95` |
| Median specific variance share 95% CI | $[0.26, 0.33]$ | `decomposition_ci.json:median_spec_var_share_ci95` |

The grey "measurement-scatter" band in `fig3_specific_epistasis.pdf` (`axvspan(0.1, 0.3)`)
is an **illustrative reference only**, not a committed result — the §7.1 body text makes no
numeric noise claim (iter17 removed the earlier unbacked "$\sim 0.1$ kcal/mol / four times"
phrasing). The "not noise" conclusion rests on the committed variance-share rows above
(median specific share 28%, $\geq 10\%$ in 98% of proteins).

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
| GB1 Model 1 doubles gain over zero-shot | $+0.34$ | `model1_GB1_Olson_summary.json:delta_doubles` |
| GFP Model 1 doubles gain over zero-shot | $+0.13$ | `model1_GFP_summary.json:delta_doubles` |

### Bootstrap 95% CIs (Section 7.3 significance sentence) — GROUNDED

Source: `outputs/epistasis/model1_ci.json` — 2,000-resample percentile bootstrap over
the held-out doubles of the same committed `model1_oof_{GB1_Olson,GFP}.csv`
(`scripts/bootstrap_model1_ci.py`, seed 0; point estimates equal the committed
summaries, and the per-resample Model 1 − zero-shot difference is paired).
Guarded by `tests/test_model1_ci.py` (no-drift of point estimates + delta CI excludes 0
+ the held-out-position CIs exclude 0).

| Claim in paper | Value | Artifact key / column |
|---|---|---|
| GB1 Model 1 doubles Spearman 95% CI | $[0.34, 0.38]$ | `model1_ci.json:assays.GB1_Olson.model1_doubles_ci95` |
| GFP Model 1 doubles Spearman 95% CI | $[0.12, 0.16]$ | `model1_ci.json:assays.GFP.model1_doubles_ci95` |
| GB1 zero-shot Spearman 95% CI | $[0.00, 0.04]$ | `model1_ci.json:assays.GB1_Olson.zeroshot_ci95` |
| GFP zero-shot Spearman 95% CI (includes 0) | $[-0.01, 0.02]$ | `model1_ci.json:assays.GFP.zeroshot_ci95` |
| GB1 gain-over-zero-shot 95% CI | $[0.31, 0.36]$ | `model1_ci.json:assays.GB1_Olson.delta_ci95` |
| GFP gain-over-zero-shot 95% CI | $[0.11, 0.16]$ | `model1_ci.json:assays.GFP.delta_ci95` |
| GB1 held-out-position Spearman 95% CI (excludes 0) | $[0.10, 0.14]$ | `model1_ci.json:assays.GB1_Olson.model1_position_ci95` |
| GFP held-out-position Spearman 95% CI (excludes 0) | $[0.06, 0.10]$ | `model1_ci.json:assays.GFP.model1_position_ci95` |

## Cross-protein generalization, e2e LoRA fine-tune (Section 7.5 / Table "e2e") — GROUNDED

Source: `outputs/epistasis/e2e_results.json` (end-to-end LoRA fine-tune of DPLM-650M
+ pairwise epistasis head over Megascale STABILITY doubles, trained on 124 proteins
and evaluated on the 25 held-out proteins that contribute no training examples; run
`ap-ayiJz1iYjfF0M5qUt0Rhaj`). Guarded by `tests/test_e2e_artifact.py`.

| Claim in paper | Value | Artifact key / column |
|---|---|---|
| Held-out proteins (no training examples) | 25 | `e2e_results.json:n_holdout_proteins` |
| Training proteins | 124 | `e2e_results.json:n_proteins` − `n_holdout_proteins` |
| LoRA rank | 64 | `e2e_results.json:lora_r` |
| Epochs | 10 | `e2e_results.json:epochs` |
| Trainable params (M) | 12.9 | `e2e_results.json:trainable_params_M` |
| Median per-held-out-protein Spearman | $0.33$ | `e2e_results.json:heldout_protein_median_spearman` |
| Mean per-held-out-protein Spearman | $0.33$ | `e2e_results.json:heldout_protein_mean_spearman` |
| Fraction of held-out proteins with $\rho>0.2$ | 0.80 | `e2e_results.json:frac_heldout_prot_gt_0p2` |
| Zero-shot DPLM PLL reference (corpus-level, §7.2) | $0.25$ | `headroom_summary.json:median_per_protein_spearman` (0.250987 → 0.25, the real measured zero-shot ceiling on 30 Megascale stability domains; **the paper traces the 0.25 to this measured artifact, NOT** to `e2e_results.json:zero_shot_reference`, which is a hardcoded literal `0.25` in `train_epistasis_e2e` that merely coincides with it). This is a corpus-level zero-shot ceiling on the same specific-epistasis target/metric/regime, **not** a per-test-protein zero-shot control re-measured on the exact 25 held-out domains (left as future work, §limitations). |

## Model 2 design + best-of-N rerank baseline (Section 7.4 / Table "model2", Fig "model2") — GROUNDED

Source: `outputs/epistasis/model2_oof_GB1.csv` (raw per-double table `glob, measured,
true_eps, pred_eps, pll_eps` from a 5-fold OOF retrain of Model 1 on the complete
GB1-Olson pairwise assay) + `model2_GB1_summary.json` (the paper numbers, a bit-exact
pure function of the committed CSV via `summarize_model2`). The design pool is the
additively-mediocre subset (below median `glob`); each rule selects its top 10%.
Guarded by `tests/test_model2_committed.py` (reproduces from CSV; asserts the Model 1
lift and matched-additive Spearman are positive).

| Claim in paper | Value | Artifact key / column |
|---|---|---|
| Held-out GB1 doubles | 12,000 | `model2_GB1_summary.json:n_doubles` |
| Additive pool mean binding | $-3.62$ | `model2_GB1_summary.json:pool_mean_binding` |
| Model 1 top-10% binding | $-2.53$ | `model2_GB1_summary.json:gof_top10pct_model1` |
| Best-of-$N$ rerank (zero-shot DPLM PLL) top-10% binding | $-3.84$ | `model2_GB1_summary.json:gof_top10pct_zeroshot` |
| Oracle top-10% binding | $-2.00$ | `model2_GB1_summary.json:gof_top10pct_oracle` |
| Model 1 lift over additive pool | $+1.09$ | `model2_GB1_summary.json:gof_lift_model1` |
| Best-of-$N$ rerank lift over additive pool | $-0.22$ | `model2_GB1_summary.json:gof_lift_zeroshot` |
| Oracle lift over additive pool | $+1.62$ | `gof_top10pct_oracle` − `pool_mean_binding` |
| Matched-additive within-bin Spearman | $0.28$ | `model2_GB1_summary.json:matched_additive_spearman` |
| Matched-additive bins | 10 | `model2_GB1_summary.json:n_matched_bins` |

### Model 2 design 95% CIs (Section 7.4, bootstrap) — GROUNDED

Source: `outputs/epistasis/model2_ci.json` — 2,000-resample percentile bootstrap over the
12,000 held-out GB1 doubles of `model2_oof_GB1.csv` (`scripts/bootstrap_model2_ci.py`,
seed 0), recomputing the EXACT paper statistic (`summarize_model2`) on each resample so
the matched-additive median-over-deciles and the top-10%-of-pool lifts are bootstrapped
self-consistently; the Model 1 − zero-shot lift difference is paired (same resample).
Point estimates equal the committed `model2_GB1_summary.json` (no drift, asserted in
`run()`). Guarded by `tests/test_model2_ci.py` (no-drift + the three significance CIs
exclude 0).

| Claim in paper | Value | Artifact key / column |
|---|---|---|
| Matched-additive control Spearman 95% CI (excludes 0) | $[0.24, 0.31]$ | `model2_ci.json:matched_additive_ci95` |
| Model 1 design lift 95% CI (excludes 0) | $[0.99, 1.20]$ | `model2_ci.json:gof_lift_model1_ci95` |
| Model 1 − zero-shot rerank lift | $+1.31$ | `model2_ci.json:gof_lift_delta_model1_minus_zeroshot` |
| Model 1 − zero-shot rerank lift 95% CI (excludes 0) | $[1.17, 1.44]$ | `model2_ci.json:gof_lift_delta_ci95` |

## Figures

Every committed figure under `paper/figures_epistasis/` is cited in the paper and
regenerates from a committed data artifact (no orphans; verified iter16).

| Figure | File | Regen script | Committed data source |
|---|---|---|---|
| fig2 three-layer $R^2$ | `paper/figures_epistasis/fig2_three_layer_r2.pdf` | `scripts/build_epistasis_figures.py --from-committed` | `three_layer_atlas.parquet` |
| fig3 specific epistasis | `paper/figures_epistasis/fig3_specific_epistasis.pdf` | `scripts/build_epistasis_figures.py --from-committed` | `three_layer_atlas.parquet` |
| fig4 zero-shot headroom | `paper/figures_epistasis/fig4_headroom_dplm.pdf` | `scripts/build_epistasis_figures.py --from-committed` | `outputs/epistasis/headroom_dplm.parquet` |
| fig8 Model 1 (predict) | `paper/figures_epistasis/fig8_model1_function.pdf` | `scripts/build_model1_figure.py` | `outputs/epistasis/model1_oof_{GB1_Olson,GFP}.csv` |
| fig9 Model 2 design | `paper/figures_epistasis/fig9_model2_design.pdf` | `scripts/model2_design.py --figure-only` | `model2_oof_GB1.csv` |

## PENDING — not yet in the paper body (need committed run artifacts)

These numbers from `HANDOFF_EPISTASIS_2026.md` are NOT yet committed as artifacts
under `outputs/epistasis/` and therefore do NOT appear as numbers in the paper.
They are folded in (with a row above) only when their artifact is committed.

- Function decomposition (GFP, GB1-Olson, GRB2, PABP global-link $R^2$) — needs committed function-assay decomposition table. (NOT cited in the paper; the function story rests on Model 1 / Model 2 / e2e, all grounded above.)
