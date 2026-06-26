# Handoff: Learned Epistasis for Protein Function — BioML reframing

Date: 2026-06-26. Branch: `claude/peaceful-mirzakhani-da5230`. App: Modal `phaseagent-v2`, volume `phaseagent-data` (`/data`).

## TL;DR (what this project now is)

We dropped the "PhaseSMC / phase-transition / survival" framing (reviewers gave 7/6 but *both at confidence 2, explicitly "did not understand central parts"* — i.e. accepted because intimidating, not understood). The work is now stated in field-standard BioML language and centered on a single, defensible claim:

> **Multi-mutant protein fitness decomposes as `additive (sum of singles)` + `global epistasis (a monotonic saturation link)` + `specific epistasis (residue-pair interaction)`. Specific epistasis in *function* (binding, fluorescence) is real, is invisible to zero-shot protein language models, and is *predictable* by a trained model from DPLM representations — enabling design of gain-of-function combinations additive models structurally cannot find.**

This differentiates us from (a) the zero-shot crowd (PLMs-capture-epistasis, bioRxiv Sept 2025) and (b) the stability-proxy crowd (DPLM-Evo, ICML 2026, whose "GFP directed evolution" only optimizes pTM/structure, never measured function).

## The three-layer decomposition (the spine)

On `ddG_ML` (Megascale stability) and `DMS_score` (ProteinGym function), per protein:

- **Additive** = sum of single-mutant effects. Often *fails* (median R² −0.91 on Megascale doubles; negative on several function assays) because of...
- **Global epistasis** = a monotonic saturation link (assay floor / brightness threshold). A per-protein isotonic link recovers most of it (Megascale −0.91 → **+0.72**; GFP → 0.89; GB1-Olson 0.70 → 0.94; GRB2 0.43 → 0.85; PABP −5.5 → 0.81). This is the layer the **old large-deviation theory predicts in closed form** — keep it as the theory contribution.
- **Specific epistasis** = the residual after both. The DeWitt-audit target: any held-out skill on it is genuine epistasis by construction. Megascale 0.42 kcal/mol (28% of double-mutant variance); function 6–45% (GB1-Wu's 4 coupled positions = **45% specific**, the strongest case).

Code: `src/phaseagent/epistasis_decomposition.py` (`decompose_multimutants`, `add_global_specific_layers`, `protein_epistasis_atlas`, `epistasis_gate_summary`). Tested: `tests/test_epistasis_decomposition.py`.

## Key results (all held-out = real)

### Zero-shot DPLM: works on stability, FAILS on function
DPLM pseudo-log-likelihood epistasis `LL(double) − LL(s1) − LL(s2) + LL(wt)` vs measured specific epistasis:
- Stability (Megascale, 89,706 doubles / 149 proteins): Spearman ≈ **0.25** (scale-invariant — 30 vs 150 proteins identical). Capped; contact-flat. PLM "naturalness" tracks stability-coupled epistasis.
- Function (GB1 binding): **0.02**. (GFP: 0.005.) PLM naturalness knows nothing about assay-specific function.

### Model 1 (prediction) — a trained coupling head BEATS zero-shot on function
`scripts/train_model1.py`: bilinear coupling head (`proj_i ⊙ proj_j`) on DPLM last-layer hidden states at the two mutated positions + AA-identity embeddings + zero-shot PLL as a residual base. Held-out evaluation:

| | GB1 binding | GFP fluorescence |
|---|---|---|
| zero-shot DPLM | 0.02 | 0.00 |
| Model 1, held-out **doubles** | **0.37** | 0.14 |
| Model 1, held-out **positions** | 0.10 → **0.13** (with rep-shifts) | 0.09 |

The model sees only DPLM representations + AA identities (NOT position indices), so the signal is genuinely the representation. Held-out-double = "complete a combinatorial landscape from partial DMS at known coupled positions" (strong, the real design setting). Held-out-position = cross-position transfer (hard, weak). **Representation-shift features** `H^(i→a)[j] − H^wt[j]` (`extract_function_shifts` → `train_model1.py --shift`) lift held-out-position 0.095 → 0.127. **Contact distance is a weak feature** even on GB1's real fold (Spearman(|eps|, Cα dist) = −0.08) — honest negative, not added.

### Model 2 (design) — recovers gain-of-function additive misses
`scripts/model2_design.py` (fig9): among additively-mediocre GB1 doubles (mean binding −3.62, additive rates them equal), ranking by Model 1's predicted epistasis recovers gain-of-function (top-10% → −2.50, ~70% of oracle −2.00). **Matched-additive control: within-bin Spearman(pred_eps, measured) = 0.28** (peaks ~0.48 mid-bins) = genuine epistasis additive cannot see (DeWitt-proof).

### End-to-end LoRA fine-tune (cross-protein generalization) — IN PROGRESS
`train_epistasis_e2e` (modal_app_v2.py): LoRA fine-tune of DPLM-650M (`target_modules=["query","value"]`) + pairwise head, trained across ~120 proteins, evaluated on **25 held-out proteins**. Smoke (1 epoch, 3k doubles): held-out-protein median ρ ≈ **0.44** (noisy, ~5 proteins). H100 run (r=64, batch 128, 10 epochs, 120k doubles, 12.9M trainable params) training cleanly (Huber 0.285→0.215→0.182 over 3 epochs) — **final held-out-protein number pending** (was getting killed because launched without `--detach`; relaunch with `modal run --detach`). Generative Model 2 (`train_generative_design`, function-weighted LoRA fine-tune that *generates* high-function combos, eval on GB1-Wu's complete combinatorial) is built + smoke-tested, full run pending.

## Figures (`paper/figures_epistasis/`, all BioML language)
fig1 additivity_saturation · fig2 three_layer_r2 · fig3 specific_epistasis · fig4 headroom_dplm (stability) · fig5 function_decomposition · fig6 gb1_gain_of_function · **fig7a gb1_structure** (PyMOL, G41×V54 in 3D contact, Cα 5.5Å) · fig7 gb1_main (composite) · fig8 model1_function (Model 1 beats zero-shot) · fig9 model2_design (gain-of-function recovery + matched-additive control). Built by `scripts/build_{epistasis,function,gb1_composite,model1}_*.py`. Structure render: `/applications/PyMOL.app/Contents/MacOS/PyMOL -cq <pml>`.

## Data sources (where everything comes from)

- **Megascale stability (Tsuboyama et al., Nature 2023):** HuggingFace `RosettaCommons/MegaScale`. Configs: `dataset3_single` (1.84M singles, ThermoMPNN split), `dataset2` (776k incl. ~210k curated doubles, 559 site pairs / 331 natural + 148 de novo domains), `AlphaFold_model_PDBs` (bundled structures). Loaded by `src/phaseagent/megascale.py::load_megascale(MegascaleConfig(config_name="dataset2"))`; cached on Modal volume `phaseagent-data` at `data/megascale/`. Canonical label `ddG_ML` (kcal/mol, >0 = stabilizing).
- **ProteinGym v1.3 substitutions (Marks lab):** on the Modal volume at `raw/proteingym_v1_3/DMS_ProteinGym_substitutions/` (217 assay CSVs; columns `mutant`, `mutated_sequence`, `DMS_score`). Function multi-mutant assays used: `SPG1_STRSG_Olson_2014.csv` (GB1 IgG-binding, complete pairwise), `SPG1_STRSG_Wu_2016.csv` (GB1 4-site combinatorial, ~149k of 160k combos — every generated combo measurable), `GFP_AEQVI_Sarkisyan_2016.csv` (fluorescence, up to 15 mutations), `GRB2_HUMAN_Faure_2021.csv` and `PABP_YEAST_Melamed_2013.csv` (binding). Re-download the v1.3 substitution archive from ProteinGym (Marks lab) if the volume is wiped.
- **Structures:** GB1 = PDB **1PGA** (`https://files.rcsb.org/download/1PGA.pdb`); GB1 assay→domain offset is `full_length − 226` (assay 265/266/267/280 = domain V39/D40/G41/V54). Megascale AlphaFold PDBs come from the HF `AlphaFold_model_PDBs` config, extracted to `data/megascale/pdbs/` via `modal run modal_app_v2.py::extract_megascale_pdbs`.
- **Model weights:** DPLM-650M = HuggingFace `airkingbd/dplm_650m` (downloaded on first use; cached in the image/volume).
- **All training/eval is on real data only — no synthetic fallbacks.** Pull result artifacts back with `modal volume get phaseagent-data outputs/epistasis/<file> /tmp/`.

## How to run
```bash
# decomposition / atlas (CPU)
modal run modal_app_v2.py::run_epistasis_atlas
# zero-shot headroom + DPLM hidden-state features for Model 1 (GPU A10G)
modal run modal_app_v2.py::extract_function_features --assay-csv <proteingym csv> --assay-name GB1_Olson
modal run modal_app_v2.py::extract_function_shifts   --assay-csv <csv> --assay-name GB1_Olson
modal volume get phaseagent-data outputs/epistasis/feat_GB1_Olson.npz /tmp/ && \
  python scripts/train_model1.py --feat /tmp/feat_GB1_Olson.npz --shift /tmp/shift_GB1_Olson.npz
# end-to-end cross-protein fine-tune (USE --detach; H100 if billing enabled, else A10G)
modal run --detach modal_app_v2.py::train_epistasis_e2e --epochs 10 --lora-r 64 --batch-size 128
modal run --detach modal_app_v2.py::train_generative_design --epochs 5
```

## Gotchas
- **Always `modal run --detach`** for training — non-detached runs die when the local client is cleaned up (this killed two runs).
- H100 is **billing-gated** on this Modal account ("add a payment method"); A10G always works. Budget set to $500, H100 now unlocked — set `gpu="H100"` in the decorator.
- peft must be pinned `peft==0.10.0` (newer breaks on the pinned `transformers==4.39.2`).
- Per-protein target standardization must run **before** subsampling and use `ddof=0` (else NaN loss from 1-double proteins).
- Megascale `aa_seq` is each variant's OWN sequence (a double row's aa_seq = the double-mutant seq). GB1 assay positions are full-length; GB1-domain (1PGA) = full_length − 226.
- DPLM = `airkingbd/dplm_650m`, an ESM-2-style `EsmForMaskedLM` (self-attention only, NO cross-attention). Residue position `p` (1-indexed) is at token index `p` (CLS at 0).

## Next steps — Science / Nature / ICML level

**ICML / NeurIPS workshop (submittable soon):** finish the cross-protein e2e number + the generative Model 2 number; write the paper around "specific epistasis in function is PLM-representation-predictable but zero-shot-invisible," with Model 1 (predict), Model 2 (design), and the decomposition + theory (global epistasis) as the spine. This is honest and complete.

**Nature Methods / Nature Communications (the real swing):**
1. **Cross-protein epistasis foundation model.** Scale the e2e LoRA fine-tune to ALL multi-mutant data (Megascale ~150 proteins + every ProteinGym multi-mutant function assay), held-out-protein. If it predicts epistasis on unseen proteins ≫ zero-shot, that's a generalizable epistasis predictor the field lacks. Needs H100/multi-GPU (budget now allows).
2. **Cross-domain transfer:** train on binding, predict fluorescence (and vice versa) — does the coupling head learn protein-physics-general epistasis or assay-specific rules?
3. **Generative design, validated:** the function-weighted DPLM fine-tune (`train_generative_design`) generating gain-of-function multi-mutants that beat additive at matched edit budget; on GB1's complete combinatorial every generated combo is measurable, so this is a clean closed-loop result. Then a **wet-lab collaboration** (even 10–20 designed variants measured) is the single biggest lever from "strong ML paper" to Nature main.
4. **The theory, rehabilitated:** the closed-form spectrum→global-epistasis link (large-deviation rate function of the single-mutant spectrum) is a real, unique theoretical contribution — validate it predicts the empirical global-epistasis curve label-free across proteins.

**Science-level only with wet validation:** a closed-loop "predict → design → measure → improve" engine on a real therapeutic target (antibody affinity, enzyme activity), where the epistasis-aware designer beats directed evolution. Dry-lab alone tops out at Nature Methods/Comms; the experimental loop is the multiplier.
