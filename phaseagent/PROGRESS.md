# PROGRESS — Learned Epistasis for Protein Function (ICML, story C)

Status legend: ☐ not started · ◐ in progress / pending artifact · ☑ done & verified

## Definition of Done

1. ☐ `paper/epistasis_icml.tex` exists, compiles cleanly with `tectonic`, all standard sections, no placeholder figures / \fbox stubs / TODO markers in compiled body.
2. ◐ Headline cross-protein generalization number (e2e LoRA, ~25 held-out proteins) is a REAL number from a committed run artifact, in the paper. — e2e run launched (see RUNS.md); artifact pending.
3. ☐ Model 1 (held-out doubles AND held-out positions) + Model 2 (gain-of-function recovery + matched-additive control Spearman) numbers in paper, each traceable to a committed CSV/parquet under `outputs/epistasis/`.
4. ☐ Main-conference baseline set on the SAME tasks: zero-shot DPLM PLL, additive baseline, best-of-N rerank, one strong discrete-diffusion guidance baseline (+ DRAKES / SVDD / reward-guided refinement if a generative claim is made).
5. ☐ Every paper figure regenerates from committed data via a `scripts/` script; figure files committed; no orphan/degenerate figures.
6. ☐ If a calibration claim is retained, it uses test-time-valid methodology (temp scaling fit-on-val/test-on-test, or cross-fit isotonic).
7. ☐ `pytest -q` passes.
8. ☐ `CLAIMS.md` maps every quantitative claim → exact committed artifact file + column.
9. ☐ Limitations section honestly states what is not done.

When ALL true & verified: set this file to COMPLETE, commit, report `ICML_READY`.

## State of the world (iteration 1 audit)

- Story C code is committed: `src/phaseagent/epistasis_decomposition.py`, `scripts/train_model1.py`, `scripts/model2_design.py`, `modal_app_v2.py::{train_epistasis_e2e,train_generative_design}`.
- Figures (PNG/PDF) for fig1–fig9 are committed under `paper/figures_epistasis/`, BUT their underlying numeric artifacts are NOT committed and are NOT on the Modal volume (`outputs/epistasis/` does not exist on the volume). Only committed data artifact: `paper/figures_epistasis/three_layer_atlas.parquet`.
- No `paper/epistasis_icml.tex` yet.
- `tectonic` 0.16.9 installed locally (iter2) — DoD #1 can be compiled/verified once the .tex exists.
- The numbers quoted in `HANDOFF_EPISTASIS_2026.md` need to be RE-GENERATED into committed CSV/parquet artifacts under `outputs/epistasis/` to be paper-eligible (hard rule: every claim traces to a committed artifact).

## Log

- iter1 (2026-06-25): Audited repo. Created PROGRESS.md + RUNS.md. Launched detached headline e2e run (train_epistasis_e2e, 10 epochs, lora_r=64, batch 128) — the cross-protein held-out number (DoD #2). See RUNS.md.
- iter2 (2026-06-25): The iter1 e2e run had DIED (stopped, 0 tasks, no logs) — diagnosed as killed mid-build of the COLD `gpu_lora_image` at the iter1 client teardown. Fix: pre-warm the image cache with `h100_probe` (now builds in ~3 s), then RE-LAUNCHED the headline e2e run detached (`ap-ayiJz1iYjfF0M5qUt0Rhaj`); verified it loads data (`120000 doubles, 149 proteins; holdout 20073 / 25 prot`), trains, and survives local client kill. Installed `tectonic` 0.16.9 locally (DoD #1 prereq). RUNS.md updated. DoD #2 artifact still pending the run's completion.
