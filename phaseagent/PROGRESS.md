# PROGRESS — Learned Epistasis for Protein Function (ICML, story C)

Status legend: ☐ not started · ◐ in progress / pending artifact · ☑ done & verified

## Definition of Done

1. ◐ `paper/epistasis_icml.tex` exists, compiles cleanly with `tectonic` (verified, 5 pp, 0 undefined refs/cites, official `icml2025` style committed), all standard sections present, no \fbox / TODO / placeholder figures in compiled body. — PENDING: fold in Model 1/2, zero-shot, and e2e result numbers (results subsections currently forward-reference their committed artifacts; no fabricated numbers).
2. ◐ Headline cross-protein generalization number (e2e LoRA, ~25 held-out proteins) is a REAL number from a committed run artifact, in the paper. — e2e run RUNNING (see RUNS.md); artifact pending.
3. ◐ Model 1 (held-out doubles AND held-out positions) + Model 2 (gain-of-function recovery + matched-additive control Spearman) numbers, each traceable to committed CSV/parquet under `outputs/epistasis/`. — DECOMPOSITION part grounded (`outputs/epistasis/decomposition_{atlas.csv,summary.json}`, in paper Table); Model 1/2 numbers pending GPU artifacts.
4. ◐ Main-conference baseline set on the SAME tasks: zero-shot DPLM PLL, additive baseline, best-of-N rerank, one strong discrete-diffusion guidance baseline (+ DRAKES / SVDD / reward-guided refinement if a generative claim is made). — additive baseline defined in paper; **zero-shot DPLM PLL baseline LAUNCHED** (`run_headroom_gate`, `ap-a7hQxHVfIOYVuoil7e6Go7`, verified scoring on A10G — see RUNS.md), artifact `headroom_summary.json` pending; best-of-N rerank pending. (Current paper makes a RANKING design claim, not generative, so discrete-diffusion/DRAKES are scoped out unless a generative claim is added.)
5. ◐ Every paper figure regenerates from committed data via a `scripts/` script; figure files committed; no orphan/degenerate figures. — the 2 figures CURRENTLY cited (fig2, fig3) regenerate from the committed atlas via `scripts/build_epistasis_figures.py`; more figures added as their data commits.
6. ☐ If a calibration claim is retained, it uses test-time-valid methodology (temp scaling fit-on-val/test-on-test, or cross-fit isotonic). — no calibration claim currently in the paper.
7. ☑ `pytest -q` passes — 107 passed, 4 skipped (verified iter3).
8. ◐ `CLAIMS.md` maps every quantitative claim → exact committed artifact file + column. — created; decomposition rows grounded, pending rows enumerated.
9. ◐ Limitations section honestly states what is not done. — section written; revisit as scope finalizes.

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
- iter3 (2026-06-26): Stood up the paper spine + grounded the decomposition section. (1) Committed the official ICML `icml2025` style files into `paper/` (verified tectonic compiles the upstream example cleanly). (2) `scripts/build_decomposition_artifact.py` reduces the committed atlas → committed `outputs/epistasis/decomposition_atlas.csv` + `decomposition_summary.json` (149 proteins, 131,062 doubles; median additive R² −0.91 → +global 0.72; specific std 0.42 kcal/mol, 28% var share); guarded by `tests/test_decomposition_artifact.py`. (3) Wrote `paper/epistasis_icml.tex` (all standard sections) + `paper/epistasis.bib` (12 real refs) — compiles cleanly with `tectonic` (5 pp, 0 undefined refs/cites, no \fbox/TODO/placeholder figs; cites only fig2/fig3 which regenerate from committed data). (4) Created `CLAIMS.md`. (5) `pytest -q`: 107 passed, 4 skipped. The e2e headline run (`ap-ayiJz1iYjfF0M5qUt0Rhaj`) is still RUNNING — artifact pending.
- iter4 (2026-06-26): Polled the e2e headline run — it RESTARTED but is healthy and actively training (~9.5 min/epoch, epoch 2/10 at poll time; see RUNS.md), `e2e_results.json` not yet on volume, left baking. Then (minimal launch iteration) launched the **zero-shot DPLM PLL baseline** `run_headroom_gate` (`ap-a7hQxHVfIOYVuoil7e6Go7`, A10G, detached) so DoD #4's zero-shot baseline bakes in parallel with the headline run. Verified it past the image build into GPU scoring (`built 15000 doubles; skipped 0`, `scoring 20142 unique sequences`) before ending the iteration. RUNS.md + CLAIMS.md updated; no code touched, tree green.
