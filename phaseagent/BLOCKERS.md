# BLOCKERS — decisions a human must make before declaring ICML_READY

## B1 (iter14): Does DoD #4 require a discrete-diffusion guidance baseline?

**Status (updated iter15):** BEING RESOLVED BY COMPUTATION — superseding the iter14
"human scope call" framing. Rather than argue the baseline scoped-out, iter15 launched
the actual baseline the DoD names: `train_generative_design` (`ap-BlbHQjQn930EN3cSt5zDpZ`,
H100, detached, RUNNING — an initial A10G launch OOM'd; fixed to `gpu="H100"`) — a function-weighted (reward-guided) LoRA fine-tune of DPLM-650M
(a discrete-diffusion PLM) that GENERATES high-function multi-mutants via iterative
masked-diffusion sampling, evaluated on the COMPLETE combinatorial GB1_Wu assay (4 sites,
149,360 measured combos) so every generated combo is measurable. This IS "one strong
discrete-diffusion guidance baseline." When `gen_design.json` lands and is folded into a
grounded generative-design subsection + CLAIMS.md row + guard test, **B1 dissolves** — the
baseline exists by computation, so no scope argument is needed and no human ratification is
required to satisfy DoD #4's literal wording. (If the run fails terminally, fall back to the
scope argument below, which a human would then need to ratify.)

**Original framing (iter14), retained for context:** OPEN — the last substantive gate to
`ICML_READY`. Everything else in the Definition of Done is grounded and green (see PROGRESS.md).

**The DoD #4 wording:** "Main-conference baseline set is computed and tabulated on the
SAME tasks: zero-shot DPLM PLL, additive baseline, and the design/generative
comparators the paper's claims require (at minimum best-of-N rerank and one strong
discrete-diffusion guidance baseline; add DRAKES / SVDD / Reward-Guided Iterative
Refinement if a generative claim is made)."

**What is DONE (all on the SAME tasks, all committed + in the paper):**
- zero-shot DPLM PLL — `headroom_summary.json` (stability) + `model1_*_summary.json`
  (function); §7.2 `tab:zeroshot` + `tab:model1`.
- additive baseline — the structural null throughout; the design pool mean in
  `tab:model2` (−3.62).
- best-of-N rerank — `model2_GB1_summary.json:gof_lift_zeroshot` (−0.22, reranking the
  design pool by the zero-shot DPLM PLL reward); §7.4 `tab:model2`.

**The question:** must we ALSO add "one strong discrete-diffusion guidance baseline"
(e.g. classifier/reward-guided DPLM sampling), or is it scoped out because the paper
makes a RANKING (not generative) design claim?

**The case for scoped-out (the position the paper currently takes, in §6.4 + the
Limitations section):**
- Model 2 is a *reranker/selector* over a fixed, fully-measured combinatorial pool
  (GB1-Olson complete pairwise). It is not a generative model and the paper makes no
  generative claim.
- Guidance-based discrete-diffusion methods (DRAKES, SVDD, reward-guided refinement)
  are *generative sampling* procedures. On a CLOSED pool there is nothing to sample —
  to evaluate one on this task you would have to convert it into a scorer/reranker,
  at which point it collapses into a best-of-N rerank. The strongest such reranker
  available from the discrete-diffusion PLM itself is the zero-shot DPLM PLL rerank,
  which IS reported (−0.22 lift).
- So under the paper's stated scope the matched baseline set is complete.

**The case for required-anyway:** a literal reading of "at minimum ... and one strong
discrete-diffusion guidance baseline" treats it as mandatory regardless of claim type.
Satisfying it would mean either (a) adding a generative design claim + a real
guidance baseline (DRAKES/SVDD) on an OPEN design task (a much larger scope, new GPU
runs, likely a different assay framing), or (b) arguing the zero-shot DPLM PLL rerank
already serves as the discrete-diffusion-model baseline specialized to a closed task.

**Recommendation:** keep the ranking scope (option-b framing) and treat DoD #4 as
satisfied; the paper is honest and internally coherent about this. But this is a
scope decision only a human should ratify before flipping PROGRESS.md to COMPLETE and
emitting `ICML_READY`. Do NOT declare ICML_READY until this is resolved.

## Final-audit checklist (do before ICML_READY, once B1 is resolved)
- Re-read the compiled PDF end-to-end for any stale forward-reference or orphan figure.
- Confirm every committed figure under `paper/figures_epistasis/` that the paper does
  NOT cite is either cited or intentionally unused (no orphan implied as a result).
- Confirm CLAIMS.md has a row for every number in the compiled body.
