# HARVEST PATCH — gen_design.json (DoD #4 discrete-diffusion guidance baseline)

Ready-to-apply patch for the moment `outputs/epistasis/gen_design.json` lands on the
Modal volume (run `ap-fcAdDhVRfLn1J29tdZPVyD`, `train_generative_design`, GB1_Wu, 3 epochs).
This converts the harvest from "improvise under pressure" into "paste + fill 7 numbers".
It eliminates the documented self-contradiction landmine: the paper currently *denies*
benchmarking discrete-diffusion guidance in two places (§7-baselines + §9-limitations);
both MUST be reconciled in the same commit that adds the new subsection.

All `{{SLOT}}` values come from `gen_design.json` keys (writer: `modal_app_v2.py:3260-3270`).
The 11 keys are: `assay, positions, epochs, generated_lib_mean_function,
generated_NOVEL_lib_mean_function, n_novel_generated, coverage, unconditioned_DPLM_mean,
random_lib_mean, top_library_ceiling, library_mean_overall`.

## Step 0 — pull + commit the artifact, run the guard
```
modal volume get phaseagent-data outputs/epistasis/gen_design.json outputs/epistasis/
PYTHONPATH=src python -m pytest -q tests/test_gen_design_committed.py
```
The guard (`tests/test_gen_design_committed.py`) auto-activates and asserts BOTH structural
validity AND the directional claim: `generated_lib_mean_function` >
`random_lib_mean`, > `unconditioned_DPLM_mean`, > `library_mean_overall`, and
`<= top_library_ceiling`.

**CONTINGENCY — if the directional guard FAILS** (reward-guided generation did NOT beat all
baselines): do NOT force the claim below. Reframe the subsection honestly to the actual
numbers (e.g. "reward-guided generation only marginally improves over the unconditioned
library"), and relax the guard's directional assertion to match what actually happened.
The structural-validity half of the guard must still pass. Never write a number the
artifact does not support.

## Step 1 — new results subsection
Insert AFTER `\end{figure}` of `tab:model2`/`fig:model2` (current line 604, i.e. right
before `\subsection{Cross-protein generalization}` at line 605) in `paper/epistasis_icml.tex`:

```latex
\subsection{A discrete-diffusion guidance baseline}
\label{sec:results-gen}

For completeness we benchmark one strong generative design baseline: a reward-weighted
LoRA fine-tune of DPLM-650M that generates multi-mutants by iterative masked-diffusion
sampling, the discrete-diffusion reward-guidance approach of \citet{wang2025drakes} and
\citet{li2024svdd}. Because this method edits \emph{every} site, we run it on the complete
four-site combinatorial GB1 assay (Wu, $149{,}360$ measured combinations
\citep{wu2016adaptation}), where every sampled combination has a ground-truth function
score; it is therefore a separate experiment from the pairwise Model~2 ranking task, not a
head-to-head competitor. After {{epochs}} epochs of reward-weighted fine-tuning, the
generated library reaches mean function ${{generated_lib_mean_function}}$, above the
unconditioned DPLM library (${{unconditioned_DPLM_mean}}$), a random library
(${{random_lib_mean}}$), and the assay-wide mean (${{library_mean_overall}}$), but short of
the top-library ceiling (${{top_library_ceiling}}$); coverage of measured combinations is
${{coverage}}$. Reward-guided discrete diffusion thus does lift a generated library above
the unconditioned and random baselines, confirming the guidance signal is real, yet -- like
the zero-shot reranker -- it does not reach the gain available to a method that exploits
measured specific epistasis. All numbers derive from the committed
\texttt{outputs/epistasis/gen\_design.json}.
```
Round each number the way the rest of the paper rounds (2 sig figs / 2 decimals; `coverage`
is already rounded to 3 in the artifact). Keep the sign sentence accurate to the actual
ordering — re-read the JSON before committing the words "above"/"short of".

## Step 2 — RECONCILE §7-baselines (REQUIRED; currently lines 304-309)
Replace:
```latex
Stronger
generative comparators -- discrete-diffusion guidance
\citep{gruver2023protein,nisonoff2025unlocking} and reward-guided refinement
\citep{wang2025drakes,li2024svdd} -- are required only if a generative design claim
is made; our Model 2 is a ranking policy over measurable candidates, so the
additive and best-of-$N$ controls are the matched baselines.
```
with:
```latex
For design we additionally benchmark one strong generative comparator: a
discrete-diffusion guidance baseline (reward-weighted fine-tuning of DPLM
\citep{wang2025drakes,li2024svdd,gruver2023protein,nisonoff2025unlocking}) that
generates multi-mutants on the complete four-site combinatorial GB1 assay
(\cref{sec:results-gen}). Model 2 itself is a ranking policy over measurable
candidates, so for the ranking claim the additive and best-of-$N$ controls remain
the matched baselines.
```

## Step 3 — RECONCILE §9-limitations (REQUIRED; currently lines 670-673)
Replace:
```latex
We do not make a generative design claim and therefore do not benchmark
against discrete-diffusion guidance \citep{gruver2023protein,nisonoff2025unlocking}
or reward-guided refinement \citep{wang2025drakes,li2024svdd}; Model 2 is a
ranking policy over measurable candidates.
```
with:
```latex
Model 2 is a ranking policy over measurable candidates, not a generative method;
we report a discrete-diffusion guidance baseline (\cref{sec:results-gen}) as a
reference for what reward-guided generation \citep{wang2025drakes,li2024svdd}
achieves on the same combinatorial assay, but we make no generative design claim of
our own.
```
After Steps 2-3, all four cite keys (`gruver2023protein, nisonoff2025unlocking,
wang2025drakes, li2024svdd`) still resolve (related-work §2 + the new text), so no orphan
cites. The related-work sentence at line 183-185 ("guided generation as related context
rather than a matched baseline for the ranking claim") stays accurate as-scoped (it is about
the *ranking* claim); OPTIONAL polish: append "; we additionally report a guided-generation
baseline on the complete combinatorial GB1 assay (\cref{sec:results-gen})".

## Step 4 — CLAIMS.md rows
Add a new GROUNDED section (mirror the Model 2 section format):
```markdown
## Discrete-diffusion guidance baseline (Section "results-gen", gen_design) — GROUNDED

Source: `outputs/epistasis/gen_design.json` — reward-weighted LoRA fine-tune of DPLM-650M on
the complete 4-site combinatorial GB1_Wu assay (`modal_app_v2.py::train_generative_design`,
run ap-fcAdDhVRfLn1J29tdZPVyD, 3 epochs, H100). Guarded by `tests/test_gen_design_committed.py`
(structural validity + reward-guided library beats random/unconditioned/overall, under the ceiling).

| Claim in paper | Value | Artifact key / column |
|---|---|---|
| Assay (4 sites) | GB1_Wu [265,266,267,280] | `gen_design.json:assay`, `:positions` |
| Fine-tune epochs | {{epochs}} | `gen_design.json:epochs` |
| Generated library mean function | {{generated_lib_mean_function}} | `gen_design.json:generated_lib_mean_function` |
| Unconditioned DPLM library mean | {{unconditioned_DPLM_mean}} | `gen_design.json:unconditioned_DPLM_mean` |
| Random library mean | {{random_lib_mean}} | `gen_design.json:random_lib_mean` |
| Assay-wide library mean | {{library_mean_overall}} | `gen_design.json:library_mean_overall` |
| Top-library ceiling | {{top_library_ceiling}} | `gen_design.json:top_library_ceiling` |
| Library coverage of measured combos | {{coverage}} | `gen_design.json:coverage` |
```

## Step 5 — verify + final self-contradiction grep
```
PYTHONPATH=src python -m pytest -q            # expect 123 passed (gen guard now active), 5 skipped
tectonic -X compile paper/epistasis_icml.tex --outdir _build_harvest   # 0 undefined refs/cites, 0 overfull, 0 ??
grep -niE "do not benchmark|required only if a generative|we do not make a generative" paper/epistasis_icml.tex
# ^ MUST return nothing. If it does, Steps 2-3 are incomplete.
rm -rf _build_harvest
```
Then update PROGRESS.md (DoD #1/#4/#8/#9 → ☑), RUNS.md (gen run → DONE with real numbers),
and flip PROGRESS.md to COMPLETE → report `ICML_READY`.
