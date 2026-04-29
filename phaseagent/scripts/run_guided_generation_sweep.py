"""Local CLI: sample-efficiency sweeps for prompted/guided generation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phaseagent.edit_eval import evaluate_generated_selection  # noqa: E402
from phaseagent.edit_generators import (  # noqa: E402
    ProposalConfig,
    generate_many_then_rerank,
    guided_local_generation,
    infer_wildtype_sequence,
    join_generated_to_dms,
    observed_single_mutation_tokens,
    plm_masked_proposal_generation,
    sample_random_edit_candidates,
)
from phaseagent.editing_tasks import task_from_row  # noqa: E402
from phaseagent.editguard_prior import DMSFunctionPrior  # noqa: E402


def _parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--tasks", required=True)
    p.add_argument("--prior", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--candidate-budgets", default="20,50,100,200,500")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--max-tasks", type=int, default=0)
    p.add_argument("--coverage-mode", choices=["dms_evaluable", "design"], default="dms_evaluable")
    args = p.parse_args()

    df = pd.read_parquet(args.data)
    tasks = pd.read_csv(args.tasks)
    if args.max_tasks > 0:
        tasks = tasks.head(args.max_tasks)
    prior = DMSFunctionPrior.load(args.prior)
    rows = []
    for task_idx, task_row in tasks.iterrows():
        task = task_from_row(task_row)
        wt = infer_wildtype_sequence(df, task.dataset_id)
        if wt is None:
            continue
        allowed = observed_single_mutation_tokens(df, task) if args.coverage_mode == "dms_evaluable" else None
        for budget in _parse_ints(args.candidate_budgets):
            for seed in range(args.seeds):
                methods = {
                    "random_direct_generation": sample_random_edit_candidates(
                        task.dataset_id,
                        wt,
                        task,
                        ProposalConfig(n_candidates=budget, seed=seed, coverage_mode=args.coverage_mode),
                        source="random_direct_generation",
                        allowed_tokens=allowed,
                    ),
                    "guided_local_generation": guided_local_generation(
                        task.dataset_id,
                        wt,
                        task,
                        prior,
                        ProposalConfig(n_candidates=budget, seed=seed, coverage_mode=args.coverage_mode),
                        allowed_tokens=allowed,
                    ),
                    "esm2_masked_marginal_proxy": plm_masked_proposal_generation(
                        task.dataset_id,
                        wt,
                        task,
                        n_candidates=budget,
                        seed=seed,
                        allowed_tokens=allowed,
                    ),
                    f"generate_{budget * 10}_then_rerank": generate_many_then_rerank(
                        task.dataset_id,
                        wt,
                        task,
                        prior,
                        n_generate=budget * 10,
                        k=budget,
                        seed=seed,
                        allowed_tokens=allowed,
                    ),
                }
                for method, generated in methods.items():
                    labeled = join_generated_to_dms(generated, df)
                    labeled["method"] = method
                    rows.append(
                        {
                            "task_idx": int(task_idx),
                            "dataset_id": task.dataset_id,
                            "objective": task.objective,
                            "edit_budget": task.edit_budget,
                            "seed": seed,
                            "method": method,
                            "candidate_budget": budget,
                            "n_generated": int(budget * 10 if "then_rerank" in method else budget),
                            "coverage_mode": args.coverage_mode,
                            **evaluate_generated_selection(labeled, task),
                        }
                    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {len(rows)} sweep rows to {out}")


if __name__ == "__main__":
    main()
