"""Local CLI: compare real proposal generation against generate-then-rerank."""
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
    sample_random_edit_candidates,
)
from phaseagent.editing_tasks import task_from_row  # noqa: E402
from phaseagent.editguard_prior import DMSFunctionPrior  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--tasks", required=True)
    p.add_argument("--prior", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--selections-out")
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--n-generate", type=int, default=500)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--max-tasks", type=int, default=20)
    args = p.parse_args()

    df = pd.read_parquet(args.data)
    tasks = pd.read_csv(args.tasks)
    if args.max_tasks > 0:
        tasks = tasks.head(args.max_tasks)
    prior = DMSFunctionPrior.load(args.prior)

    metrics, selections = [], []
    for task_idx, row in tasks.iterrows():
        task = task_from_row(row)
        wt = infer_wildtype_sequence(df, task.dataset_id)
        if wt is None:
            continue
        allowed_tokens = observed_single_mutation_tokens(df, task)
        for seed in range(args.seeds):
            methods = {}
            random_props = sample_random_edit_candidates(
                task.dataset_id,
                wt,
                task,
                ProposalConfig(n_candidates=args.k, seed=seed),
                source="random_direct_generation",
                allowed_tokens=allowed_tokens,
            )
            random_props["method"] = "random_direct_generation"
            methods["random_direct_generation"] = random_props
            methods[f"generate_{args.n_generate}_then_rerank"] = generate_many_then_rerank(
                task.dataset_id,
                wt,
                task,
                prior,
                n_generate=args.n_generate,
                k=args.k,
                seed=seed,
                allowed_tokens=allowed_tokens,
            )
            guided = guided_local_generation(
                task.dataset_id,
                wt,
                task,
                prior,
                ProposalConfig(n_candidates=args.k, seed=seed),
                allowed_tokens=allowed_tokens,
            )
            guided["method"] = "guided_local_generation"
            methods["guided_local_generation"] = guided

            for method, generated in methods.items():
                labeled = join_generated_to_dms(generated, df)
                labeled["task_idx"] = int(task_idx)
                labeled["seed"] = seed
                labeled["method"] = method
                selections.append(labeled)
                metrics.append(
                    {
                        "task_idx": int(task_idx),
                        "dataset_id": task.dataset_id,
                        "objective": task.objective,
                        "edit_budget": task.edit_budget,
                        "seed": seed,
                        "method": method,
                        "n_generated": int(args.n_generate if "then_rerank" in method else args.k),
                        **evaluate_generated_selection(labeled, task),
                    }
                )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(out, index=False)
    if args.selections_out and selections:
        sel_path = Path(args.selections_out)
        sel_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(selections, ignore_index=True).to_parquet(sel_path, index=False)
    print(f"Wrote {len(metrics)} generation metric rows to {out}")


if __name__ == "__main__":
    main()
