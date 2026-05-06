"""Local CLI: run the DMS-pool guided sampler.

This is the renamed-and-refactored ``editguard_diffusion`` entrypoint. It
emits ``method=dms_pool_guided`` in metric rows and is a *selection*
baseline, not a generative diffusion model. The bare name
``editguard_diffusion`` is reserved for the future DPLM-backed sampler
(Phase 2 of the implementation plan).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phaseagent.edit_eval import evaluate_edit_selection  # noqa: E402
from phaseagent.edit_splits import VALID_SPLITS, assert_tasks_in_split  # noqa: E402
from phaseagent.editing_tasks import candidate_pool_for_task, task_from_row  # noqa: E402
from phaseagent.editguard_diffusion import DMSPoolGuidedSampler, DiffusionSampleConfig  # noqa: E402
from phaseagent.editguard_prior import DMSFunctionPrior  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--tasks", required=True)
    p.add_argument("--prior", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--selections-out")
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--max-tasks", type=int, default=0)
    p.add_argument("--max-candidates", type=int, default=10000)
    p.add_argument("--splits", help="Path to edit_splits.csv. Required with --require-split.")
    p.add_argument(
        "--require-split",
        choices=VALID_SPLITS,
        help="Fail if tasks reference any dataset outside this split.",
    )
    args = p.parse_args()

    df = pd.read_parquet(args.data)
    tasks = pd.read_csv(args.tasks)
    if args.require_split:
        if not args.splits:
            p.error("--require-split needs --splits")
        assert_tasks_in_split(tasks, pd.read_csv(args.splits), args.require_split)
    if args.max_tasks > 0:
        tasks = tasks.head(args.max_tasks)
    prior = DMSFunctionPrior.load(args.prior)
    sampler = DMSPoolGuidedSampler(prior)
    metric_rows, selection_rows = [], []
    for task_idx, row in tasks.iterrows():
        task = task_from_row(row)
        pool = candidate_pool_for_task(df, task)
        if len(pool) == 0:
            continue
        for seed in range(args.seeds):
            if args.max_candidates > 0 and len(pool) > args.max_candidates:
                pool_seed = int(task_idx) * 1000 + seed
                pool_run = pool.sample(args.max_candidates, random_state=pool_seed)
            else:
                pool_run = pool
            selected = sampler.sample(pool_run, task, DiffusionSampleConfig(n_samples=args.k, seed=seed))
            metric_rows.append(
                {
                    "task_idx": int(task_idx),
                    "dataset_id": task.dataset_id,
                    "objective": task.objective,
                    "edit_budget": task.edit_budget,
                    "seed": seed,
                    "method": "dms_pool_guided",
                    **evaluate_edit_selection(selected, task),
                }
            )
            tmp = selected.copy()
            tmp["task_idx"] = int(task_idx)
            tmp["seed"] = seed
            selection_rows.append(tmp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(out, index=False)
    if args.selections_out and selection_rows:
        sel_path = Path(args.selections_out)
        sel_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(selection_rows, ignore_index=True).to_parquet(sel_path, index=False)
    print(f"Wrote {len(metric_rows)} diffusion metric rows to {out}")


if __name__ == "__main__":
    main()
