"""Local CLI: run EditGuard baselines on measured DMS candidate pools."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phaseagent.edit_baselines import run_edit_baselines  # noqa: E402
from phaseagent.edit_eval import evaluate_methods  # noqa: E402
from phaseagent.edit_splits import VALID_SPLITS, assert_tasks_in_split  # noqa: E402
from phaseagent.editing_tasks import candidate_pool_for_task, task_from_row  # noqa: E402
from phaseagent.editguard_prior import DMSFunctionPrior  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--tasks", required=True)
    p.add_argument("--prior", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--selections-out")
    p.add_argument("--external-candidates")
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
    external = None
    if args.external_candidates:
        ext_path = Path(args.external_candidates)
        external = pd.read_parquet(ext_path) if ext_path.suffix == ".parquet" else pd.read_csv(ext_path)
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
            selections = run_edit_baselines(
                pool_run,
                task,
                prior,
                k=args.k,
                seed=seed,
                external_candidates=external,
            )
            metrics = evaluate_methods(selections, task)
            metrics["task_idx"] = int(task_idx)
            metrics["dataset_id"] = task.dataset_id
            metrics["objective"] = task.objective
            metrics["edit_budget"] = task.edit_budget
            metrics["seed"] = seed
            metric_rows.append(metrics)
            for method, sel in selections.items():
                tmp = sel.copy()
                tmp["task_idx"] = int(task_idx)
                tmp["method"] = method
                tmp["seed"] = seed
                selection_rows.append(tmp)
    out_df = pd.concat(metric_rows, ignore_index=True) if metric_rows else pd.DataFrame()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    if args.selections_out and selection_rows:
        sel_path = Path(args.selections_out)
        sel_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(selection_rows, ignore_index=True).to_parquet(sel_path, index=False)
    print(f"Wrote {len(out_df)} metric rows to {out}")


if __name__ == "__main__":
    main()
