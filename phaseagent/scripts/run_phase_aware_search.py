"""Local CLI: phase-aware vs baseline search benchmark."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phaseagent.search import run_search_benchmark  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--boundaries", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--budget", type=int, default=50)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--lambda-boundary", type=float, default=0.5)
    args = p.parse_args()

    df = pd.read_parquet(args.data)
    bd = pd.read_csv(args.boundaries)
    bd_dict = {
        r["dataset_id"]: {"dc": r["dc"], "alpha": r["alpha"], "dc_true": r["dc"]}
        for _, r in bd.iterrows()
    }
    datasets_dict = {ds: sub.reset_index(drop=True) for ds, sub in df.groupby("dataset_id")}
    seeds = list(range(args.repeats))
    out_df = run_search_benchmark(
        datasets=datasets_dict,
        boundaries=bd_dict,
        budgets=[args.budget],
        seeds=seeds,
        lambda_boundary=args.lambda_boundary,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    main()
