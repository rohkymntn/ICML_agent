"""Local CLI: simulate active boundary discovery."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phaseagent.agents import (  # noqa: E402
    BoundaryGreedyPolicy,
    PhaseAgentPolicy,
    RandomPolicy,
    UncertaintyShellPolicy,
    UniformShellPolicy,
    simulate_boundary_discovery,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--initial-n", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--datasets", nargs="*", default=None)
    args = p.parse_args()

    df = pd.read_parquet(args.data)
    if args.datasets:
        df = df[df["dataset_id"].isin(args.datasets)]

    rng = np.random.default_rng(0)
    policies = [
        RandomPolicy(rng),
        UniformShellPolicy(rng),
        UncertaintyShellPolicy(rng),
        BoundaryGreedyPolicy(rng),
        PhaseAgentPolicy(rng),
    ]
    rows = []
    for ds_id, sub in df.groupby("dataset_id"):
        for pol in policies:
            sim = simulate_boundary_discovery(
                sub.reset_index(drop=True),
                policy=pol,
                initial_n=args.initial_n,
                batch_size=args.batch_size,
                n_steps=args.steps,
                n_repeats=args.repeats,
            )
            sim["dataset_id"] = ds_id
            rows.append(sim)
    out_df = pd.concat(rows, ignore_index=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    main()
