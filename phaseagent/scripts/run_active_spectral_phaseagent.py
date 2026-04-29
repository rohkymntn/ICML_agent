"""Local CLI: simulate spectral-prior active phase querying."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phaseagent.active_phaseagent import POLICIES  # noqa: E402
from phaseagent.agents import simulate_boundary_discovery  # noqa: E402


class _FunctionPolicy:
    def __init__(self, name, fn, seed: int = 0):
        self.name = name
        self.fn = fn
        self.seed = seed

    def select_batch(self, observed_df, pool_df, batch_size):
        try:
            return self.fn(observed_df, pool_df, batch_size, seed=self.seed)
        except TypeError:
            return self.fn(observed_df, pool_df, batch_size)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--policy", default="mutual_information_phaseagent", choices=sorted(POLICIES))
    p.add_argument("--initial-n", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--repeats", type=int, default=10)
    args = p.parse_args()

    df = pd.read_parquet(args.data)
    rows = []
    for ds_id, sub in df.groupby("dataset_id"):
        if sub["mutation_distance"].nunique() < 3:
            continue
        policy = _FunctionPolicy(args.policy, POLICIES[args.policy])
        sim = simulate_boundary_discovery(
            sub.reset_index(drop=True),
            policy,
            initial_n=args.initial_n,
            batch_size=args.batch_size,
            n_steps=args.steps,
            n_repeats=args.repeats,
        )
        sim["dataset_id"] = ds_id
        rows.append(sim)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(out, index=False)
    print(f"Wrote active-query simulation rows to {out}")


if __name__ == "__main__":
    main()
