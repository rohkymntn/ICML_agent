"""Local CLI: run survival-aware search frontier benchmark."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phaseagent.survival_search import run_survival_search_benchmark  # noqa: E402


def _parse_grid(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--survival-curves", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--boundaries")
    p.add_argument("--survival-col", default="survival_large_deviation")
    p.add_argument("--budget", type=int, default=50)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--beta-grid", default="0,0.01,0.03,0.05")
    p.add_argument("--gamma-grid", default="0.1,0.3,0.5,1.0")
    args = p.parse_args()

    df = pd.read_parquet(args.data)
    curves = pd.read_csv(args.survival_curves)
    datasets = {ds: sub.reset_index(drop=True) for ds, sub in df.groupby("dataset_id")}
    curve_dict = {ds: sub.reset_index(drop=True) for ds, sub in curves.groupby("dataset_id")}
    boundaries = {}
    if args.boundaries:
        bd = pd.read_csv(args.boundaries)
        boundaries = {
            r["dataset_id"]: {"dc": r.get("dc"), "alpha": r.get("alpha"), "dc_true": r.get("dc")}
            for _, r in bd.iterrows()
        }

    out_df = run_survival_search_benchmark(
        datasets=datasets,
        survival_curves=curve_dict,
        boundaries=boundaries,
        budget=args.budget,
        seeds=range(args.repeats),
        beta_grid=_parse_grid(args.beta_grid),
        gamma_grid=_parse_grid(args.gamma_grid),
        survival_col=args.survival_col,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"Wrote {len(out_df)} search rows to {out}")


if __name__ == "__main__":
    main()
