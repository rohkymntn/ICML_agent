"""Local CLI: train and evaluate the EditGuard DMS function prior."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phaseagent.edit_splits import add_splits  # noqa: E402
from phaseagent.editguard_prior import DMSFunctionPrior, evaluate_prior  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--model-out", required=True)
    p.add_argument("--metrics-out", required=True)
    p.add_argument("--n-estimators", type=int, default=200)
    args = p.parse_args()

    df = pd.read_parquet(args.data)
    splits = pd.read_csv(args.splits)
    labeled = add_splits(df, splits)
    train = labeled[labeled["split"] == "train"]
    val = labeled[labeled["split"] == "val"]
    prior = DMSFunctionPrior(n_estimators=args.n_estimators).fit(train, calibrate_df=val)
    prior.save(args.model_out)

    rows = []
    for split, sub in labeled.groupby("split"):
        rows.append({"split": split, **evaluate_prior(prior, sub)})
    metrics = pd.DataFrame(rows)
    out = Path(args.metrics_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out, index=False)
    print(f"Wrote prior to {args.model_out} and metrics to {out}")


if __name__ == "__main__":
    main()
