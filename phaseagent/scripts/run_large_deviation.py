"""Local CLI: compute additive survival baselines."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phaseagent.eval_survival import evaluate_survival_prediction, true_survival_curve  # noqa: E402
from phaseagent.large_deviation import additive_survival_curve  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-mc", type=int, default=20000)
    p.add_argument("--min-shells", type=int, default=2)
    args = p.parse_args()

    df = pd.read_parquet(args.data)
    curves, metrics = [], []
    for ds_id, sub in df.groupby("dataset_id"):
        if sub["mutation_distance"].nunique() < args.min_shells:
            continue
        depths = range(0, int(sub["mutation_distance"].max()) + 1)
        curve = additive_survival_curve(sub, depths=depths, n_mc=args.n_mc)
        curve["dataset_id"] = ds_id
        true = true_survival_curve(sub)
        curve = curve.merge(
            true[["mutation_distance", "survival_true", "n", "stderr"]],
            on="mutation_distance",
            how="left",
        )
        curves.append(curve)

        pred = curve.rename(columns={"survival_large_deviation": "survival_pred"})
        m = evaluate_survival_prediction(pred, true, pred_col="survival_pred")
        metrics.append({"dataset_id": ds_id, **m})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(curves, ignore_index=True).to_csv(out_path, index=False)
    pd.DataFrame(metrics).to_csv(out_path.parent / "large_deviation_metrics.csv", index=False)
    print(f"Wrote {len(curves)} additive survival curves to {out_path}")


if __name__ == "__main__":
    main()
