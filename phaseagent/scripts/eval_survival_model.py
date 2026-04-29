"""Local CLI: evaluate predicted survival curves."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phaseagent.eval_survival import evaluate_survival_prediction, true_survival_curve  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pred-col", default="survival_pred")
    args = p.parse_args()

    df = pd.read_parquet(args.data)
    pred = pd.read_csv(args.predictions)
    rows = []
    for ds_id, sub in df.groupby("dataset_id"):
        pred_sub = pred[pred["dataset_id"] == ds_id] if "dataset_id" in pred.columns else pred
        if len(pred_sub) == 0:
            continue
        rows.append(
            {
                "dataset_id": ds_id,
                **evaluate_survival_prediction(
                    pred_sub,
                    true_survival_curve(sub),
                    pred_col=args.pred_col,
                ),
            }
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {len(rows)} evaluation rows to {out}")


if __name__ == "__main__":
    main()
