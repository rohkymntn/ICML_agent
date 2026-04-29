"""Local CLI: ESM2 pseudo-likelihood scoring (requires torch + fair-esm).

For Modal-native GPU execution, prefer:
    modal run modal_app.py::run_plm_scoring
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phaseagent.plm import score_dataset_plm  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model-name", default="esm2_t33_650M_UR50D")
    p.add_argument("--max-per-dataset", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()

    df = pd.read_parquet(args.data)
    out_df = score_dataset_plm(
        df,
        model_name=args.model_name,
        max_per_dataset=args.max_per_dataset,
        batch_size=args.batch_size,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"Wrote {len(out_df)} rows to {out}")


if __name__ == "__main__":
    main()
