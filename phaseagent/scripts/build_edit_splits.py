"""Local CLI: create deterministic EditGuard dataset splits."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phaseagent.edit_splits import make_dataset_splits  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    df = pd.read_parquet(args.data)
    splits = make_dataset_splits(df["dataset_id"].unique(), seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(out, index=False)
    print(f"Wrote {len(splits)} split rows to {out}")


if __name__ == "__main__":
    main()
