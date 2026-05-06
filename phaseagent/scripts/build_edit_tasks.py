"""Local CLI: build function-preserving editing task table."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phaseagent.edit_splits import VALID_SPLITS  # noqa: E402
from phaseagent.editing_tasks import build_editing_tasks  # noqa: E402


def _parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _parse_strs(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--budgets", default="1,2,3,5")
    p.add_argument("--objectives", default="novelty,fragility_aware,motif_avoidance")
    p.add_argument("--min-candidates", type=int, default=20)
    p.add_argument("--splits", help="Path to edit_splits.csv. Required when --split is given.")
    p.add_argument(
        "--split",
        choices=VALID_SPLITS,
        help="If set, build tasks only from datasets in this split.",
    )
    args = p.parse_args()
    df = pd.read_parquet(args.data)
    splits_df = pd.read_csv(args.splits) if args.splits else None
    if args.split and splits_df is None:
        p.error("--split requires --splits")
    tasks = build_editing_tasks(
        df,
        budgets=_parse_ints(args.budgets),
        objectives=_parse_strs(args.objectives),
        min_candidates=args.min_candidates,
        splits=splits_df,
        split_filter=args.split,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tasks.to_csv(out, index=False)
    print(f"Wrote {len(tasks)} editing tasks to {out}")


if __name__ == "__main__":
    main()
