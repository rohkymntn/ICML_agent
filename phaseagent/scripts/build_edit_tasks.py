"""Local CLI: build function-preserving editing task table."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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
    args = p.parse_args()
    df = pd.read_parquet(args.data)
    tasks = build_editing_tasks(
        df,
        budgets=_parse_ints(args.budgets),
        objectives=_parse_strs(args.objectives),
        min_candidates=args.min_candidates,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tasks.to_csv(out, index=False)
    print(f"Wrote {len(tasks)} editing tasks to {out}")


if __name__ == "__main__":
    main()
