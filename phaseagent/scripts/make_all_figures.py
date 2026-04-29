"""Local CLI: build all figures from result tables."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phaseagent.plots import make_all_figures  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tables", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data", default=None)
    args = p.parse_args()

    n = make_all_figures(
        Path(args.tables),
        Path(args.out),
        Path(args.data) if args.data else None,
    )
    print(f"Wrote {n} figures to {args.out}")


if __name__ == "__main__":
    main()
