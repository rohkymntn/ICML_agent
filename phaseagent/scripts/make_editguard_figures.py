"""Local CLI: render EditGuard ICML-style figures."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phaseagent.edit_plots import plot_editing_frontier, plot_method_boxplot  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    results = pd.read_csv(args.results)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_editing_frontier(results, out / "fig2_editing_frontier")
    plot_method_boxplot(results, out / "fig4_functional_hit_boxplot")
    print(f"Wrote EditGuard figures to {out}")


if __name__ == "__main__":
    main()
