"""Local CLI: make selected advanced Spectral PhaseAgent figures."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phaseagent.plots_advanced import (  # noqa: E402
    plot_distance_shell_histogram,
    plot_search_frontier,
    plot_spectrum_histograms,
    plot_survival_curves,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data")
    p.add_argument("--single-effects")
    p.add_argument("--survival-curves")
    p.add_argument("--search-results")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    if args.data:
        df = pd.read_parquet(args.data)
        made.append(plot_distance_shell_histogram(df, out_dir / "fig1_distance_shell_histogram.png"))
    if args.single_effects:
        effects = pd.read_parquet(args.single_effects)
        made.append(plot_spectrum_histograms(effects, out_dir / "fig2_spectrum_histograms.png"))
    if args.survival_curves:
        curves = pd.read_csv(args.survival_curves)
        true_col = "survival_true" if "survival_true" in curves.columns else "survival_additive_mc"
        pred_col = "survival_large_deviation" if "survival_large_deviation" in curves.columns else None
        made.append(plot_survival_curves(curves, out_dir / "fig3_survival_curves.png", true_col=true_col, pred_col=pred_col))
    if args.search_results:
        results = pd.read_csv(args.search_results)
        made.append(plot_search_frontier(results, out_dir / "fig7_search_frontier.png"))
    for path in made:
        print(path)


if __name__ == "__main__":
    main()
