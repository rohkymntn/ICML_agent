"""Commit the three-layer decomposition numbers as a traceable artifact.

The committed source is ``paper/figures_epistasis/three_layer_atlas.parquet``,
the per-protein additive / +global-link / specific-epistasis decomposition over
the Megascale double-mutant stability landscape (149 protein domains, produced by
``run_epistasis_atlas`` -> ``build_epistasis_figures.py``). This script reduces it
to the two committed artifacts the paper cites for its decomposition claims:

  outputs/epistasis/decomposition_atlas.csv     per-protein table (one row / domain)
  outputs/epistasis/decomposition_summary.json  the medians / fractions in the paper

Pure pandas, no GPU, no network. Run: ``python scripts/build_decomposition_artifact.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(atlas: pd.DataFrame) -> dict:
    def med(col: str) -> float:
        v = pd.to_numeric(atlas[col], errors="coerce").to_numpy(float)
        v = v[np.isfinite(v)]
        return float(np.median(v)) if v.size else float("nan")

    spec = pd.to_numeric(atlas["spec_std"], errors="coerce").to_numpy(float)
    spec = spec[np.isfinite(spec)]
    share = pd.to_numeric(atlas["spec_var_share"], errors="coerce").to_numpy(float)
    share = share[np.isfinite(share)]
    add = pd.to_numeric(atlas["r2_additive"], errors="coerce").to_numpy(float)
    add = add[np.isfinite(add)]
    return {
        "n_proteins": int(len(atlas)),
        "total_double_mutants": int(pd.to_numeric(atlas["n"], errors="coerce").sum()),
        "median_r2_additive": med("r2_additive"),
        "median_r2_global": med("r2_global"),
        "median_spec_std_kcal": med("spec_std"),
        "median_spec_var_share": med("spec_var_share"),
        "frac_proteins_additive_r2_negative": float(np.mean(add < 0)) if add.size else float("nan"),
        "frac_proteins_spec_std_gt_0p3": float(np.mean(spec > 0.3)) if spec.size else float("nan"),
        "frac_proteins_spec_var_share_gt_0p1": float(np.mean(share > 0.1)) if share.size else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="paper/figures_epistasis/three_layer_atlas.parquet")
    ap.add_argument("--outdir", default="outputs/epistasis")
    args = ap.parse_args()

    atlas = pd.read_parquet(args.atlas)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    atlas.to_csv(outdir / "decomposition_atlas.csv", index=False)
    summary = summarize(atlas)
    (outdir / "decomposition_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"[decomp] {summary['n_proteins']} proteins, "
          f"{summary['total_double_mutants']:,} double mutants")
    print(f"[decomp] median R2 additive {summary['median_r2_additive']:.3f} "
          f"-> +global {summary['median_r2_global']:.3f}; "
          f"specific std {summary['median_spec_std_kcal']:.3f} kcal/mol "
          f"({summary['median_spec_var_share']*100:.0f}% of variance)")
    print(f"[decomp] wrote {outdir}/decomposition_atlas.csv + decomposition_summary.json")


if __name__ == "__main__":
    main()
