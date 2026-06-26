"""Guard the committed zero-shot DPLM PLL stability baseline the paper cites.

The headroom run is GPU-only and cannot be recomputed locally, so instead we
assert the committed per-double parquet reproduces the committed summary keys
(no parquet/summary drift) and that those keys equal the numbers in paper
section 7.2 (``tab:zeroshot``) / CLAIMS.md.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "outputs" / "epistasis" / "headroom_dplm.parquet"
SUMMARY = ROOT / "outputs" / "epistasis" / "headroom_summary.json"


def test_parquet_reproduces_summary():
    df = pd.read_parquet(PARQUET)
    s = json.loads(SUMMARY.read_text())
    assert len(df) == s["n_doubles_scored"]
    assert df["dataset_id"].nunique() == s["n_proteins"]

    # per-protein Spearman (min 10 doubles, matching the run) reproduces the summary
    rhos = [
        spearmanr(d["dplm_epistasis"], d["eps_specific"]).correlation
        for _, d in df.groupby("dataset_id")
        if len(d) >= 10
    ]
    rhos = np.array(rhos)
    assert abs(float(np.median(rhos)) - s["median_per_protein_spearman"]) < 1e-6
    assert abs(float((rhos > 0.2).mean()) - s["frac_proteins_pp_spearman_gt_0p2"]) < 1e-6
    overall = spearmanr(df["dplm_epistasis"], df["eps_specific"]).correlation
    assert abs(float(overall) - s["overall_spearman"]) < 1e-6


def test_summary_matches_paper_numbers():
    """The rounded values printed in paper section 7.2 / CLAIMS.md."""
    s = json.loads(SUMMARY.read_text())
    assert s["n_proteins"] == 30
    assert s["n_doubles_scored"] == 15000
    assert round(s["median_per_protein_spearman"], 2) == 0.25
    assert round(s["overall_spearman"], 2) == 0.23
    assert round(s["frac_proteins_pp_spearman_gt_0p2"], 2) == 0.60
