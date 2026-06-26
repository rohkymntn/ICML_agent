"""Guard the committed decomposition artifact the paper cites.

These assert the summary that ``build_decomposition_artifact.py`` derives from
the committed atlas matches the numbers in the paper / CLAIMS.md, so a silent
regen drift (or a broken atlas) fails CI instead of fabricating a paper number.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / "paper" / "figures_epistasis" / "three_layer_atlas.parquet"
SUMMARY = ROOT / "outputs" / "epistasis" / "decomposition_summary.json"


def test_summary_matches_committed_atlas():
    from scripts.build_decomposition_artifact import summarize

    atlas = pd.read_parquet(ATLAS)
    s = summarize(atlas)
    assert s["n_proteins"] == 149
    assert s["total_double_mutants"] == 131062
    # additive baseline is worse than predicting the mean on a majority of proteins
    assert s["median_r2_additive"] < -0.5
    assert s["frac_proteins_additive_r2_negative"] > 0.7
    # the global saturation link recovers most of it
    assert s["median_r2_global"] > 0.7
    # specific epistasis is real (>> 0.1 kcal/mol noise) and widespread
    assert s["median_spec_std_kcal"] > 0.3
    assert s["frac_proteins_spec_var_share_gt_0p1"] > 0.9


def test_committed_summary_in_sync():
    """The committed JSON must equal a fresh recompute (no stale numbers)."""
    from scripts.build_decomposition_artifact import summarize

    fresh = summarize(pd.read_parquet(ATLAS))
    committed = json.loads(SUMMARY.read_text())
    for k, v in fresh.items():
        assert k in committed, f"missing key {k}"
        assert abs(committed[k] - v) < 1e-9, f"{k}: committed {committed[k]} != fresh {v}"
