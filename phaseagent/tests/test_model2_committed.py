"""Guard the Model 2 design artifacts the paper cites (DoD #3, second half).

The real per-double table is produced by a CPU run over the GPU-extracted
`feat_GB1_Olson.npz` (volume-only) + the GB1 assay CSV. The committed traceable
artifact is `model2_oof_<assay>.csv`; `model2_<assay>_summary.json` must be a
pure function of it. We (1) smoke-test `summarize_model2` on a synthetic table so
the summary-from-CSV recompute is guarded without GPU/data, and (2) once the real
artifacts are committed, assert they reproduce and back the design claim.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from model2_design import summarize_model2  # noqa: E402

OUT = ROOT / "outputs" / "epistasis"
ASSAY = "GB1"


def _synthetic_table(n=600, seed=0):
    """A per-double table with real epistasis signal: predicted epistasis carries
    measured binding within the additively-mediocre pool, the structure Model 2
    exploits. Enough rows that the decile bins have >=30 each."""
    rng = np.random.default_rng(seed)
    glob = rng.standard_normal(n)
    true_eps = rng.standard_normal(n)
    pred_eps = 0.7 * true_eps + 0.5 * rng.standard_normal(n)
    measured = glob + true_eps + 0.1 * rng.standard_normal(n)
    return pd.DataFrame({"glob": glob, "measured": measured,
                         "true_eps": true_eps, "pred_eps": pred_eps})


def test_summarize_reproduces_from_csv(tmp_path):
    df = _synthetic_table()
    summ = summarize_model2(df)
    # round-trip through CSV: the committed summary must recompute exactly
    csv = tmp_path / "model2_oof_SYN.csv"
    df.to_csv(csv, index=False)
    recomputed = summarize_model2(pd.read_csv(csv))
    for k, v in summ.items():
        assert recomputed[k] == v, f"summary/csv drift on {k}"
    # the synthetic pred_eps is genuinely informative, so the lift is positive
    assert summ["gof_lift_model1"] > 0
    assert summ["matched_additive_spearman"] > 0
    assert summ["n_matched_bins"] >= 5


def _committed():
    csv = OUT / f"model2_oof_{ASSAY}.csv"
    js = OUT / f"model2_{ASSAY}_summary.json"
    return (csv, js) if csv.exists() and js.exists() else None


def test_committed_model2_reproduces_and_backs_claim():
    paths = _committed()
    if paths is None:
        pytest.skip("Model 2 artifacts not yet committed (design run pending)")
    csv, js = paths
    df = pd.read_csv(csv)
    s = json.loads(js.read_text())
    assert len(df) == s["n_doubles"]
    recomputed = summarize_model2(df)
    for k, v in recomputed.items():
        assert s[k] == v, f"summary/csv drift on {k}"
    # the paper's design claim: epistasis survives at matched additive, and
    # ranking by predicted epistasis lifts measured binding over the pool.
    assert s["matched_additive_spearman"] > 0
    assert s["gof_lift_model1"] > 0
