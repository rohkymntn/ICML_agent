"""Guard the committed Model 1 function artifacts the paper cites (DoD #3).

The real `feat_<assay>.npz` is GPU-only and lives on the Modal volume, so the
committed traceable artifact is the per-double out-of-fold CSV. We assert the
committed summary JSON reproduces exactly from that CSV (no csv/summary drift)
and that its keys equal the rounded numbers in paper section 7.4 / CLAIMS.md.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_model1 import summarize  # noqa: E402

OUT = ROOT / "outputs" / "epistasis"
ASSAYS = ["GFP", "GB1_Olson"]


def _committed(assay):
    """The committed Model 1 artifacts for an assay, or None if not yet landed.

    The artifacts are produced by a CPU training run over the GPU-extracted
    `feat_<assay>.npz` (volume-only); this guard activates the moment they are
    committed and skips cleanly before then, so the tree stays green meanwhile."""
    csv = OUT / f"model1_oof_{assay}.csv"
    js = OUT / f"model1_{assay}_summary.json"
    return (csv, js) if csv.exists() and js.exists() else None


def test_committed_summary_reproduces_from_oof_csv():
    seen = False
    for assay in ASSAYS:
        paths = _committed(assay)
        if paths is None:
            continue
        seen = True
        csv, js = paths
        df = pd.read_csv(csv)
        s = json.loads(js.read_text())
        assert len(df) == s["n_doubles"]
        recomputed = summarize(df, assay, s["dim"], s["use_shift"])
        for k, v in recomputed.items():
            assert s[k] == v, f"{assay}: summary/csv drift on {k}"
    if not seen:
        pytest.skip("Model 1 artifacts not yet committed (training run pending)")


# (decimals, expected) pinned to paper section 7.2 / 7.3 tab:model1 / CLAIMS.md
PAPER = {
    "GB1_Olson": {"zeroshot_spearman": (3, 0.021), "model1_doubles_spearman": (2, 0.36),
                  "model1_position_spearman": (2, 0.12), "delta_doubles": (2, 0.34)},
    "GFP": {"zeroshot_spearman": (3, 0.005), "model1_doubles_spearman": (2, 0.14),
            "model1_position_spearman": (2, 0.08), "delta_doubles": (2, 0.13)},
}


def test_summary_matches_paper_numbers():
    """Pin the exact rounded values in tab:model1 so a regenerated+recommitted
    summary cannot silently desync the paper (the no-drift test would still pass)."""
    seen = False
    for assay, want in PAPER.items():
        paths = _committed(assay)
        if paths is None:
            continue
        seen = True
        _, js = paths
        s = json.loads(js.read_text())
        for k, (nd, val) in want.items():
            assert round(s[k], nd) == val, f"{assay} {k}: {s[k]} != paper {val}"
    if not seen:
        pytest.skip("Model 1 artifacts not yet committed (training run pending)")


def test_model1_beats_zeroshot_on_function():
    """The paper's claim: learned Model 1 beats the (near-zero) zero-shot floor on
    held-out doubles for both functional assays."""
    seen = False
    for assay in ASSAYS:
        paths = _committed(assay)
        if paths is None:
            continue
        seen = True
        _, js = paths
        s = json.loads(js.read_text())
        # zero-shot PLL epistasis is essentially blind on function
        assert abs(s["zeroshot_spearman"]) < 0.1
        # learned head clears it on held-out doubles
        assert s["model1_doubles_spearman"] > s["zeroshot_spearman"]
        assert s["delta_doubles"] > 0
    if not seen:
        pytest.skip("Model 1 artifacts not yet committed (training run pending)")
