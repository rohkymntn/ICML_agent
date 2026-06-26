"""Smoke-test the Model 1 artifact pipeline (DoD #3) without GPU / real data.

The real `feat_<assay>.npz` is GPU-only and not committed (it lives on the Modal
volume), so we exercise `run_model1` on a tiny synthetic feature file that uses
the SAME `meta` schema the extractor writes
(`[eps, pll, mut_i, mut_j, wt_i, wt_j, pos_i, pos_j]`, see
`modal_app_v2.py::extract_function_features`). The point is to guard the write
path and the summary-from-CSV recompute so that when the real features land the
committed artifact is self-consistent and traceable.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_model1 import run_model1, summarize  # noqa: E402


def _synthetic_feat(path, n=240, D=32, seed=0):
    rng = np.random.default_rng(seed)
    Hi = rng.standard_normal((n, D)).astype(np.float16)
    Hj = rng.standard_normal((n, D)).astype(np.float16)
    # eps carries real signal from the hidden states so the head can learn it;
    # pll is a weak/noisy view of eps (the zero-shot baseline).
    eps = 0.6 * Hi[:, 0].astype(np.float32) + 0.6 * Hj[:, 0].astype(np.float32) \
        + 0.3 * rng.standard_normal(n).astype(np.float32)
    pll = 0.3 * eps + rng.standard_normal(n).astype(np.float32)
    ids = rng.integers(0, 20, size=(n, 4)).astype(np.float32)   # mut_i, mut_j, wt_i, wt_j
    pos = rng.integers(0, 10, size=(n, 2)).astype(np.float32)   # >=5 groups for GroupKFold
    meta = np.column_stack([eps, pll, ids, pos]).astype(np.float32)
    np.savez(path, Hi=Hi, Hj=Hj, meta=meta)


def test_run_model1_writes_self_consistent_artifacts(tmp_path):
    feat = tmp_path / "feat_SYN.npz"
    _synthetic_feat(feat)
    summ = run_model1(str(feat), "SYN", out=str(tmp_path), epochs=3)

    csv = tmp_path / "model1_oof_SYN.csv"
    js = tmp_path / "model1_SYN_summary.json"
    assert csv.exists() and js.exists()

    df = pd.read_csv(csv)
    assert len(df) == summ["n_doubles"] == 240
    assert set(df.columns) == {"y_true", "pll", "oof_doubles", "oof_position", "pos_i", "pos_j"}
    assert df.notna().all().all()  # every row got an out-of-fold prediction

    # the committed summary JSON must be reproducible from the raw OOF CSV
    on_disk = json.loads(js.read_text())
    recomputed = summarize(df, "SYN", summ["dim"], summ["use_shift"])
    for k, v in recomputed.items():
        assert on_disk[k] == v, f"summary/csv drift on {k}"

    # sanity: deltas are exactly model1 - zeroshot (the WIN margin the paper reports)
    assert abs(summ["delta_doubles"] - (summ["model1_doubles_spearman"] - summ["zeroshot_spearman"])) < 1e-9
    assert abs(summ["delta_position"] - (summ["model1_position_spearman"] - summ["zeroshot_spearman"])) < 1e-9
