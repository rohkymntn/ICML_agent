"""Math tests for the sigmoid fitter — uses deterministic V(d) tables, not synthetic landscapes."""
import numpy as np
import pandas as pd

from phaseagent.phase import (
    compute_viability_by_distance,
    estimate_susceptibility,
    fit_phase_boundary,
    sigmoid,
)


def _v_table_from_truth(dc=5.0, alpha=1.0, n_per_d=200, max_d=12):
    """Build a V(d) table by evaluating the sigmoid exactly — pure math, no random sampling."""
    d = np.arange(0, max_d + 1)
    v = sigmoid(d, dc, alpha)
    df = pd.DataFrame({
        "mutation_distance": d,
        "n": n_per_d,
        "viable_count": (v * n_per_d).round().astype(int),
        "viability_density": v,
        "stderr": np.sqrt(v * (1 - v) / n_per_d),
        "sufficient": True,
    })
    return df


def test_sigmoid_fit_recovers_dc():
    v = _v_table_from_truth(dc=5.0, alpha=1.0)
    fit = fit_phase_boundary(v)
    assert fit["fit_success"]
    assert abs(fit["dc"] - 5.0) < 0.5
    assert abs(fit["alpha"] - 1.0) < 0.3


def test_sigmoid_fit_high_r2_on_clean_signal():
    v = _v_table_from_truth(dc=4.0, alpha=2.0)
    fit = fit_phase_boundary(v)
    assert fit["r2"] > 0.95


def test_susceptibility_peak_near_dc():
    v = _v_table_from_truth(dc=5.0, alpha=1.0)
    chi = estimate_susceptibility(v)
    peak_d = float(chi.loc[chi["susceptibility"].idxmax(), "mutation_distance"])
    assert abs(peak_d - 5.0) <= 1.0


def test_compute_viability_by_distance_basic():
    rng = np.random.default_rng(0)
    rows = []
    for d in range(8):
        for _ in range(50):
            rows.append({"mutation_distance": d, "viable": int(rng.random() < (1.0 / (1 + np.exp(d - 4))))})
    df = pd.DataFrame(rows)
    out = compute_viability_by_distance(df, min_count_per_distance=10)
    assert (out["sufficient"]).all()
    assert "viability_density" in out.columns


def test_fit_returns_nan_on_insufficient():
    df = pd.DataFrame({
        "mutation_distance": [0, 1],
        "n": [5, 5],
        "viable_count": [5, 4],
        "viability_density": [1.0, 0.8],
        "stderr": [0.0, 0.18],
        "sufficient": [False, False],
    })
    fit = fit_phase_boundary(df)
    assert not fit["fit_success"]
    assert np.isnan(fit["dc"])
