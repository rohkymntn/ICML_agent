"""Tests for additive multi-mutant baseline + conformal selective prediction."""
import numpy as np
import pandas as pd
import pytest

from phaseagent.additive_baseline import (
    AdditivePredictor,
    epistasis_signal,
    epistasis_slice_metrics,
    stabilizing_pair_recall,
)
from phaseagent.conformal import ConformalRegressor, selective_top_k
from phaseagent.megascale import normalize_mutation_notation


# ------------------------- Additive baseline ------------------------- #

def _toy_singles() -> pd.DataFrame:
    """A small single-mutant table for two proteins."""
    rows = [
        {"dataset_id": "P1", "mutation_notation": "A1G", "fitness_norm": 0.9},
        {"dataset_id": "P1", "mutation_notation": "A2C", "fitness_norm": 0.4},
        {"dataset_id": "P1", "mutation_notation": "A3T", "fitness_norm": 0.6},
        {"dataset_id": "P2", "mutation_notation": "M1R", "fitness_norm": 0.2},
        {"dataset_id": "P2", "mutation_notation": "M2K", "fitness_norm": 0.7},
    ]
    return pd.DataFrame(rows)


def _toy_multis() -> pd.DataFrame:
    rows = [
        {"dataset_id": "P1", "mutation_notation": "A1G:A2C", "fitness_norm": 1.0},  # additive: 1.3
        {"dataset_id": "P1", "mutation_notation": "A2C:A3T", "fitness_norm": 1.5},  # additive: 1.0
        {"dataset_id": "P2", "mutation_notation": "M1R:M2K", "fitness_norm": 0.8},  # additive: 0.9
        {"dataset_id": "P2", "mutation_notation": "M1R:M99X", "fitness_norm": 0.5},  # missing M99X
    ]
    return pd.DataFrame(rows)


def test_additive_predictor_sums_singles():
    pred = AdditivePredictor().fit(_toy_singles())
    multis = _toy_multis()
    p = pred.predict(multis)
    # Row 0: A1G(0.9) + A2C(0.4) = 1.3
    assert pytest.approx(p[0], rel=1e-9) == 1.3
    # Row 1: A2C(0.4) + A3T(0.6) = 1.0
    assert pytest.approx(p[1], rel=1e-9) == 1.0
    # Row 2: M1R(0.2) + M2K(0.7) = 0.9
    assert pytest.approx(p[2], rel=1e-9) == 0.9
    # Row 3: missing component -> NaN
    assert np.isnan(p[3])


def test_additive_predictor_coverage():
    pred = AdditivePredictor().fit(_toy_singles())
    cov = pred.coverage(_toy_multis())
    # 3 of 4 rows fully predictable.
    assert pytest.approx(cov, rel=1e-9) == 0.75


def test_additive_predictor_unfitted_raises():
    pred = AdditivePredictor()
    with pytest.raises(ValueError, match="not fitted"):
        pred.predict(_toy_multis())


def test_additive_predictor_per_dataset_lookup():
    """Same single-mutant token in two proteins should resolve per-dataset."""
    df = pd.DataFrame(
        [
            {"dataset_id": "P1", "mutation_notation": "A1G", "fitness_norm": 0.1},
            {"dataset_id": "P2", "mutation_notation": "A1G", "fitness_norm": 0.9},
        ]
    )
    pred = AdditivePredictor().fit(df)
    test = pd.DataFrame(
        [
            {"dataset_id": "P1", "mutation_notation": "A1G"},
            {"dataset_id": "P2", "mutation_notation": "A1G"},
        ]
    )
    p = pred.predict(test)
    assert p[0] == pytest.approx(0.1)
    assert p[1] == pytest.approx(0.9)


def test_epistasis_signal_zero_when_perfectly_additive():
    pred = AdditivePredictor().fit(_toy_singles())
    multis = pd.DataFrame(
        [{"dataset_id": "P1", "mutation_notation": "A1G:A2C", "fitness_norm": 1.3}]
    )
    eps = epistasis_signal(multis, pred)
    assert eps.iloc[0] == pytest.approx(0.0)


def test_epistasis_slice_metrics_returns_expected_keys():
    pred = AdditivePredictor().fit(_toy_singles())
    multis = _toy_multis().dropna(subset=["fitness_norm"]).iloc[:3]
    add = pred.predict(multis)
    # A model that just returns observed values should beat additive on the slice.
    model = multis["fitness_norm"].to_numpy(dtype=float)
    out = epistasis_slice_metrics(multis, model, add, epistasis_threshold=0.1)
    assert {"n", "overall_model", "overall_additive", "slice_n", "slice_model", "slice_additive", "slice_delta"} <= set(out.keys())


def test_stabilizing_pair_recall_perfect_for_oracle_model():
    df = pd.DataFrame({"fitness_norm": np.linspace(0, 1, 100)})
    perfect = df["fitness_norm"].to_numpy()
    r = stabilizing_pair_recall(df, perfect, stabilizing_threshold=0.75, k=25)
    # Oracle model picks top-25 = exactly the truly stabilizing set.
    assert r == pytest.approx(1.0)


# ------------------------- Megascale notation parsing ------------------------- #

def test_normalize_mutation_notation_one_letter():
    assert normalize_mutation_notation("P123A") == "P123A"


def test_normalize_mutation_notation_three_letter():
    assert normalize_mutation_notation("PRO123ALA") == "P123A"


def test_normalize_mutation_notation_multi_mutant():
    assert normalize_mutation_notation("P123A:K456E") == "P123A:K456E"
    assert normalize_mutation_notation("P123A,K456E") == "P123A:K456E"


def test_normalize_mutation_notation_returns_none_on_garbage():
    assert normalize_mutation_notation("not a mutation") is None
    assert normalize_mutation_notation("") is None
    assert normalize_mutation_notation(None) is None


# ------------------------- Conformal regressor ------------------------- #

class _OracleRegressor:
    """Trivial regressor that returns the true label - useful for testing."""

    def __init__(self, noise: float = 0.0, seed: int = 0):
        self.noise = float(noise)
        self.rng = np.random.default_rng(seed)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        y = pd.to_numeric(df["fitness_norm"], errors="coerce").to_numpy(dtype=float)
        return y + self.rng.normal(0.0, self.noise, size=len(y))


def _toy_calib_test(n_per_group: int = 60, noise: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for ds in ("P1", "P2", "P3"):
        for _ in range(n_per_group):
            rows.append({"dataset_id": ds, "fitness_norm": float(rng.random())})
    df = pd.DataFrame(rows)
    calib = df.sample(frac=0.5, random_state=seed).reset_index(drop=True)
    test = df.drop(calib.index, errors="ignore").reset_index(drop=True)
    return calib, test


def test_conformal_calibrate_then_coverage_close_to_nominal():
    calib, test = _toy_calib_test(n_per_group=120, noise=0.1, seed=0)
    base = _OracleRegressor(noise=0.1, seed=42)
    cr = ConformalRegressor(base, alpha=0.1, group_col="dataset_id")
    cr.calibrate(calib)
    cov = cr.coverage(test)
    # Should be within 5 points of the nominal 0.9 with this much data.
    assert 0.85 <= cov["global"] <= 0.97


def test_conformal_predict_interval_returns_two_arrays():
    calib, test = _toy_calib_test(n_per_group=80)
    base = _OracleRegressor(noise=0.1, seed=0)
    cr = ConformalRegressor(base, alpha=0.1, group_col="dataset_id").calibrate(calib)
    lo, hi = cr.predict_interval(test)
    assert lo.shape == hi.shape == (len(test),)
    assert np.all(lo <= hi)


def test_conformal_calibrate_too_small_raises():
    df = pd.DataFrame({"dataset_id": ["A"] * 5, "fitness_norm": [0.1, 0.2, 0.3, 0.4, 0.5]})
    base = _OracleRegressor()
    with pytest.raises(ValueError, match="at least 30"):
        ConformalRegressor(base).calibrate(df)


def test_selective_top_k_skips_abstained():
    df = pd.DataFrame({"id": list(range(5)), "fitness_norm": [0.1, 0.2, 0.3, 0.4, 0.5]})
    scores = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
    abstain = np.array([True, False, False, False, False])
    out = selective_top_k(df, scores, abstain, k=2)
    # Top-1 (id=0) is abstained, so we get id=1, id=2 (scores 0.4, 0.3).
    assert list(out["id"]) == [1, 2]


def test_selective_top_k_all_abstained_returns_empty():
    df = pd.DataFrame({"id": list(range(3)), "fitness_norm": [0.1, 0.2, 0.3]})
    scores = np.array([0.3, 0.2, 0.1])
    abstain = np.array([True, True, True])
    out = selective_top_k(df, scores, abstain, k=2)
    assert len(out) == 0
