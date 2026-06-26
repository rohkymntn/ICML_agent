"""Tests for the few-shot subsampling module."""
import numpy as np
import pandas as pd
import pytest

from phaseagent.few_shot import (
    FewShotConfig,
    active_subsample,
    random_subsample,
    standard_few_shot_grid,
    stratified_subsample,
    subsample,
)


def _toy_pool(n_per_ds: int = 200, seed: int = 0) -> pd.DataFrame:
    """Two-dataset toy DMS pool with realistic columns."""
    rng = np.random.default_rng(seed)
    rows = []
    for ds in ("A", "B"):
        for i in range(n_per_ds):
            d = int(rng.integers(1, 5))
            rows.append(
                {
                    "dataset_id": ds,
                    "mutation_notation": f"M{i+1}A",
                    "mutation_distance": d,
                    "fitness_norm": float(rng.random()),
                    "viable": int(rng.random() > 0.5),
                }
            )
    return pd.DataFrame(rows)


def test_random_subsample_returns_exact_count():
    df = _toy_pool()
    out = random_subsample(df, n_train=100, seed=0)
    assert len(out) == 100
    assert set(out.columns) == set(df.columns)


def test_random_subsample_is_deterministic():
    df = _toy_pool()
    a = random_subsample(df, n_train=50, seed=42)
    b = random_subsample(df, n_train=50, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_random_subsample_returns_full_when_requested_too_many():
    df = _toy_pool()
    out = random_subsample(df, n_train=10_000, seed=0)
    assert len(out) == len(df)


def test_stratified_subsample_preserves_dataset_balance():
    df = _toy_pool(n_per_ds=200)
    out = stratified_subsample(df, n_train=100, seed=0, n_distance_bins=4)
    counts = out["dataset_id"].value_counts()
    # Each dataset has equal pool size, so subsample should be roughly balanced.
    assert abs(counts.get("A", 0) - counts.get("B", 0)) <= 4
    assert len(out) == 100


def test_stratified_subsample_preserves_distance_diversity():
    df = _toy_pool(n_per_ds=400)
    out = stratified_subsample(df, n_train=120, seed=0, n_distance_bins=4)
    # Should retain variants from at least 3 of 4 distance bins.
    assert out["mutation_distance"].nunique() >= 3


def test_active_subsample_reaches_close_to_target_size():
    df = _toy_pool(n_per_ds=200)
    out = active_subsample(df, n_train=100, seed=0, initial_frac=0.2, n_rounds=4)
    # Active loop may be off by ±1 due to per-round flooring; that's acceptable.
    assert abs(len(out) - 100) <= 4


def test_subsample_dispatches_on_strategy():
    df = _toy_pool()
    cfg_r = FewShotConfig(n_train=50, strategy="random", seed=0)
    cfg_s = FewShotConfig(n_train=50, strategy="stratified", seed=0)
    cfg_a = FewShotConfig(n_train=80, strategy="active", seed=0, n_rounds=2)
    assert len(subsample(df, cfg_r)) == 50
    assert len(subsample(df, cfg_s)) == 50
    assert abs(len(subsample(df, cfg_a)) - 80) <= 4


def test_unknown_strategy_raises():
    df = _toy_pool()
    cfg = FewShotConfig(n_train=50, strategy="bogus", seed=0)
    with pytest.raises(ValueError, match="Unknown few-shot strategy"):
        subsample(df, cfg)


def test_standard_grid_matches_protocol():
    grid = standard_few_shot_grid()
    # 4 sizes × 3 strategies × 5 seeds = 60.
    assert len(grid) == 60
    sizes = {c.n_train for c in grid}
    strategies = {c.strategy for c in grid}
    assert sizes == {50, 100, 200, 500}
    assert strategies == {"random", "stratified", "active"}


def test_subsample_does_not_leak_across_seeds():
    df = _toy_pool()
    a = random_subsample(df, n_train=50, seed=0)
    b = random_subsample(df, n_train=50, seed=1)
    # Different seeds should yield different (but overlapping) samples.
    common = set(a["mutation_notation"]) & set(b["mutation_notation"])
    assert len(common) < 50, "different seeds returned identical subsamples"
