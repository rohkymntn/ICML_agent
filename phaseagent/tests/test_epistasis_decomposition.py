"""Unit tests for the additive / specific-epistasis decomposition.

These check the residual arithmetic and per-protein atlas on a tiny hand-built
frame in the canonical Megascale schema. No model training, no synthetic
science -- just verifying eps = obs - sum(singles) and coverage handling.
"""
import numpy as np
import pandas as pd

from phaseagent.epistasis_decomposition import (
    decompose_multimutants,
    epistasis_gate_summary,
    protein_epistasis_atlas,
)


def _toy_frame() -> pd.DataFrame:
    rows = [
        # domain A singles
        ("A", "P1A", 1, -1.0),
        ("A", "K2E", 1, -0.5),
        ("A", "L3V", 1, 0.2),
        # domain A doubles
        ("A", "P1A:K2E", 2, -2.0),   # additive -1.5 -> eps -0.5
        ("A", "P1A:L3V", 2, -0.7),   # additive -0.8 -> eps +0.1
        ("A", "K2E:M4A", 2, -1.0),   # M4A single missing -> eps NaN
        # domain B singles + a double with zero epistasis
        ("B", "A5G", 1, 0.4),
        ("B", "T6S", 1, -0.3),
        ("B", "A5G:T6S", 2, 0.1),    # additive 0.1 -> eps 0.0
    ]
    return pd.DataFrame(
        rows, columns=["dataset_id", "mutation_notation", "mutation_distance", "ddG_ML"]
    )


def test_epsilon_arithmetic_and_coverage():
    df = _toy_frame()
    dec = decompose_multimutants(df)
    dec = dec.set_index("mutation_notation")

    # Covered doubles: eps = observed - sum(constituent singles).
    assert np.isclose(dec.loc["P1A:K2E", "epsilon"], -2.0 - (-1.0 + -0.5))   # -0.5
    assert np.isclose(dec.loc["P1A:L3V", "epsilon"], -0.7 - (-1.0 + 0.2))    # +0.1
    assert np.isclose(dec.loc["A5G:T6S", "epsilon"], 0.1 - (0.4 + -0.3))     # 0.0

    # Missing constituent single -> additive + eps are NaN (no leakage/guess).
    assert not np.isfinite(dec.loc["K2E:M4A", "ddG_additive"])
    assert not np.isfinite(dec.loc["K2E:M4A", "epsilon"])

    # n_edits is the substitution count.
    assert int(dec.loc["P1A:K2E", "n_edits"]) == 2


def test_atlas_additive_r2_and_share():
    df = _toy_frame()
    atlas = protein_epistasis_atlas(df, min_covered=2)
    a = atlas.set_index("dataset_id")

    # Domain A has 2 covered doubles (the M4A one is dropped).
    assert int(a.loc["A", "n_covered"]) == 2
    # epsilon for A's covered doubles = {-0.5, +0.1}; std(ddof=1) of those.
    assert np.isclose(a.loc["A", "epistasis_std"], np.std([-0.5, 0.1], ddof=1))
    # mean eps captures directional bias.
    assert np.isclose(a.loc["A", "mean_eps"], np.mean([-0.5, 0.1]))


def test_gate_summary_keys():
    df = _toy_frame()
    atlas = protein_epistasis_atlas(df, min_covered=1)
    summary = epistasis_gate_summary(atlas)
    for key in (
        "n_proteins",
        "median_additive_r2",
        "median_epistasis_std_kcal",
        "median_epistasis_var_share",
    ):
        assert key in summary
    assert summary["n_proteins"] >= 1
