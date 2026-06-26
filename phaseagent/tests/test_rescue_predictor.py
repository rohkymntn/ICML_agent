"""Tests for the rescue predictor + conformal rescue set."""
import numpy as np
import pandas as pd
import pytest

from phaseagent.rescue_events import extract_rescue_events
from phaseagent.rescue_predictor import (
    ConformalRescueSet,
    RescuePredictor,
    enumerate_candidate_pairs_for_target,
    featurize_event_table,
    featurize_rescue_pair,
)


def _toy_singles_doubles(seed: int = 0):
    """Larger toy data so the predictor + conformal layer have enough signal."""
    rng = np.random.default_rng(seed)
    rows_singles = []
    rows_doubles = []
    for ds in ("P1", "P2"):
        for i in range(1, 41):
            wt = "ACDEFGHIKLMNPQRSTVWY"[i % 20]
            mut = "ACDEFGHIKLMNPQRSTVWY"[(i + 3) % 20]
            fitness = float(rng.beta(2, 5)) if i % 5 == 0 else float(rng.beta(5, 2))
            rows_singles.append(
                {
                    "dataset_id": ds,
                    "mutation_notation": f"{wt}{i}{mut}",
                    "fitness_norm": fitness,
                }
            )
        for k in range(60):
            i = int(rng.integers(1, 41))
            j = int(rng.integers(1, 41))
            if i == j:
                continue
            wt_i = "ACDEFGHIKLMNPQRSTVWY"[i % 20]
            mut_i = "ACDEFGHIKLMNPQRSTVWY"[(i + 3) % 20]
            wt_j = "ACDEFGHIKLMNPQRSTVWY"[j % 20]
            mut_j = "ACDEFGHIKLMNPQRSTVWY"[(j + 5) % 20]
            base = float(rng.beta(3, 3))
            # Inject a "rescue" signal: when m1 is damaging (fitness < 0.3),
            # combinations with positions far from m1 are slightly higher fitness.
            sep = abs(i - j)
            combined = base + 0.05 * (sep > 10) + 0.05 * float(rng.random() < 0.3)
            rows_doubles.append(
                {
                    "dataset_id": ds,
                    "mutation_notation": f"{wt_i}{i}{mut_i}:{wt_j}{j}{mut_j}",
                    "fitness_norm": float(np.clip(combined, 0, 1)),
                }
            )
    return pd.DataFrame(rows_singles), pd.DataFrame(rows_doubles)


def test_featurize_rescue_pair_returns_39d_vector():
    f = featurize_rescue_pair(
        m1_pos=10, m1_aa="A",
        m2_pos=20, m2_aa="V",
        m1_score=0.1, m2_score=0.5,
        seq_len=100, wt_at_m1="L", wt_at_m2="K",
    )
    assert f.shape == (39,)
    assert f.dtype == np.float32


def test_featurize_uses_sequence_separation():
    near = featurize_rescue_pair(m1_pos=10, m1_aa="A", m2_pos=12, m2_aa="V",
                                 m1_score=0.1, m2_score=0.5, seq_len=100)
    far = featurize_rescue_pair(m1_pos=10, m1_aa="A", m2_pos=80, m2_aa="V",
                                m1_score=0.1, m2_score=0.5, seq_len=100)
    # Sequence separation feature (index 21) should differ.
    assert near[21] < far[21]
    # within-8 indicator (index 23) should fire for near, not far.
    assert near[23] == 1.0 and far[23] == 0.0


def test_featurize_event_table_handles_empty():
    empty = pd.DataFrame(columns=[
        "dataset_id", "m1_notation", "m2_notation", "m1_pos", "m2_pos",
        "m1_aa", "m2_aa", "m1_score", "m2_score",
    ])
    out = featurize_event_table(empty)
    assert out.shape == (0, 39)


def test_rescue_predictor_fits_and_scores():
    singles, doubles = _toy_singles_doubles()
    events = extract_rescue_events(doubles, singles, only_damaging_first=True)
    if len(events) < 20:
        pytest.skip("toy data didn't yield enough rescue events")
    model = RescuePredictor(n_estimators=30, max_depth=3).fit(events)
    scores = model.predict_rescue_proba(events.head(10))
    assert scores.shape == (10,)
    assert scores.min() >= 0.0 and scores.max() <= 1.0


def test_rescue_predictor_unfitted_raises():
    df = pd.DataFrame(columns=[
        "dataset_id", "m1_notation", "m2_notation", "m1_pos", "m2_pos",
        "m1_aa", "m2_aa", "m1_score", "m2_score",
    ])
    with pytest.raises(ValueError, match="not fitted"):
        RescuePredictor().predict_rescue_proba(df)


def test_conformal_rescue_set_calibration_requires_min_data():
    singles, doubles = _toy_singles_doubles()
    events = extract_rescue_events(doubles, singles, only_damaging_first=False)
    if len(events) < 20:
        pytest.skip("toy data did not yield enough events")
    model = RescuePredictor(n_estimators=20).fit(events.iloc[: max(20, len(events) // 2)])
    cset = ConformalRescueSet(predictor=model, alpha=0.2)
    too_small = events.head(5)
    with pytest.raises(ValueError, match="at least 30"):
        cset.calibrate(too_small)


def test_conformal_rescue_set_select_returns_top_k():
    singles, doubles = _toy_singles_doubles()
    events = extract_rescue_events(doubles, singles, only_damaging_first=True)
    if len(events) < 60:
        pytest.skip("not enough toy events for calibration")
    train = events.iloc[: len(events) // 2]
    calib = events.iloc[len(events) // 2 :]
    model = RescuePredictor(n_estimators=20).fit(train)
    cset = ConformalRescueSet(predictor=model, alpha=0.2).calibrate(calib)
    out = cset.select(events.head(50), max_k=5)
    assert len(out) <= 5
    assert "rescue_score" in out.columns
    assert "above_calibrated_threshold" in out.columns


def test_enumerate_candidate_pairs_creates_valid_frame():
    seq = "M" * 30 + "ACDEKHQ" + "M" * 13  # length 50
    cands = enumerate_candidate_pairs_for_target(
        m1_notation="A31C", m1_score=0.1, seq=seq, dataset_id="P1",
    )
    # Self-position excluded; one row per (other position, candidate AA).
    assert len(cands) > 0
    assert (cands["m2_pos"] != 31).all()
    # Forbidden residues should not appear as m2.
    assert not cands["m2_aa"].isin({"C", "M", "W", "P"}).any()
    # m1 features carry through.
    assert (cands["m1_notation"] == "A31C").all()
    assert (cands["m1_score"] == 0.1).all()


def test_enumerate_then_predict_end_to_end():
    singles, doubles = _toy_singles_doubles()
    events = extract_rescue_events(doubles, singles, only_damaging_first=True)
    if len(events) < 20:
        pytest.skip("toy data didn't yield enough rescue events")
    model = RescuePredictor(n_estimators=20).fit(events)

    seq = "ACDEFGHIKLMNPQRSTVWY" * 5  # length 100
    candidates = enumerate_candidate_pairs_for_target(
        m1_notation="A21V", m1_score=0.1, seq=seq, dataset_id="NEW_PROT",
    )
    scores = model.predict_rescue_proba(candidates)
    assert scores.shape == (len(candidates),)
    assert scores.min() >= 0.0 and scores.max() <= 1.0
