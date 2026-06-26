"""Tests for the rescue-event extractor."""
import numpy as np
import pandas as pd

from phaseagent.rescue_events import (
    DEFAULT_DAMAGING_QUANTILE,
    derive_thresholds,
    extract_rescue_events,
    summarize_rescue_events,
)


def _toy_singles_doubles():
    """One protein, 6 singles + 4 doubles with mixed rescue outcomes."""
    singles = pd.DataFrame(
        [
            {"dataset_id": "P1", "mutation_notation": "A1G", "fitness_norm": 0.9},
            {"dataset_id": "P1", "mutation_notation": "A2C", "fitness_norm": 0.85},
            {"dataset_id": "P1", "mutation_notation": "A3D", "fitness_norm": 0.10},  # damaging
            {"dataset_id": "P1", "mutation_notation": "A4E", "fitness_norm": 0.70},
            {"dataset_id": "P1", "mutation_notation": "A5F", "fitness_norm": 0.05},  # damaging
            {"dataset_id": "P1", "mutation_notation": "A6G", "fitness_norm": 0.50},
        ]
    )
    doubles = pd.DataFrame(
        [
            # damaging A3D + restorative A4E -> "full" rescue (combined > viable, m1 < viable)
            {"dataset_id": "P1", "mutation_notation": "A3D:A4E", "fitness_norm": 0.65},
            # damaging A5F + neutral A6G -> "partial" (slight bump)
            {"dataset_id": "P1", "mutation_notation": "A5F:A6G", "fitness_norm": 0.30},
            # damaging A3D + damaging A5F -> still no rescue (combined ~ same)
            {"dataset_id": "P1", "mutation_notation": "A3D:A5F", "fitness_norm": 0.05},
            # neutral pair -> not classified as rescue (m1 not damaging)
            {"dataset_id": "P1", "mutation_notation": "A1G:A2C", "fitness_norm": 0.95},
        ]
    )
    return singles, doubles


def test_derive_thresholds_returns_quantiles():
    singles, _ = _toy_singles_doubles()
    thr = derive_thresholds(singles)
    assert thr.n_singles == 6
    # damaging = 25th percentile of the toy fitness distribution
    assert 0.05 <= thr.damaging <= 0.20
    assert thr.viable > thr.damaging
    assert thr.wt_equivalent > thr.viable


def test_extract_rescue_events_classifies_correctly():
    singles, doubles = _toy_singles_doubles()
    events = extract_rescue_events(doubles, singles, only_damaging_first=True)
    classes = events["rescue_class"].value_counts().to_dict()
    # We expect at least one full rescue and one partial.
    assert "full" in classes
    assert "partial" in classes


def test_extract_rescue_events_emits_both_orderings_for_two_damaging():
    singles, doubles = _toy_singles_doubles()
    events = extract_rescue_events(doubles, singles, only_damaging_first=True)
    # A3D:A5F is two damaging singles, both orderings should appear when both are damaging.
    a3d_first = events[(events["m1_notation"] == "A3D") & (events["m2_notation"] == "A5F")]
    a5f_first = events[(events["m1_notation"] == "A5F") & (events["m2_notation"] == "A3D")]
    assert len(a3d_first) == 1
    assert len(a5f_first) == 1


def test_extract_rescue_events_skips_non_damaging_first_when_filtered():
    singles, doubles = _toy_singles_doubles()
    events = extract_rescue_events(doubles, singles, only_damaging_first=True)
    # The neutral A1G:A2C pair has both halves above damaging threshold -> not in output.
    neutral = events[
        events["m1_notation"].isin(["A1G", "A2C"])
        & events["m2_notation"].isin(["A1G", "A2C"])
    ]
    assert len(neutral) == 0


def test_extract_rescue_events_keeps_all_pairs_when_unfiltered():
    singles, doubles = _toy_singles_doubles()
    events = extract_rescue_events(doubles, singles, only_damaging_first=False)
    # 4 doubles × 2 orderings = 8 candidate rows (assuming all single-mutant lookups succeed).
    assert len(events) == 8


def test_extract_handles_missing_single_lookups():
    # Add a double whose components aren't measured as singles.
    singles, doubles = _toy_singles_doubles()
    doubles_extended = pd.concat(
        [
            doubles,
            pd.DataFrame(
                [{"dataset_id": "P1", "mutation_notation": "A99X:A100Y", "fitness_norm": 0.5}]
            ),
        ],
        ignore_index=True,
    )
    events = extract_rescue_events(
        doubles_extended, singles, only_damaging_first=False, require_both_singles_measured=True
    )
    # The unmeasured pair should be silently dropped.
    assert not (events["m1_notation"].isin(["A99X", "A100Y"])).any()


def test_summarize_rescue_events_returns_per_protein_table():
    singles, doubles = _toy_singles_doubles()
    events = extract_rescue_events(doubles, singles)
    summary = summarize_rescue_events(events)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["dataset_id"] == "P1"
    assert row["n_pairs"] >= 2
    # rescue_rate should be a valid probability.
    assert 0.0 <= row["rescue_rate"] <= 1.0


def test_summarize_empty_events_returns_empty_df():
    summary = summarize_rescue_events(pd.DataFrame())
    assert summary.empty


def test_extract_handles_multi_protein_doubles_correctly():
    # Two proteins with enough singles to derive thresholds; per-protein lookup
    # must isolate them. P1's A1G is damaging; P2's A1G is healthy.
    singles = pd.DataFrame(
        [
            {"dataset_id": "P1", "mutation_notation": "A1G", "fitness_norm": 0.10},
            {"dataset_id": "P1", "mutation_notation": "A2C", "fitness_norm": 0.85},
            {"dataset_id": "P1", "mutation_notation": "A3D", "fitness_norm": 0.90},
            {"dataset_id": "P1", "mutation_notation": "A4E", "fitness_norm": 0.80},
            {"dataset_id": "P2", "mutation_notation": "A1G", "fitness_norm": 0.90},
            {"dataset_id": "P2", "mutation_notation": "A2C", "fitness_norm": 0.95},
            {"dataset_id": "P2", "mutation_notation": "A3D", "fitness_norm": 0.92},
            {"dataset_id": "P2", "mutation_notation": "A4E", "fitness_norm": 0.88},
        ]
    )
    doubles = pd.DataFrame(
        [
            {"dataset_id": "P1", "mutation_notation": "A1G:A2C", "fitness_norm": 0.7},
            {"dataset_id": "P2", "mutation_notation": "A1G:A2C", "fitness_norm": 1.0},
        ]
    )
    events = extract_rescue_events(doubles, singles, only_damaging_first=True)
    # P1 has a damaging A1G; P2 doesn't (A1G is fitness 0.9 there).
    p1_events = events[events["dataset_id"] == "P1"]
    p2_events = events[events["dataset_id"] == "P2"]
    assert len(p1_events) >= 1
    # P2 has no damaging singles so should produce no rescue events under the filter.
    assert len(p2_events) == 0


def test_full_rescue_requires_damaging_first_and_viable_combined():
    singles = pd.DataFrame(
        [
            {"dataset_id": "P1", "mutation_notation": f"A{i}A", "fitness_norm": 0.05 if i == 3 else 0.85}
            for i in range(1, 11)
        ]
    )
    doubles = pd.DataFrame(
        [{"dataset_id": "P1", "mutation_notation": "A3A:A4A", "fitness_norm": 0.95}]
    )
    events = extract_rescue_events(doubles, singles, only_damaging_first=True)
    # m1=A3A is damaging; combined=0.95 > viable; should be a "full" or "super" rescue.
    rescue_row = events[events["m1_notation"] == "A3A"]
    assert len(rescue_row) == 1
    assert rescue_row.iloc[0]["rescue_class"] in {"full", "super"}
