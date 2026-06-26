"""Tests for the OOD family-split module."""
import pandas as pd
import pytest

from phaseagent.family_splits import (
    FAMILY_ASSIGNMENTS,
    FamilyOODConfig,
    assert_no_family_leakage,
    get_family,
    known_families,
    make_family_ood_splits,
    standard_ood_configs,
)


PROTEINGYM_11 = list(FAMILY_ASSIGNMENTS.keys())


def test_all_curated_assays_have_family():
    for ds in FAMILY_ASSIGNMENTS:
        assert get_family(ds) is not None
        assert get_family(ds) == FAMILY_ASSIGNMENTS[ds]


def test_unknown_dataset_returns_none():
    assert get_family("FOO_BAR_2099") is None


def test_known_families_returns_sorted_unique():
    fams = known_families()
    assert list(fams) == sorted(set(FAMILY_ASSIGNMENTS.values()))


def test_make_family_ood_splits_holds_out_entire_family():
    cfg = FamilyOODConfig(held_out_family="fluorescent_protein", seed=0)
    splits = make_family_ood_splits(PROTEINGYM_11, cfg)
    fp_rows = splits[splits["family"] == "fluorescent_protein"]
    # All four FP assays should land in ood_test.
    assert len(fp_rows) == 4
    assert set(fp_rows["split"]) == {"ood_test"}


def test_make_family_ood_splits_keeps_other_families_in_train_or_val():
    cfg = FamilyOODConfig(held_out_family="fluorescent_protein", seed=0)
    splits = make_family_ood_splits(PROTEINGYM_11, cfg)
    non_fp = splits[splits["family"] != "fluorescent_protein"]
    # Every non-FP, non-clinical, non-unknown family row should be train or val.
    actual = set(non_fp.loc[~non_fp["family"].isin({"_unknown"}), "split"])
    assert actual.issubset({"train", "val", "clinical_test", "unfamilied"})


def test_unknown_family_raises():
    cfg = FamilyOODConfig(held_out_family="not_a_real_family", seed=0)
    with pytest.raises(ValueError, match="Unknown family"):
        make_family_ood_splits(PROTEINGYM_11, cfg)


def test_assert_no_family_leakage_passes_clean_split():
    cfg = FamilyOODConfig(held_out_family="toxin_antitoxin", seed=0)
    splits = make_family_ood_splits(PROTEINGYM_11, cfg)
    # Should not raise.
    assert_no_family_leakage(splits, "toxin_antitoxin")


def test_assert_no_family_leakage_catches_corruption():
    cfg = FamilyOODConfig(held_out_family="toxin_antitoxin", seed=0)
    splits = make_family_ood_splits(PROTEINGYM_11, cfg)
    # Manually corrupt: relabel one held-out row as train.
    corrupted = splits.copy()
    mask = corrupted["family"] == "toxin_antitoxin"
    if mask.any():
        idx = corrupted.index[mask][0]
        corrupted.loc[idx, "split"] = "train"
        with pytest.raises(ValueError, match="train/val datasets in held-out family"):
            assert_no_family_leakage(corrupted, "toxin_antitoxin")


def test_standard_ood_configs_covers_every_family():
    cfgs = standard_ood_configs(seeds=(0,))
    families = {c.held_out_family for c in cfgs}
    assert families == set(known_families())
    assert len(cfgs) == len(known_families())


def test_unfamilied_dataset_does_not_leak_into_train():
    """Datasets not in the curated map should be dropped, not silently added."""
    cfg = FamilyOODConfig(held_out_family="fluorescent_protein", seed=0)
    splits = make_family_ood_splits(PROTEINGYM_11 + ["MYSTERY_PROTEIN_9999"], cfg)
    mystery = splits[splits["dataset_id"] == "MYSTERY_PROTEIN_9999"]
    assert len(mystery) == 1
    assert mystery.iloc[0]["split"] == "unfamilied"
