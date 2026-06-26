"""Out-of-distribution (OOD) family splits for ProteinGym editing tasks.

The default ``edit_splits.make_dataset_splits`` uses a deterministic hash-based
random split over assays. That is **in-distribution** by protein family — the
train pool may contain assays from the same family as the test pool, so a model
that learns family-specific chemistry will still score well on test.

For the EditGuard paper we want a stricter test of generalization: hold out an
entire protein family and evaluate the prior, the PLM baselines, and the
DPLM-guided sampler on a family it has never seen during training. This module
provides:

- ``FAMILY_ASSIGNMENTS``: a curated mapping from ProteinGym dataset_id to
  protein family. Curation is done by hand from the assay's UniProt /
  description because Pfam queries against ProteinGym CSVs are noisy.
- ``make_family_ood_splits``: build train/val/test splits where one family is
  held out entirely as the OOD test set.
- ``stratify_by_family``: optional per-family stratification of an existing
  split for ablations.

The OOD split sits alongside (not in place of) the default random split so we
can report both the in-distribution and OOD numbers in the headline table.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .edit_splits import is_clinical_dataset, stable_fraction


# Curated family labels for the 11 multi-distance ProteinGym assays.
# Family labels are coarse functional categories chosen to give meaningful
# OOD splits — not Pfam clan names.
FAMILY_ASSIGNMENTS: dict[str, str] = {
    # Fluorescent proteins (GFP-like β-barrel chromophore family).
    "GFP_AEQVI_Sarkisyan_2016": "fluorescent_protein",
    "D7PM05_CLYGR_Somermeyer_2022": "fluorescent_protein",
    "Q8WTC7_9CNID_Somermeyer_2022": "fluorescent_protein",
    "Q6WV12_9MAXI_Somermeyer_2022": "fluorescent_protein",
    # Bacterial / phage toxin–antitoxin and small folded domains.
    "F7YBW8_MESOW_Ding_2023": "toxin_antitoxin",
    "F7YBW8_MESOW_Aakre_2015": "toxin_antitoxin",
    # Yeast / fungal metabolic enzymes and transcription factors.
    "HIS7_YEAST_Pokusaeva_2019": "yeast_metabolism",
    "GCN4_YEAST_Staller_2018": "yeast_transcription_factor",
    # Photoreceptor / signaling LOV domain.
    "PHOT_CHLRE_Chen_2023": "photoreceptor",
    # Viral capsid.
    "CAPSD_AAV2S_Sinai_2021": "viral_capsid",
    # Streptococcal protein G B1 domain (small β-grasp; classic editing target).
    "SPG1_STRSG_Wu_2016": "small_beta_domain",
}


@dataclass(frozen=True)
class FamilyOODConfig:
    """Configuration for a family-OOD split."""

    held_out_family: str
    val_frac_within_train: float = 0.15
    seed: int = 0
    clinical_holdout: bool = True


def get_family(dataset_id: str) -> str | None:
    """Look up the curated family label for a dataset_id.

    Returns None if the dataset is not in our curated map. Callers should
    handle None by either dropping the dataset or assigning a default family.
    """
    return FAMILY_ASSIGNMENTS.get(str(dataset_id))


def known_families() -> tuple[str, ...]:
    """Tuple of all curated family labels, sorted alphabetically."""
    return tuple(sorted(set(FAMILY_ASSIGNMENTS.values())))


def make_family_ood_splits(
    dataset_ids: Iterable[str],
    config: FamilyOODConfig,
) -> pd.DataFrame:
    """Build a train/val/test split that holds out an entire family as test.

    All datasets in ``config.held_out_family`` go to ``test`` (label
    ``"ood_test"`` to distinguish from the default in-distribution test
    split). The remaining datasets are split into train / val using a
    deterministic hash, with the same clinical-keyword holdout policy as
    ``make_dataset_splits``.
    """
    if config.held_out_family not in known_families():
        raise ValueError(
            f"Unknown family {config.held_out_family!r}; "
            f"expected one of {known_families()}"
        )
    rows = []
    for ds_id in sorted(set(map(str, dataset_ids))):
        clinical = is_clinical_dataset(ds_id)
        family = get_family(ds_id)
        if family == config.held_out_family:
            split = "ood_test"
        elif config.clinical_holdout and clinical:
            split = "clinical_test"
        elif family is None:
            # Unfamilied datasets are dropped from both train and val to avoid
            # silently leaking unknown chemistry into the train pool.
            split = "unfamilied"
        else:
            r = stable_fraction(ds_id, seed=config.seed)
            split = "val" if r < config.val_frac_within_train else "train"
        rows.append(
            {
                "dataset_id": ds_id,
                "split": split,
                "family": family or "_unknown",
                "clinical": clinical,
            }
        )
    return pd.DataFrame(rows)


def assert_no_family_leakage(splits: pd.DataFrame, family: str) -> None:
    """Raise if any non-test row carries the held-out ``family`` label.

    Used as a hard guard at the top of OOD evaluation entrypoints.
    """
    if "family" not in splits.columns:
        raise ValueError("splits dataframe must include a 'family' column")
    leaks = splits.loc[
        (splits["family"] == family) & (~splits["split"].isin({"ood_test", "clinical_test", "unfamilied"})),
        ["dataset_id", "split"],
    ]
    if len(leaks) > 0:
        head = ", ".join(f"{r.dataset_id}({r.split})" for r in leaks.head(5).itertuples())
        raise ValueError(
            f"Found {len(leaks)} train/val datasets in held-out family {family!r}: {head}"
        )


def standard_ood_configs(seeds: Iterable[int] = (0,)) -> list[FamilyOODConfig]:
    """The OOD configurations we report in the headline table.

    Holds out each family one at a time. With one seed per family this gives a
    leave-one-family-out evaluation. Add more seeds for the val-split sensitivity
    ablation in the appendix.
    """
    out: list[FamilyOODConfig] = []
    for fam in known_families():
        for s in seeds:
            out.append(FamilyOODConfig(held_out_family=fam, seed=int(s)))
    return out
