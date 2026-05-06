"""Deterministic protein-level splits for EditGuard experiments."""
from __future__ import annotations

import hashlib
from typing import Iterable

import pandas as pd


CLINICAL_KEYWORDS = (
    "BRCA",
    "P53",
    "PTEN",
    "MSH2",
    "TPMT",
    "VKOR",
    "NUD15",
    "MTHR",
    "CBS",
    "OTC",
    "SCN5A",
    "KCNH2",
    "KCNE1",
    "KCNJ2",
    "CP2C9",
    "NPC1",
    "PRKN",
    "S22A1",
    "SC6A4",
    "PPARG",
    "MET",
    "ERBB",
    "RASH",
    "RASK",
    "RAF",
)


def stable_fraction(value: str, seed: int = 0) -> float:
    key = f"{seed}:{value}".encode()
    digest = hashlib.sha256(key).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def is_clinical_dataset(dataset_id: str, keywords: Iterable[str] = CLINICAL_KEYWORDS) -> bool:
    upper = dataset_id.upper()
    return any(k.upper() in upper for k in keywords)


def make_dataset_splits(
    dataset_ids: Iterable[str],
    seed: int = 0,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    clinical_holdout: bool = True,
) -> pd.DataFrame:
    """Assign datasets to train/val/test, optionally holding clinical IDs out."""
    rows = []
    for ds_id in sorted(set(map(str, dataset_ids))):
        clinical = is_clinical_dataset(ds_id)
        if clinical_holdout and clinical:
            split = "clinical_test"
        else:
            r = stable_fraction(ds_id, seed=seed)
            if r < train_frac:
                split = "train"
            elif r < train_frac + val_frac:
                split = "val"
            else:
                split = "test"
        rows.append({"dataset_id": ds_id, "split": split, "clinical": clinical})
    return pd.DataFrame(rows)


def add_splits(df: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Attach split labels to a canonical DMS dataframe."""
    return df.merge(splits, on="dataset_id", how="left")


def split_dataframe(df: pd.DataFrame, splits: pd.DataFrame) -> dict[str, pd.DataFrame]:
    labeled = add_splits(df, splits)
    return {name: sub.reset_index(drop=True) for name, sub in labeled.groupby("split")}


VALID_SPLITS = ("train", "val", "test", "clinical_test")


def filter_by_split(df: pd.DataFrame, splits: pd.DataFrame, target: str) -> pd.DataFrame:
    """Restrict ``df`` to rows whose ``dataset_id`` is assigned to ``target``."""
    if target not in VALID_SPLITS:
        raise ValueError(f"Unknown split {target!r}; expected one of {VALID_SPLITS}")
    keep = set(splits.loc[splits["split"] == target, "dataset_id"].astype(str))
    if not keep:
        raise ValueError(f"No datasets assigned to split={target!r}")
    return df[df["dataset_id"].astype(str).isin(keep)].copy()


def assert_tasks_in_split(
    tasks: pd.DataFrame,
    splits: pd.DataFrame,
    target: str,
) -> None:
    """Raise if any task's ``dataset_id`` is not assigned to ``target``.

    Used at the top of every downstream benchmark entrypoint as a hard guard
    against accidental train-data leakage into headline metrics.
    """
    if target not in VALID_SPLITS:
        raise ValueError(f"Unknown split {target!r}; expected one of {VALID_SPLITS}")
    if "dataset_id" not in tasks.columns:
        raise ValueError("tasks dataframe must have a 'dataset_id' column")
    splits_map = dict(zip(splits["dataset_id"].astype(str), splits["split"].astype(str)))
    offenders = []
    for ds_id in tasks["dataset_id"].astype(str).unique():
        actual = splits_map.get(ds_id)
        if actual != target:
            offenders.append((ds_id, actual))
    if offenders:
        head = ", ".join(f"{d}({s!r})" for d, s in offenders[:5])
        raise ValueError(
            f"Tasks contain {len(offenders)} dataset(s) not in split={target!r}: {head}"
            + ("…" if len(offenders) > 5 else "")
        )
