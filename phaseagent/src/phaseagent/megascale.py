"""Megascale (Tsuboyama et al. Nature 2023) data loader.

The canonical resource is ``RosettaCommons/MegaScale`` on Hugging Face. We use
two configs:

- ``dataset3_single``: 1.84M single-mutant rows with the ThermoMPNN-published
  train / val / test split (also has a 5-fold CV variant ``dataset3_single_cv``).
- ``dataset2``: 776k high-quality measurements including ~210K curated double
  mutants across 559 site pairs in 331 natural + 148 de novo domains.

Both come with bundled AlphaFold-2 PDBs (``AlphaFold_model_PDBs`` config). The
loader caches Parquet shards in a Modal volume / local cache so subsequent
training runs are I/O-cheap.

Label conventions (from the HF README):
- ``deltaG`` (kcal/mol): WT minus mutant ΔG (so positive = stabilizing).
- ``ddG_ML``: ML-cleaned ΔΔG used by ThermoMPNN, Stability Oracle.
- ``log10_K50_t`` / ``log10_K50_c``: raw proteolysis half-life proxies.

For our editing benchmark we map ΔΔG into a [0, 1] viability score via
rank-normalization within each protein, mirroring the ProteinGym convention.
This lets us reuse the existing EditGuard pipeline without per-dataset
threshold gymnastics.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Three-letter -> one-letter amino acid table for parsing Megascale mutation strings.
_AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


@dataclass(frozen=True)
class MegascaleConfig:
    """Which Megascale slice to load."""

    config_name: str = "dataset3_single"  # or "dataset2", "dataset3_single_cv"
    split: str = "train"  # "train" | "val" | "test"
    label_col: str = "ddG_ML"
    cache_dir: str = "data/megascale"
    # If True, drop variants whose mutation_notation cannot be parsed.
    drop_unparseable: bool = True


def normalize_mutation_notation(raw: str) -> str | None:
    """Convert Megascale mutation strings to ProteinGym-style notation.

    Megascale uses two formats. The ``dataset3_single`` table stores single
    mutations as ``"P123A"`` already (matching ProteinGym). The ``dataset2``
    table can use colon-separated ``"P123A:K456E"`` for multi-mutants. Some
    older rows use 3-letter codes like ``"PRO123ALA"``.

    Returns None if the string cannot be parsed.
    """
    if raw is None or not isinstance(raw, str) or len(raw) == 0:
        return None
    parts = []
    for chunk in raw.replace(",", ":").split(":"):
        chunk = chunk.strip().upper()
        if len(chunk) == 0:
            continue
        # Already 1-letter style? (e.g., "P123A" or "M-1A")
        if len(chunk) >= 3 and chunk[0].isalpha() and chunk[-1].isalpha():
            wt, mut = chunk[0], chunk[-1]
            pos = chunk[1:-1]
            if wt in _AA3.values() and mut in _AA3.values() and pos.lstrip("-").isdigit():
                parts.append(f"{wt}{pos}{mut}")
                continue
        # 3-letter style? (e.g., "PRO123ALA")
        if len(chunk) >= 7:
            wt3 = chunk[:3]
            mut3 = chunk[-3:]
            pos = chunk[3:-3]
            if wt3 in _AA3 and mut3 in _AA3 and pos.lstrip("-").isdigit():
                parts.append(f"{_AA3[wt3]}{pos}{_AA3[mut3]}")
                continue
        return None
    return ":".join(parts) if parts else None


def to_proteingym_schema(
    raw: pd.DataFrame,
    *,
    label_col: str,
    drop_unparseable: bool = True,
) -> pd.DataFrame:
    """Adapt a raw Megascale frame into the EditGuard canonical schema.

    Adds columns: dataset_id, mutation_notation, mutation_distance,
    fitness_norm, viable, wildtype_sequence (when present), mutated_sequence
    (when present). Per-protein rank-normalizes the label into [0, 1].
    """
    df = raw.copy()
    if label_col not in df.columns:
        raise ValueError(f"label column {label_col!r} missing from raw Megascale frame")

    # Detect wildtype id column. The HF schema uses ``WT_name`` (PDB-like) for
    # the protein identifier; fall back to ``name`` or ``protein_name`` if needed.
    for cand in ("WT_name", "name", "protein_name", "domain_id"):
        if cand in df.columns:
            df["dataset_id"] = df[cand].astype(str)
            break
    if "dataset_id" not in df.columns:
        raise ValueError("Could not infer dataset_id from raw Megascale columns")

    # Mutation notation. Megascale stores mutations in ``mut_type`` for the
    # multi-mutant table and ``mutant`` / ``mutation`` for singles.
    mut_col = None
    for cand in ("mut_type", "mutant", "mutation", "mutations"):
        if cand in df.columns:
            mut_col = cand
            break
    if mut_col is None:
        raise ValueError("Could not infer mutation column from raw Megascale frame")
    df["mutation_notation"] = df[mut_col].apply(normalize_mutation_notation)
    if drop_unparseable:
        df = df[df["mutation_notation"].notna()].reset_index(drop=True)
    df["mutation_distance"] = df["mutation_notation"].apply(
        lambda s: 0 if not isinstance(s, str) else len(s.split(":"))
    )

    # Per-protein rank normalization of the label into [0, 1]. For ΔΔG, larger
    # ddG_ML in Megascale convention is more *stabilizing*; rank-norm with
    # method="average" so ties get an averaged rank.
    def _rank_norm(group: pd.Series) -> pd.Series:
        v = pd.to_numeric(group, errors="coerce")
        if v.notna().sum() < 2:
            return pd.Series(np.full(len(v), 0.5), index=group.index)
        ranks = v.rank(method="average")
        return (ranks - 1) / (ranks.notna().sum() - 1)

    df["fitness_norm"] = (
        df.groupby("dataset_id")[label_col]
        .transform(_rank_norm)
        .astype(float)
    )

    # Top quartile = viable (matches the ProteinGym EditGuard convention).
    df["viable"] = (df["fitness_norm"] >= 0.75).astype(int)

    # Carry through sequences if present. dataset3_single sometimes ships
    # ``aa_seq``; dataset2 provides ``WT_seq`` and ``mutated_seq``.
    if "WT_seq" in df.columns and "wildtype_sequence" not in df.columns:
        df["wildtype_sequence"] = df["WT_seq"].astype(str)
    if "mutated_seq" in df.columns and "mutated_sequence" not in df.columns:
        df["mutated_sequence"] = df["mutated_seq"].astype(str)
    if "aa_seq" in df.columns and "wildtype_sequence" not in df.columns:
        df["wildtype_sequence"] = df["aa_seq"].astype(str)

    df["variant_id"] = df["dataset_id"].astype(str) + "::" + df["mutation_notation"].astype(str)
    return df


def load_megascale(config: MegascaleConfig) -> pd.DataFrame:
    """Load a Megascale config from Hugging Face into the canonical schema.

    Caches Parquet shards under ``config.cache_dir`` so repeat reads are local.
    Requires ``pip install datasets`` — raises a clear error otherwise.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "loading Megascale requires `datasets` (run `pip install datasets`)"
        ) from exc
    cache_path = Path(config.cache_dir).resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(
        "RosettaCommons/MegaScale",
        config.config_name,
        split=config.split,
        cache_dir=str(cache_path),
    )
    raw = ds.to_pandas()
    return to_proteingym_schema(raw, label_col=config.label_col, drop_unparseable=config.drop_unparseable)


def load_megascale_double_mutant_test(cache_dir: str = "data/megascale") -> pd.DataFrame:
    """Convenience: just the double-mutant rows from ``dataset2``.

    Used as the "T2 — double mutant" tier in the EditGuard 3-tier protocol.
    """
    cfg = MegascaleConfig(config_name="dataset2", split="train", cache_dir=cache_dir)
    df = load_megascale(cfg)
    return df[df["mutation_distance"] == 2].reset_index(drop=True)


def megascale_three_tier_splits(cache_dir: str = "data/megascale") -> dict[str, pd.DataFrame]:
    """Return the (T1 single test, T2 doubles, T3 placeholder) frames.

    T3 (PTmul-NR k≥2) is loaded separately by ``load_ptmul_nr`` because the
    underlying source is not on Hugging Face.
    """
    return {
        "t1_single_test": load_megascale(
            MegascaleConfig(config_name="dataset3_single", split="test", cache_dir=cache_dir)
        ),
        "t1_single_train": load_megascale(
            MegascaleConfig(config_name="dataset3_single", split="train", cache_dir=cache_dir)
        ),
        "t1_single_val": load_megascale(
            MegascaleConfig(config_name="dataset3_single", split="val", cache_dir=cache_dir)
        ),
        "t2_doubles": load_megascale_double_mutant_test(cache_dir=cache_dir),
    }
