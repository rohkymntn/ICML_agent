"""Normalize raw DMS into the project's canonical schema."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io import (
    infer_fitness_column,
    infer_mutation_column,
    infer_sequence_column,
    infer_wildtype_column,
)
from .mutations import (
    mutation_distance_from_notation,
    mutation_distance_from_sequences,
)


def normalize_fitness(df: pd.DataFrame, method: str = "rank") -> pd.DataFrame:
    """Add 'fitness_norm' column from 'fitness_raw'."""
    out = df.copy()
    raw = out["fitness_raw"].astype(float)
    if method == "rank":
        out["fitness_norm"] = raw.rank(pct=True, method="average")
    elif method == "zscore":
        std = raw.std(ddof=0)
        out["fitness_norm"] = (raw - raw.mean()) / (std if std > 0 else 1.0)
    elif method == "minmax":
        rng = raw.max() - raw.min()
        out["fitness_norm"] = (raw - raw.min()) / (rng if rng > 0 else 1.0)
    elif method == "raw":
        out["fitness_norm"] = raw
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return out


def add_viability_label(
    df: pd.DataFrame,
    threshold_mode: str,
    threshold_value: float,
) -> pd.DataFrame:
    """Add 'viable' (0/1) using either a quantile cutoff on fitness_norm or an absolute value."""
    out = df.copy()
    if threshold_mode == "quantile":
        cutoff = float(out["fitness_norm"].quantile(threshold_value))
    elif threshold_mode == "absolute":
        cutoff = float(threshold_value)
    else:
        raise ValueError(f"Unknown threshold_mode: {threshold_mode}")
    out["viable"] = (out["fitness_norm"] >= cutoff).astype(int)
    out.attrs["viability_cutoff"] = cutoff
    return out


def preprocess_dataset(raw_df: pd.DataFrame, dataset_id: str, config: dict) -> pd.DataFrame:
    """Lower a raw DMS table to the canonical schema.

    Columns out: dataset_id, variant_id, wildtype_sequence, mutated_sequence,
    mutation_notation, mutation_distance, fitness_raw, fitness_norm, viable.
    """
    fitness_col = config.get("fitness_col") or infer_fitness_column(raw_df)
    mutation_col = config.get("mutation_col") or infer_mutation_column(raw_df)
    sequence_col = config.get("sequence_col") or infer_sequence_column(raw_df)
    wt_col = config.get("wildtype_col") or infer_wildtype_column(raw_df)

    wt = config.get("wildtype_sequence")
    if wt is None and wt_col is not None:
        wts = raw_df[wt_col].dropna().unique()
        if len(wts) == 1:
            wt = wts[0]

    work = pd.DataFrame()
    n = len(raw_df)
    work["dataset_id"] = [dataset_id] * n
    work["variant_id"] = (
        raw_df[mutation_col].astype(str).values
        if mutation_col is not None
        else [f"{dataset_id}_v{i}" for i in range(n)]
    )
    work["wildtype_sequence"] = wt
    work["mutated_sequence"] = (
        raw_df[sequence_col].astype(str).values if sequence_col is not None else None
    )
    work["mutation_notation"] = (
        raw_df[mutation_col].astype(str).values if mutation_col is not None else None
    )

    if mutation_col is not None:
        work["mutation_distance"] = work["mutation_notation"].apply(
            mutation_distance_from_notation
        )
    elif sequence_col is not None and wt is not None:
        work["mutation_distance"] = work["mutated_sequence"].apply(
            lambda s: mutation_distance_from_sequences(wt, s)
        )
    else:
        raise ValueError(
            f"Cannot compute mutation distance for {dataset_id}: need mutation notation or (WT + sequence)."
        )

    work["fitness_raw"] = pd.to_numeric(raw_df[fitness_col], errors="coerce").values
    work = work.dropna(subset=["fitness_raw"]).reset_index(drop=True)

    method = config.get("fitness_normalization", "rank")
    work = normalize_fitness(work, method=method)

    threshold_mode = config.get("threshold_mode", "quantile")
    threshold_value = config.get("threshold_value", 0.75)
    work = add_viability_label(work, threshold_mode, threshold_value)
    return work
