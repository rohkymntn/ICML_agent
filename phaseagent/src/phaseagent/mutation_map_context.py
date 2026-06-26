"""Experimental mutation-map context for DMS-conditioned generation.

The central object is a target-specific single-mutant DMS map.  It is
represented as a dense ``[L, 20]`` score matrix plus a matching observation
mask.  Multi-mutant rows are deliberately rejected by the strict constructor:
the context is allowed to contain only one-step experimental measurements.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .mutations import parse_mutation_notation
from .spectrum import AMINO_ACIDS, AA_TO_ID


AA_LIST = tuple(AMINO_ACIDS)
CONTEXT_FEATURE_DIM = 49  # 20 scores + 20 mask + 9 per-position/global features


@dataclass(frozen=True)
class MutationMapStats:
    n_observed: int
    coverage: float
    mean: float
    std: float
    median: float
    iqr: float
    min_score: float
    max_score: float


@dataclass(frozen=True)
class MutationMapContext:
    """Single-mutant experimental context for one protein assay."""

    dataset_id: str
    wildtype_sequence: str
    score_matrix: np.ndarray  # [L, 20], normalized single-mutant effects
    raw_score_matrix: np.ndarray  # [L, 20], original assay scores
    observed_mask: np.ndarray  # [L, 20], 1 when measured
    position_features: np.ndarray  # [L, 9]
    stats: MutationMapStats
    fitness_col: str = "fitness_norm"

    @property
    def length(self) -> int:
        return int(len(self.wildtype_sequence))

    @property
    def context_dim(self) -> int:
        return CONTEXT_FEATURE_DIM

    def features_for_positions(self, positions: list[int] | tuple[int, ...] | np.ndarray) -> np.ndarray:
        """Return one context vector per 1-indexed sequence position."""
        if len(positions) == 0:
            return np.zeros((0, CONTEXT_FEATURE_DIM), dtype=np.float32)
        idx = np.asarray(positions, dtype=int) - 1
        if np.any(idx < 0) or np.any(idx >= self.length):
            raise ValueError("positions must be 1-indexed and inside the wildtype sequence")
        return np.concatenate(
            [
                self.score_matrix[idx],
                self.observed_mask[idx],
                self.position_features[idx],
            ],
            axis=1,
        ).astype(np.float32)

    def additive_score(self, mutation_notation: str, missing_value: float = 0.0) -> float:
        """Sum normalized single-mutant context values for a multi-mutant notation."""
        total = 0.0
        for tok in parse_mutation_notation(mutation_notation):
            try:
                pos = int(tok[1:-1])
            except ValueError:
                continue
            aa = tok[-1].upper()
            aa_id = AA_TO_ID.get(aa)
            if aa_id is None or not (1 <= pos <= self.length):
                continue
            if self.observed_mask[pos - 1, aa_id] > 0:
                total += float(self.score_matrix[pos - 1, aa_id])
            else:
                total += float(missing_value)
        return float(total)


def _resolve_fitness_col(df: pd.DataFrame, fitness_col: str | None) -> str:
    if fitness_col is not None:
        if fitness_col not in df.columns:
            raise ValueError(f"fitness column {fitness_col!r} missing")
        return fitness_col
    for col in ("fitness_norm", "DMS_score", "fitness_raw", "ddG_ML"):
        if col in df.columns:
            return col
    raise ValueError("could not infer a fitness column")


def _validate_single_context(df: pd.DataFrame, strict: bool) -> pd.DataFrame:
    if "mutation_notation" not in df.columns:
        raise ValueError("DMS context requires mutation_notation")
    if "mutation_distance" in df.columns:
        dist = pd.to_numeric(df["mutation_distance"], errors="coerce")
        multi = df[(dist.notna()) & (dist > 1)]
        if strict and len(multi) > 0:
            raise ValueError(
                "MutationMapContext received multi-mutant rows. Pass only single-mutant "
                "measurements to avoid target-label leakage."
            )
        df = df[(dist == 1) | (df["mutation_notation"].apply(lambda x: len(parse_mutation_notation(x)) == 1))]
    else:
        parsed_len = df["mutation_notation"].apply(lambda x: len(parse_mutation_notation(x)))
        if strict and (parsed_len > 1).any():
            raise ValueError(
                "MutationMapContext received multi-mutant rows. Pass only single-mutant "
                "measurements to avoid target-label leakage."
            )
        df = df[parsed_len == 1]
    return df.copy().reset_index(drop=True)


def _infer_wildtype(df: pd.DataFrame, dataset_id: str, wildtype_sequence: str | None) -> str:
    if wildtype_sequence:
        return str(wildtype_sequence)
    from .edit_generators import infer_wildtype_sequence

    wt = infer_wildtype_sequence(df, dataset_id)
    if not wt:
        raise ValueError(f"could not infer wildtype sequence for {dataset_id}")
    return wt


def build_mutation_map_context(
    dms_df: pd.DataFrame,
    *,
    dataset_id: str | None = None,
    wildtype_sequence: str | None = None,
    fitness_col: str | None = None,
    strict_singles: bool = True,
    normalize: bool = True,
    clip_z: float = 5.0,
) -> MutationMapContext:
    """Build a leakage-safe single-mutant map for one assay.

    Parameters
    ----------
    dms_df:
        Canonical DMS rows.  In strict mode this frame must contain only
        single-mutant context rows for the target assay.
    dataset_id:
        Optional assay identifier.  Required when ``dms_df`` contains multiple
        assays.
    normalize:
        If true, observed scores are robust-z normalized within the target
        assay.  The raw scores remain available in ``raw_score_matrix``.
    """
    if len(dms_df) == 0:
        raise ValueError("cannot build MutationMapContext from an empty frame")
    if dataset_id is None:
        if "dataset_id" not in dms_df.columns:
            dataset_id = "dataset"
        else:
            vals = sorted(dms_df["dataset_id"].dropna().astype(str).unique())
            if len(vals) != 1:
                raise ValueError("dataset_id is required when context frame has multiple assays")
            dataset_id = vals[0]
    df = dms_df.copy()
    if "dataset_id" in df.columns:
        df = df[df["dataset_id"].astype(str) == str(dataset_id)].copy()
    if len(df) == 0:
        raise ValueError(f"no DMS rows found for dataset_id={dataset_id!r}")
    fitness = _resolve_fitness_col(df, fitness_col)
    wt = _infer_wildtype(df, str(dataset_id), wildtype_sequence)
    singles = _validate_single_context(df, strict=strict_singles)

    L = len(wt)
    raw = np.full((L, 20), np.nan, dtype=np.float32)
    observed = np.zeros((L, 20), dtype=np.float32)
    for _, row in singles.iterrows():
        toks = parse_mutation_notation(row.get("mutation_notation", ""))
        if len(toks) != 1:
            continue
        tok = toks[0]
        try:
            pos = int(tok[1:-1])
        except ValueError:
            continue
        alt = tok[-1].upper()
        aa_id = AA_TO_ID.get(alt)
        value = pd.to_numeric(row.get(fitness), errors="coerce")
        if aa_id is None or not (1 <= pos <= L) or pd.isna(value):
            continue
        raw[pos - 1, aa_id] = float(value)
        observed[pos - 1, aa_id] = 1.0

    values = raw[np.isfinite(raw)]
    if values.size == 0:
        raise ValueError(f"no parseable single-mutant measurements for {dataset_id}")
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size > 1 else 1.0
    std = std if np.isfinite(std) and std > 1e-8 else 1.0
    median = float(np.median(values))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    iqr = float(q75 - q25)
    iqr = iqr if np.isfinite(iqr) and iqr > 1e-8 else std
    if normalize:
        score = (raw - median) / iqr
        score = np.clip(score, -clip_z, clip_z)
    else:
        score = raw.copy()
    score = np.where(np.isfinite(score), score, 0.0).astype(np.float32)
    raw_filled = np.where(np.isfinite(raw), raw, np.nan).astype(np.float32)

    pos_features = np.zeros((L, 9), dtype=np.float32)
    for i in range(L):
        mask = observed[i].astype(bool)
        vals = score[i, mask]
        wt_aa = wt[i].upper()
        wt_id = AA_TO_ID.get(wt_aa, -1)
        pos_features[i, 0] = (i + 1) / max(L, 1)
        pos_features[i, 1] = wt_id / 19.0 if wt_id >= 0 else -1.0
        pos_features[i, 2] = float(mask.mean())
        pos_features[i, 3] = float(vals.mean()) if vals.size else 0.0
        pos_features[i, 4] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        pos_features[i, 5] = float(vals.max()) if vals.size else 0.0
        pos_features[i, 6] = float(vals.min()) if vals.size else 0.0
        pos_features[i, 7] = mean
        pos_features[i, 8] = std

    stats = MutationMapStats(
        n_observed=int(observed.sum()),
        coverage=float(observed.mean()),
        mean=mean,
        std=std,
        median=median,
        iqr=iqr,
        min_score=float(np.min(values)),
        max_score=float(np.max(values)),
    )
    return MutationMapContext(
        dataset_id=str(dataset_id),
        wildtype_sequence=wt,
        score_matrix=score,
        raw_score_matrix=raw_filled,
        observed_mask=observed,
        position_features=pos_features,
        stats=stats,
        fitness_col=fitness,
    )


def build_contexts_by_dataset(
    df: pd.DataFrame,
    *,
    dataset_ids: list[str] | tuple[str, ...] | None = None,
    fitness_col: str | None = None,
    strict_singles: bool = True,
) -> dict[str, MutationMapContext]:
    """Build one mutation-map context per dataset."""
    if "dataset_id" not in df.columns:
        return {"dataset": build_mutation_map_context(df, fitness_col=fitness_col, strict_singles=strict_singles)}
    ids = dataset_ids or sorted(df["dataset_id"].dropna().astype(str).unique())
    out: dict[str, MutationMapContext] = {}
    for ds in ids:
        sub = df[df["dataset_id"].astype(str) == str(ds)]
        singles = sub[sub["mutation_notation"].apply(lambda x: len(parse_mutation_notation(x)) == 1)].copy()
        if len(singles) == 0:
            continue
        out[str(ds)] = build_mutation_map_context(
            singles,
            dataset_id=str(ds),
            fitness_col=fitness_col,
            strict_singles=strict_singles,
        )
    return out

