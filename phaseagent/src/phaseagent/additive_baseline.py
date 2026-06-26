"""Additive multi-mutant baseline.

Recent stability-prediction work (ThermoMPNN-D, Stability Oracle, the
MULTI-evolve critique on bioRxiv 2026.04.23) has established that any claim
of "epistasis-aware" or "multi-mutant" prediction must be reported alongside
the simple additive baseline:

    score_additive(variant) = sum_i score_single(M_i)

Across Megascale double mutants, no published model meaningfully beats this
null model overall — gains live in the stabilizing-pair recall slice and at
small inter-residue distances. This module provides:

- ``AdditivePredictor``: fit on single-mutant labels, predict multi-mutant
  scores by summing the single-mutant predictions.
- ``epistasis_slice_metrics``: split the test set into epistatic and
  non-epistatic subsets and report metrics on each.
- ``stabilizing_pair_recall``: the "where the gains live" metric.

It works for any score (fitness, ΔΔG, log-likelihood) — the caller picks the
``score_col`` and ``mutation_col`` and we treat the data as a flat table.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from .mutations import parse_mutation_notation


@dataclass
class AdditivePredictor:
    """Predict multi-mutant scores as the sum of single-mutant scores.

    Fit on a table of *single*-mutant variants (one row per single mutation)
    and call ``predict`` on a table of variants of arbitrary mutation count.
    The score column name and mutation-notation column name are configurable
    so this works for ProteinGym fitness, Megascale ΔΔG, or any other
    per-variant score.
    """

    score_col: str = "fitness_norm"
    mutation_col: str = "mutation_notation"
    dataset_col: str | None = "dataset_id"
    fill_value: float = float("nan")
    # Map of (dataset_id_or_None, "P123A"-style single mutation) -> score.
    _table: dict[tuple[str | None, str], float] = field(default_factory=dict)

    def fit(self, single_df: pd.DataFrame) -> "AdditivePredictor":
        """Memorize per-(dataset, single-mutation) scores from single_df.

        Variants with mutation_count != 1 are silently ignored — this is a
        single-mutant-trained model by construction.
        """
        if self.score_col not in single_df.columns:
            raise ValueError(f"{self.score_col!r} missing from training frame")
        if self.mutation_col not in single_df.columns:
            raise ValueError(f"{self.mutation_col!r} missing from training frame")
        self._table = {}
        for _, row in single_df.iterrows():
            notation = str(row[self.mutation_col])
            tokens = parse_mutation_notation(notation)
            if len(tokens) != 1:
                continue
            ds = (
                str(row[self.dataset_col])
                if self.dataset_col is not None and self.dataset_col in single_df.columns
                else None
            )
            score = pd.to_numeric(row[self.score_col], errors="coerce")
            if pd.isna(score):
                continue
            self._table[(ds, tokens[0])] = float(score)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict the additive score for each row of df.

        Rows whose component single mutations are not all in the lookup table
        are filled with ``self.fill_value`` (default NaN). Single-mutant rows
        return the looked-up score directly.
        """
        if not self._table:
            raise ValueError("AdditivePredictor is not fitted")
        if self.mutation_col not in df.columns:
            raise ValueError(f"{self.mutation_col!r} missing from prediction frame")
        out = np.empty(len(df), dtype=float)
        for i, row in enumerate(df.itertuples(index=False)):
            notation = str(getattr(row, self.mutation_col))
            tokens = parse_mutation_notation(notation)
            ds = (
                str(getattr(row, self.dataset_col))
                if self.dataset_col is not None and hasattr(row, self.dataset_col)
                else None
            )
            scores = []
            for tok in tokens:
                v = self._table.get((ds, tok))
                if v is None and ds is not None:
                    v = self._table.get((None, tok))
                scores.append(v)
            if any(s is None for s in scores) or len(scores) == 0:
                out[i] = self.fill_value
            else:
                out[i] = float(sum(scores))
        return out

    def coverage(self, df: pd.DataFrame) -> float:
        """Fraction of rows for which every component single is in the table."""
        if not self._table:
            return 0.0
        preds = self.predict(df)
        return float(np.mean(~np.isnan(preds)))


def epistasis_signal(
    multi_df: pd.DataFrame,
    additive_predictor: AdditivePredictor,
    score_col: str = "fitness_norm",
) -> pd.Series:
    """Per-row |observed - additive| epistasis magnitude.

    Larger values indicate the multi-mutant score deviates from the simple
    sum-of-singles prediction, i.e. epistasis is present.
    """
    obs = pd.to_numeric(multi_df[score_col], errors="coerce").to_numpy(dtype=float)
    pred = additive_predictor.predict(multi_df)
    return pd.Series(np.abs(obs - pred), index=multi_df.index, name="epistasis_signal")


def epistasis_slice_metrics(
    test_df: pd.DataFrame,
    model_pred: np.ndarray,
    additive_pred: np.ndarray,
    obs_col: str = "fitness_norm",
    epistasis_threshold: float = 0.5,
) -> dict[str, float]:
    """Compare a model against the additive baseline on the epistatic slice.

    Returns a dict with overall and epistasis-slice Spearman, plus the
    Δ(model − additive) on the slice. The headline finding the field expects
    in 2026 is the slice number: even if a model ties additive overall, it
    can win or lose on the rows where epistasis actually exists.
    """
    from scipy.stats import spearmanr

    obs = pd.to_numeric(test_df[obs_col], errors="coerce").to_numpy(dtype=float)
    add = np.asarray(additive_pred, dtype=float)
    mod = np.asarray(model_pred, dtype=float)
    finite = np.isfinite(obs) & np.isfinite(add) & np.isfinite(mod)
    if finite.sum() < 5:
        return {"n": int(finite.sum()), "overall_model": float("nan"), "overall_additive": float("nan"),
                "slice_n": 0, "slice_model": float("nan"), "slice_additive": float("nan"), "slice_delta": float("nan")}
    obs_f = obs[finite]
    add_f = add[finite]
    mod_f = mod[finite]
    eps = np.abs(obs_f - add_f)
    in_slice = eps > float(epistasis_threshold)
    overall_model = float(spearmanr(obs_f, mod_f, nan_policy="omit").statistic)
    overall_additive = float(spearmanr(obs_f, add_f, nan_policy="omit").statistic)
    out = {
        "n": int(finite.sum()),
        "overall_model": overall_model,
        "overall_additive": overall_additive,
        "slice_n": int(in_slice.sum()),
    }
    if in_slice.sum() >= 5:
        slice_model = float(spearmanr(obs_f[in_slice], mod_f[in_slice], nan_policy="omit").statistic)
        slice_additive = float(spearmanr(obs_f[in_slice], add_f[in_slice], nan_policy="omit").statistic)
        out["slice_model"] = slice_model
        out["slice_additive"] = slice_additive
        out["slice_delta"] = slice_model - slice_additive
    else:
        out["slice_model"] = float("nan")
        out["slice_additive"] = float("nan")
        out["slice_delta"] = float("nan")
    return out


def stabilizing_pair_recall(
    test_df: pd.DataFrame,
    model_pred: np.ndarray,
    obs_col: str = "fitness_norm",
    stabilizing_threshold: float = 0.75,
    k: int = 50,
) -> float:
    """Recall of the top-``k`` model predictions among truly stabilizing pairs.

    A "stabilizing" pair is one whose observed score is in the top
    ``stabilizing_threshold`` quantile (e.g. top quartile by default for
    fitness; for ΔΔG, the caller should pre-flip the sign so 'top' means
    most stabilizing).
    """
    obs = pd.to_numeric(test_df[obs_col], errors="coerce").to_numpy(dtype=float)
    pred = np.asarray(model_pred, dtype=float)
    finite = np.isfinite(obs) & np.isfinite(pred)
    if finite.sum() < k:
        return float("nan")
    obs_f = obs[finite]
    pred_f = pred[finite]
    cut = np.quantile(obs_f, float(stabilizing_threshold))
    truly_stab = obs_f >= cut
    n_stab = int(truly_stab.sum())
    if n_stab == 0:
        return float("nan")
    top_k_idx = np.argsort(-pred_f)[: int(k)]
    hit_in_topk = int(truly_stab[top_k_idx].sum())
    return float(hit_in_topk / min(k, n_stab))
