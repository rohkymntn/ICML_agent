"""Per-protein conformal abstention for selective protein editing.

The headline calibration finding from the v1 EditGuard runs is that the DMS
function prior is well-calibrated on GFP (ECE 0.035) and GCN4 (0.117) but
fails on F7YBW8 (0.535). The clean response — recommended by Boger et al.
(*Nat Commun* 2025, "Functional protein mining with conformal guarantees") and
by Greenman et al. (*PLOS Comp Bio* 2025) — is to add a selective-prediction
layer that abstains on variants whose calibrated prediction interval is too
wide. Per-protein (Mondrian) calibration is the right granularity because the
miscalibration is family-dependent (Gordon et al., "PLM Fitness is a Matter of
Preference", ICLR 2025).

This module is a thin, dependency-light wrapper around split-conformal
regression with optional Mondrian (per-protein) calibration. We implement it
ourselves rather than depending on ``crepes`` to keep the package install
small; if ``crepes`` is available the caller can swap it in trivially.

Protocol (Boger 2025 / Fannjiang 2022):
1. Train the predictor on training data only.
2. Score a held-out *calibration* set; record nonconformity scores.
3. At inference, return a (1-α) prediction interval per variant.
4. Abstain when the per-variant interval width exceeds a per-protein
   threshold calibrated to keep abstention rate ≤ ``max_abstain_rate``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class ConformalRegressor:
    """Split-conformal regression with optional Mondrian (per-group) calibration.

    The wrapped ``base_predictor`` must expose a ``predict(df) -> np.ndarray``
    method (matching the ``DMSFunctionPrior.predict_fitness`` interface).
    """

    base_predictor: object
    alpha: float = 0.1  # 1 - alpha = nominal coverage (0.9 for default).
    group_col: str | None = "dataset_id"
    score_col: str = "fitness_norm"
    max_abstain_rate: float = 0.3
    # Internal: per-group nonconformity quantiles after calibration.
    _group_thresholds: dict[str | None, float] = field(default_factory=dict)
    _global_threshold: float = field(default=0.0)
    _group_widths: dict[str | None, float] = field(default_factory=dict)
    _global_width: float = field(default=0.0)

    def _predict_base(self, df: pd.DataFrame) -> np.ndarray:
        """Call the wrapped base predictor and coerce to a 1-D float array."""
        if hasattr(self.base_predictor, "predict_fitness"):
            return np.asarray(self.base_predictor.predict_fitness(df), dtype=float)
        if hasattr(self.base_predictor, "predict"):
            return np.asarray(self.base_predictor.predict(df), dtype=float)
        raise ValueError("base_predictor must expose predict_fitness or predict")

    def calibrate(self, calib_df: pd.DataFrame) -> "ConformalRegressor":
        """Compute (Mondrian) nonconformity quantiles on a calibration set.

        Nonconformity score = |y_obs - y_hat|. The (1 - alpha) empirical
        quantile of these scores is the prediction interval half-width.
        Per-group: compute one quantile per group; fall back to the global
        quantile when a group has < 30 calibration points.
        """
        if self.score_col not in calib_df.columns:
            raise ValueError(f"calibration frame missing {self.score_col!r}")
        y = pd.to_numeric(calib_df[self.score_col], errors="coerce").to_numpy(dtype=float)
        y_hat = self._predict_base(calib_df)
        nonconformity = np.abs(y - y_hat)
        finite = np.isfinite(nonconformity)
        n = int(finite.sum())
        if n < 30:
            raise ValueError(
                f"need at least 30 finite calibration points; got {n}"
            )
        nc_finite = nonconformity[finite]
        # Conformal quantile correction: ⌈(n+1)(1-alpha)⌉ / n.
        q = float(np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        q = float(np.clip(q, 0.0, 1.0))
        self._global_threshold = float(np.quantile(nc_finite, q))
        self._global_width = 2.0 * self._global_threshold
        self._group_thresholds = {}
        self._group_widths = {}
        if self.group_col is not None and self.group_col in calib_df.columns:
            calib_df_loc = calib_df.copy()
            calib_df_loc["_nc"] = nonconformity
            calib_df_loc = calib_df_loc[finite]
            for g, sub in calib_df_loc.groupby(self.group_col):
                vals = sub["_nc"].to_numpy(dtype=float)
                if len(vals) < 30:
                    continue
                gq = float(np.ceil((len(vals) + 1) * (1.0 - self.alpha)) / len(vals))
                gq = float(np.clip(gq, 0.0, 1.0))
                thr = float(np.quantile(vals, gq))
                self._group_thresholds[str(g)] = thr
                self._group_widths[str(g)] = 2.0 * thr
        return self

    def predict_interval(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) prediction intervals at nominal coverage."""
        if self._global_threshold == 0.0 and not self._group_thresholds:
            raise ValueError("ConformalRegressor not calibrated")
        y_hat = self._predict_base(df)
        thresholds = np.full(len(df), self._global_threshold, dtype=float)
        if self.group_col is not None and self.group_col in df.columns and self._group_thresholds:
            groups = df[self.group_col].astype(str).to_numpy()
            for i, g in enumerate(groups):
                thresholds[i] = self._group_thresholds.get(g, self._global_threshold)
        return y_hat - thresholds, y_hat + thresholds

    def abstain_mask(self, df: pd.DataFrame) -> np.ndarray:
        """Return True for variants we should abstain on.

        Strategy: rank every variant by its per-group interval width
        (proxy for uncertainty) and abstain on the top ``max_abstain_rate``
        fraction. With only a handful of groups the per-variant widths are
        constant per group, so the rank effectively abstains on whole
        groups in width-descending order.

        We then break ties *within* a group using a secondary uncertainty
        signal (predicted probability close to 0.5) so the abstention is
        gradual rather than all-or-nothing.
        """
        if self.max_abstain_rate <= 0:
            return np.zeros(len(df), dtype=bool)
        lo, hi = self.predict_interval(df)
        widths = (hi - lo).astype(float)
        # Secondary tiebreaker: |predicted - 0.5| (lower = more uncertain).
        y_hat = self._predict_base(df)
        tie = -np.abs(np.asarray(y_hat, dtype=float) - 0.5)  # negate so larger = more uncertain
        # Composite rank score: width is the primary signal, tie is a small
        # additive nudge to break group-wide ties so we get partial abstention.
        score = widths + 1e-3 * tie
        n = len(df)
        n_abstain = int(np.ceil(n * float(self.max_abstain_rate)))
        if n_abstain == 0:
            return np.zeros(n, dtype=bool)
        threshold = np.sort(score)[-n_abstain]
        return score >= threshold

    def coverage(self, test_df: pd.DataFrame) -> dict[str, float]:
        """Empirical coverage of the prediction intervals on a held-out frame.

        Returns the global coverage and (when group_col is set) per-group
        coverage so the caller can verify Mondrian calibration is honest.
        """
        if self.score_col not in test_df.columns:
            raise ValueError(f"test frame missing {self.score_col!r}")
        y = pd.to_numeric(test_df[self.score_col], errors="coerce").to_numpy(dtype=float)
        lo, hi = self.predict_interval(test_df)
        finite = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
        if not finite.any():
            return {"global": float("nan")}
        in_band = (y >= lo) & (y <= hi)
        out: dict[str, float] = {
            "global": float(in_band[finite].mean()),
            "n": float(int(finite.sum())),
        }
        if self.group_col is not None and self.group_col in test_df.columns:
            groups = test_df[self.group_col].astype(str).to_numpy()
            for g in np.unique(groups):
                gmask = (groups == g) & finite
                if gmask.sum() >= 5:
                    out[f"group::{g}"] = float(in_band[gmask].mean())
        return out


def selective_top_k(
    df: pd.DataFrame,
    scores: np.ndarray,
    abstain_mask: np.ndarray,
    k: int,
) -> pd.DataFrame:
    """Pick top-k variants from the non-abstained subset.

    If fewer than ``k`` non-abstained variants exist for a (dataset, task),
    we return however many remain after abstention. The caller is expected
    to record the abstention rate alongside the hit rate.
    """
    keep = ~np.asarray(abstain_mask, dtype=bool)
    if keep.sum() == 0:
        return df.iloc[:0].copy()
    order = np.argsort(-np.asarray(scores, dtype=float))
    out_idx = []
    for i in order:
        if keep[i]:
            out_idx.append(i)
        if len(out_idx) >= k:
            break
    return df.iloc[out_idx].copy()
