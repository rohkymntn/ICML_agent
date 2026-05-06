"""Lightweight survival-curve models for Spectral PhaseAgent.

PyTorch is optional: importing this module should not require GPU dependencies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn

    _HAS_TORCH = True
except Exception:  # pragma: no cover - exercised only without torch installed
    torch = None
    nn = object
    _HAS_TORCH = False


def logit(p, eps: float = 1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def enforce_monotone_survival(values) -> np.ndarray:
    """Project a curve to non-increasing survival probabilities."""
    v = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    if v.size == 0:
        return v
    return np.minimum.accumulate(v)


class _TorchRequired:
    def __init__(self, *args, **kwargs):
        raise ImportError("PyTorch is required for neural survival models.")


if _HAS_TORCH:

    class SummaryMLP(nn.Module):
        """MLP mapping summary features and optional V_LD values to a survival curve."""

        def __init__(self, input_dim: int, depth_count: int, hidden_dim: int = 128, dropout: float = 0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, depth_count),
            )

        def forward(self, x):
            return torch.sigmoid(self.net(x))


    class SpectralSetTransformer(nn.Module):
        """Small Transformer encoder over unordered single-mutant tokens."""

        def __init__(
            self,
            token_dim: int = 4,
            depth_count: int = 16,
            model_dim: int = 128,
            n_heads: int = 4,
            n_layers: int = 2,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.input_proj = nn.Linear(token_dim, model_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=n_heads,
                dim_feedforward=4 * model_dim,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.pool = nn.Sequential(nn.Linear(model_dim, model_dim), nn.Tanh(), nn.Linear(model_dim, 1))
            self.head = nn.Sequential(nn.Linear(model_dim, model_dim), nn.ReLU(), nn.Linear(model_dim, depth_count))

        def forward(self, tokens, mask=None):
            h = self.encoder(self.input_proj(tokens), src_key_padding_mask=mask)
            scores = self.pool(h).squeeze(-1)
            if mask is not None:
                scores = scores.masked_fill(mask, -1e9)
            weights = torch.softmax(scores, dim=1)
            pooled = torch.sum(h * weights.unsqueeze(-1), dim=1)
            return self.head(pooled)


    class ResidualSurvivalModel(nn.Module):
        """Predict ``sigmoid(logit(V_LD) + residual(tokens, d))``."""

        def __init__(self, token_dim: int = 4, depth_count: int = 16, **kwargs):
            super().__init__()
            self.residual = SpectralSetTransformer(token_dim=token_dim, depth_count=depth_count, **kwargs)

        def forward(self, tokens, v_ld, mask=None):
            eps = 1e-6
            base = torch.logit(torch.clamp(v_ld, eps, 1.0 - eps))
            return torch.sigmoid(base + self.residual(tokens, mask=mask))

else:
    SummaryMLP = _TorchRequired
    SpectralSetTransformer = _TorchRequired
    ResidualSurvivalModel = _TorchRequired


def tokens_to_numpy(tokens: pd.DataFrame) -> np.ndarray:
    """Convert spectrum token table to numeric model inputs."""
    cols = ["pos_norm", "wt_aa_id", "mut_aa_id", "delta_f"]
    missing = [c for c in cols if c not in tokens.columns]
    if missing:
        raise ValueError(f"Missing token columns: {missing}")
    return tokens[cols].to_numpy(dtype=float)


def residual_curve_prediction(
    v_ld: pd.DataFrame,
    residual: np.ndarray | list[float],
    survival_col: str = "survival_large_deviation",
    monotone: bool = True,
) -> pd.DataFrame:
    """Combine an additive survival baseline with a residual-logit curve."""
    base = v_ld[survival_col].to_numpy(dtype=float)
    values = sigmoid(logit(base) + np.asarray(residual, dtype=float))
    if monotone:
        values = enforce_monotone_survival(values)
    out = v_ld[["mutation_distance"]].copy()
    out["survival_pred"] = values
    out["survival_base"] = base
    out["residual_logit"] = np.asarray(residual, dtype=float)
    if "dataset_id" in v_ld.columns:
        out["dataset_id"] = v_ld["dataset_id"].values
    return out


def boundary_weighted_curve_loss(y_pred, y_true, weight_scale: float = 5.0) -> float:
    """Numpy version of the boundary-weighted curve MSE."""
    pred = np.asarray(y_pred, dtype=float)
    true = np.asarray(y_true, dtype=float)
    if pred.shape != true.shape:
        raise ValueError("Predicted and true curves must have the same shape.")
    if pred.size > 1:
        chi = np.abs(np.gradient(true))
        denom = float(np.nanmax(chi)) if np.isfinite(chi).any() else 0.0
        weights = 1.0 + weight_scale * (chi / denom) if denom > 0 else np.ones_like(true)
    else:
        weights = np.ones_like(true)
    return float(np.average((pred - true) ** 2, weights=weights))
