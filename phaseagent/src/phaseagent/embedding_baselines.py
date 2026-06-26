"""Tiny-model-on-frozen-PLM-embedding baselines (EVOLVEpro / FLIP2 family).

Two baselines, both standard in 2025/2026 protein-engineering literature:

- ``RFEmbeddingBaseline``: random forest regressor on ESM-2 mean-pool features.
  This is the EVOLVEpro winning configuration (Jiang et al., *Science* 2025).
- ``LightGBMEmbeddingBaseline``: gradient boosting on the same features. The
  Sci Rep 2025 crystallization-prediction benchmark and FLIP2 both find LGBM
  matches or beats XGBoost on PLM-embedding features.
- ``OneHotZeroShotRidgeBaseline``: ridge regression on one-hot mutation
  encoding concatenated with a zero-shot likelihood column. This is the
  surprisingly-strong FLIP2 baseline that matches fine-tuned PLMs on small data.

All three accept frozen embeddings (or compute them through an Embedder) and
expose a uniform ``predict_fitness`` interface so they can drop into the
existing ``DMSFunctionPrior``-style downstream pipeline (rerank, conformal,
etc.).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .embeddings import Embedder, embed_dataframe
from .mutations import parse_mutation_notation
from .spectrum import AA_TO_ID, AMINO_ACIDS

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge

    _HAS_SK = True
except Exception:  # pragma: no cover
    _HAS_SK = False

try:
    import lightgbm as lgb

    _HAS_LGB = True
except Exception:
    _HAS_LGB = False


@dataclass
class RFEmbeddingBaseline:
    """Random forest on frozen PLM embeddings (EVOLVEpro recipe)."""

    embedder: Embedder | None = None
    n_estimators: int = 200
    max_depth: int = 12
    min_samples_leaf: int = 3
    random_state: int = 0
    seq_col: str = "mutated_sequence"
    score_col: str = "fitness_norm"

    def __post_init__(self):
        self._model = None
        self._train_X_cache = None

    def fit(
        self,
        train_df: pd.DataFrame,
        precomputed_embeddings: np.ndarray | None = None,
    ) -> "RFEmbeddingBaseline":
        if not _HAS_SK:
            raise ImportError("RFEmbeddingBaseline requires scikit-learn")
        if precomputed_embeddings is not None:
            X = np.asarray(precomputed_embeddings, dtype=np.float32)
        else:
            if self.embedder is None:
                raise ValueError("either precomputed_embeddings or embedder must be provided")
            X = embed_dataframe(train_df, self.embedder, seq_col=self.seq_col)
        y = pd.to_numeric(train_df[self.score_col], errors="coerce").to_numpy(dtype=float)
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
        ).fit(X, y)
        return self

    def predict_fitness(
        self,
        df: pd.DataFrame,
        precomputed_embeddings: np.ndarray | None = None,
    ) -> np.ndarray:
        if self._model is None:
            raise ValueError("RFEmbeddingBaseline not fitted")
        if precomputed_embeddings is not None:
            X = np.asarray(precomputed_embeddings, dtype=np.float32)
        else:
            if self.embedder is None:
                raise ValueError("either precomputed_embeddings or embedder must be provided")
            X = embed_dataframe(df, self.embedder, seq_col=self.seq_col)
        return np.round(np.clip(self._model.predict(X).astype(float), 0.0, 1.0), 12)

    # Compatibility with DMSFunctionPrior interface for the conformal layer.
    def predict(
        self,
        df: pd.DataFrame,
        precomputed_embeddings: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.predict_fitness(df, precomputed_embeddings=precomputed_embeddings)


@dataclass
class LightGBMEmbeddingBaseline:
    """LightGBM regressor on frozen PLM embeddings (FLIP2-style)."""

    embedder: Embedder | None = None
    n_estimators: int = 500
    learning_rate: float = 0.05
    num_leaves: int = 63
    random_state: int = 0
    seq_col: str = "mutated_sequence"
    score_col: str = "fitness_norm"

    def __post_init__(self):
        self._model = None

    def fit(
        self,
        train_df: pd.DataFrame,
        precomputed_embeddings: np.ndarray | None = None,
        val_df: pd.DataFrame | None = None,
        val_embeddings: np.ndarray | None = None,
    ) -> "LightGBMEmbeddingBaseline":
        if not _HAS_LGB:
            raise ImportError("LightGBMEmbeddingBaseline requires lightgbm")
        if precomputed_embeddings is not None:
            X = np.asarray(precomputed_embeddings, dtype=np.float32)
        else:
            if self.embedder is None:
                raise ValueError("either precomputed_embeddings or embedder must be provided")
            X = embed_dataframe(train_df, self.embedder, seq_col=self.seq_col)
        y = pd.to_numeric(train_df[self.score_col], errors="coerce").to_numpy(dtype=float)
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "random_state": self.random_state,
            "verbose": -1,
            "n_jobs": -1,
        }
        train_set = lgb.Dataset(X, y)
        valid_sets = [train_set]
        if val_df is not None:
            if val_embeddings is None and self.embedder is not None:
                val_embeddings = embed_dataframe(val_df, self.embedder, seq_col=self.seq_col)
            if val_embeddings is not None:
                vy = pd.to_numeric(val_df[self.score_col], errors="coerce").to_numpy(dtype=float)
                valid_sets.append(lgb.Dataset(np.asarray(val_embeddings, dtype=np.float32), vy))
        self._model = lgb.train(
            params,
            train_set,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)] if len(valid_sets) > 1 else None,
        )
        return self

    def predict_fitness(
        self,
        df: pd.DataFrame,
        precomputed_embeddings: np.ndarray | None = None,
    ) -> np.ndarray:
        if self._model is None:
            raise ValueError("LightGBMEmbeddingBaseline not fitted")
        if precomputed_embeddings is not None:
            X = np.asarray(precomputed_embeddings, dtype=np.float32)
        else:
            if self.embedder is None:
                raise ValueError("either precomputed_embeddings or embedder must be provided")
            X = embed_dataframe(df, self.embedder, seq_col=self.seq_col)
        return np.clip(self._model.predict(X).astype(float), 0.0, 1.0)

    def predict(
        self,
        df: pd.DataFrame,
        precomputed_embeddings: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.predict_fitness(df, precomputed_embeddings=precomputed_embeddings)


def _onehot_mutation_features(
    df: pd.DataFrame,
    wt_seq_lookup: dict[str, str],
    mutation_col: str = "mutation_notation",
    dataset_col: str = "dataset_id",
    max_positions: int = 1024,
) -> np.ndarray:
    """One-hot encode (position, mutant aa) pairs, padded to a fixed length.

    For each variant, returns a sparse 21-dim one-hot at each mutated position
    (20 amino acids + a wt-marker channel), summed into a (max_positions, 21)
    matrix and flattened. Variants in different proteins use the same fixed
    layout — distinct positions across proteins are handled by the per-dataset
    branch in the downstream model (typically a one-hot of dataset_id).
    """
    rows = len(df)
    out = np.zeros((rows, max_positions * 21), dtype=np.float32)
    for i, row in enumerate(df.itertuples(index=False)):
        notation = str(getattr(row, mutation_col))
        tokens = parse_mutation_notation(notation)
        for tok in tokens:
            try:
                pos = int(tok[1:-1])
                mut_aa = tok[-1]
            except (ValueError, IndexError):
                continue
            if pos < 0 or pos >= max_positions:
                continue
            if mut_aa not in AA_TO_ID:
                continue
            out[i, pos * 21 + AA_TO_ID[mut_aa]] = 1.0
            out[i, pos * 21 + 20] = 1.0  # wt-marker channel = "this position was mutated"
    return out


@dataclass
class OneHotZeroShotRidgeBaseline:
    """Ridge on one-hot mutation features + zero-shot likelihood column.

    The FLIP2 paper (Dallago et al. bioRxiv 2026) showed this baseline matches
    full PLM fine-tuning across 7 new fitness datasets in the few-shot regime.
    Cheap, fast, surprisingly strong.

    The caller passes a ``zero_shot_col`` (default ``esm2_log_likelihood``)
    that is concatenated to the one-hot features as one extra column. Without
    a zero-shot column the model degenerates to plain one-hot ridge.
    """

    alpha: float = 1.0
    seq_col: str = "mutated_sequence"
    score_col: str = "fitness_norm"
    zero_shot_col: str | None = "esm2_log_likelihood"
    mutation_col: str = "mutation_notation"
    dataset_col: str = "dataset_id"
    max_positions: int = 1024

    def __post_init__(self):
        self._model = None

    def _features(self, df: pd.DataFrame) -> np.ndarray:
        oh = _onehot_mutation_features(
            df,
            wt_seq_lookup={},  # not actually needed for the simple encoding
            mutation_col=self.mutation_col,
            dataset_col=self.dataset_col,
            max_positions=self.max_positions,
        )
        if self.zero_shot_col is not None and self.zero_shot_col in df.columns:
            zs = pd.to_numeric(df[self.zero_shot_col], errors="coerce").to_numpy(dtype=float)
            zs = np.where(np.isnan(zs), 0.0, zs).reshape(-1, 1)
            return np.concatenate([oh, zs.astype(np.float32)], axis=1)
        return oh

    def fit(self, train_df: pd.DataFrame) -> "OneHotZeroShotRidgeBaseline":
        if not _HAS_SK:
            raise ImportError("OneHotZeroShotRidgeBaseline requires scikit-learn")
        X = self._features(train_df)
        y = pd.to_numeric(train_df[self.score_col], errors="coerce").to_numpy(dtype=float)
        self._model = Ridge(alpha=self.alpha).fit(X, y)
        return self

    def predict_fitness(self, df: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise ValueError("OneHotZeroShotRidgeBaseline not fitted")
        X = self._features(df)
        return np.clip(self._model.predict(X).astype(float), 0.0, 1.0)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_fitness(df)
