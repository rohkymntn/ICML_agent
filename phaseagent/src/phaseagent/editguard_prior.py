"""DMS-trained function prior for EditGuard."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from .mutations import parse_mutation_notation
from .spectrum import AA_TO_ID

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression, Ridge

    _HAS_SK = True
except Exception:  # pragma: no cover
    _HAS_SK = False


def _token_features(mutation_notation) -> tuple[float, float, float, float, float]:
    toks = parse_mutation_notation(mutation_notation)
    if not toks:
        return 0.0, 0.0, -1.0, -1.0, 0.0
    positions, wt_ids, mut_ids = [], [], []
    for tok in toks:
        try:
            positions.append(int(tok[1:-1]))
        except ValueError:
            positions.append(0)
        wt_ids.append(AA_TO_ID.get(tok[0].upper(), -1))
        mut_ids.append(AA_TO_ID.get(tok[-1].upper(), -1))
    return (
        float(np.mean(positions)),
        float(np.std(positions)),
        float(np.mean(wt_ids)),
        float(np.mean(mut_ids)),
        float(len(set(positions))),
    )


def featurize_variants(df: pd.DataFrame) -> np.ndarray:
    """Build lightweight mutation/context features for DMS prior models."""
    n = len(df)
    X = np.zeros((n, 12), dtype=float)
    if n == 0:
        return X
    dist = df["mutation_distance"].to_numpy(dtype=float)
    X[:, 0] = dist
    X[:, 1] = dist**2
    X[:, 2] = np.log1p(dist)
    nota = df["mutation_notation"].astype(str) if "mutation_notation" in df.columns else pd.Series([""] * n)
    X[:, 3] = nota.str.len().to_numpy(dtype=float)
    X[:, 4] = nota.apply(lambda s: sum(ord(c) for c in s) % 997 / 997.0).to_numpy(dtype=float)
    tok = np.array([_token_features(s) for s in nota], dtype=float)
    X[:, 5:10] = tok
    if "dataset_id" in df.columns:
        ds = df["dataset_id"].astype(str)
        X[:, 10] = ds.apply(lambda s: len(s) % 101 / 101.0).to_numpy(dtype=float)
        X[:, 11] = ds.apply(lambda s: sum(ord(c) for c in s) % 997 / 997.0).to_numpy(dtype=float)
    return X


@dataclass
class PriorMetrics:
    auroc: float
    auprc: float
    spearman: float
    n: int


class DMSFunctionPrior:
    """Calibrated DMS function prior with classifier and regressor heads."""

    def __init__(self, random_state: int = 0, n_estimators: int = 200):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.classifier = None
        self.regressor = None
        self.calibrator = None

    def fit(self, train_df: pd.DataFrame, calibrate_df: pd.DataFrame | None = None):
        if "viable" not in train_df.columns or "fitness_norm" not in train_df.columns:
            raise ValueError("train_df must contain viable and fitness_norm columns")
        X = featurize_variants(train_df)
        y_cls = train_df["viable"].to_numpy(dtype=int)
        y_reg = train_df["fitness_norm"].to_numpy(dtype=float)
        if _HAS_SK:
            if len(np.unique(y_cls)) > 1 and len(train_df) >= 50:
                self.classifier = RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=12,
                    min_samples_leaf=3,
                    random_state=self.random_state,
                    n_jobs=-1,
                ).fit(X, y_cls)
            else:
                self.classifier = LogisticRegression(max_iter=1000).fit(X, y_cls)
            if len(train_df) >= 50:
                self.regressor = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=12,
                    min_samples_leaf=3,
                    random_state=self.random_state,
                    n_jobs=-1,
                ).fit(X, y_reg)
            else:
                self.regressor = Ridge(alpha=1.0).fit(X, y_reg)
            if calibrate_df is not None and len(calibrate_df) >= 20:
                p = self.predict_proba(calibrate_df)
                self.calibrator = IsotonicRegression(out_of_bounds="clip").fit(
                    p, calibrate_df["viable"].to_numpy(dtype=int)
                )
        else:
            self.classifier = _NumpyPrior(train_df)
            self.regressor = self.classifier
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.classifier is None:
            raise ValueError("DMSFunctionPrior is not fitted")
        X = featurize_variants(df)
        if hasattr(self.classifier, "predict_proba"):
            p = self.classifier.predict_proba(X)
            vals = p[:, 1] if p.shape[1] > 1 else np.full(len(df), float(self.classifier.classes_[0]))
        else:
            vals = self.classifier.predict_proba(X)
        vals = np.clip(vals, 1e-6, 1.0 - 1e-6)
        if self.calibrator is not None:
            vals = np.clip(self.calibrator.predict(vals), 1e-6, 1.0 - 1e-6)
        return vals

    def predict_fitness(self, df: pd.DataFrame) -> np.ndarray:
        if self.regressor is None:
            raise ValueError("DMSFunctionPrior is not fitted")
        X = featurize_variants(df)
        vals = self.regressor.predict(X)
        return np.clip(np.asarray(vals, dtype=float), 0.0, 1.0)

    def uncertainty(self, df: pd.DataFrame) -> np.ndarray:
        p = self.predict_proba(df)
        return p * (1.0 - p)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "DMSFunctionPrior":
        with open(path, "rb") as f:
            return pickle.load(f)


class _NumpyPrior:
    """Small fallback prior when sklearn is unavailable."""

    def __init__(self, df: pd.DataFrame):
        self.mean_viable = float(df["viable"].mean()) if len(df) else 0.5
        self.mean_fitness = float(df["fitness_norm"].mean()) if len(df) else 0.5

    def predict_proba(self, X):
        return np.full(X.shape[0], self.mean_viable)

    def predict(self, X):
        return np.full(X.shape[0], self.mean_fitness)


def evaluate_prior(prior: DMSFunctionPrior, test_df: pd.DataFrame) -> dict:
    """Evaluate prior with sklearn metrics when available."""
    p = prior.predict_proba(test_df)
    f = prior.predict_fitness(test_df)
    y = test_df["viable"].to_numpy(dtype=int)
    fitness = test_df["fitness_norm"].to_numpy(dtype=float)
    out = {"n": int(len(test_df))}
    try:
        from scipy.stats import spearmanr
        from sklearn.metrics import average_precision_score, roc_auc_score

        out["auroc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan
        out["auprc"] = float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else np.nan
        out["spearman"] = float(spearmanr(fitness, f, nan_policy="omit").statistic)
    except Exception:
        out["auroc"] = np.nan
        out["auprc"] = np.nan
        out["spearman"] = np.nan
    return out
