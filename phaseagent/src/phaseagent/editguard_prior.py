"""DMS-trained function prior for EditGuard.

Features are deliberately chemistry/position/context-based and contain no
per-variant identifier (no notation hash, no dataset_id hash). Two variants
with the same chemistry, positions, and per-position DMS context produce
identical feature vectors regardless of notation order or dataset name.

Per-position DMS context (mean single-mutant fitness, fragile-position
membership) is computed from a ``dms_context`` argument that defaults to the
input dataframe — matching the standard "single-mutant DMS available at edit
time" assumption used in DMS-guided multi-mutant prediction.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Iterable

import numpy as np
import pandas as pd

from .editing_tasks import infer_fragile_positions
from .mutations import parse_mutation_notation
from .spectrum import AMINO_ACIDS, AA_TO_ID

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression, Ridge

    _HAS_SK = True
except Exception:  # pragma: no cover
    _HAS_SK = False


LIABILITY_AAS = frozenset("CMWP")
FEATURE_DIM = 3 + 20 + 20 + 2 + 1 + 1 + 1  # = 48


def _compute_position_means(dms_df: pd.DataFrame) -> dict[tuple[str, int], float]:
    """Mean per-(dataset_id, position) single-mutant fitness."""
    out: dict[tuple[str, int], list[float]] = {}
    if "dataset_id" not in dms_df.columns or "mutation_notation" not in dms_df.columns:
        return {}
    if "fitness_norm" not in dms_df.columns:
        return {}
    if "mutation_distance" in dms_df.columns:
        singles = dms_df[dms_df["mutation_distance"] == 1]
    else:
        singles = dms_df[dms_df["mutation_notation"].apply(
            lambda s: len(parse_mutation_notation(s)) == 1
        )]
    if len(singles) == 0:
        return {}
    for ds_id, sub in singles.groupby("dataset_id"):
        ds_key = str(ds_id)
        for notation, fit in zip(sub["mutation_notation"].astype(str), sub["fitness_norm"]):
            toks = parse_mutation_notation(notation)
            if len(toks) != 1:
                continue
            try:
                pos = int(toks[0][1:-1])
            except ValueError:
                continue
            f = pd.to_numeric(fit, errors="coerce")
            if pd.isna(f):
                continue
            out.setdefault((ds_key, pos), []).append(float(f))
    return {k: float(np.mean(v)) for k, v in out.items()}


def _compute_fragile_positions(
    dms_df: pd.DataFrame,
    quantile: float = 0.25,
) -> dict[str, frozenset[int]]:
    """Per-dataset fragile-position sets, frozen as immutable lookups."""
    out: dict[str, frozenset[int]] = {}
    if "dataset_id" not in dms_df.columns:
        return out
    for ds_id, sub in dms_df.groupby("dataset_id"):
        try:
            positions = infer_fragile_positions(sub, quantile=quantile)
        except Exception:
            positions = ()
        out[str(ds_id)] = frozenset(int(p) for p in positions)
    return out


def featurize_variants(
    df: pd.DataFrame,
    *,
    dms_context: pd.DataFrame | None = None,
) -> np.ndarray:
    """Chemistry/position/context features for the DMS function prior.

    Returns an ``(n, FEATURE_DIM)`` array. Layout (no per-variant identifiers):

    - 0..2:    distance, distance², log1p(distance)
    - 3..22:   per-token WT amino-acid counts / distance (20-D)
    - 23..42:  per-token MUT amino-acid counts / distance (20-D)
    - 43, 44:  mean / std of mutation positions, normalized by sequence length
    - 45:      fraction of tokens at fragile positions (per-dataset DMS-derived)
    - 46:      fraction of tokens whose mutant residue is in liability set CMWP
    - 47:      mean per-position single-mutant fitness across the variant's positions

    ``dms_context`` provides the DMS table used to derive per-(dataset, position)
    fragility / single-mutant-fitness lookups. Defaults to ``df`` itself.
    """
    n = len(df)
    if n == 0:
        return np.zeros((0, FEATURE_DIM), dtype=float)
    context = dms_context if dms_context is not None else df
    pos_means = _compute_position_means(context)
    fragile = _compute_fragile_positions(context)

    X = np.zeros((n, FEATURE_DIM), dtype=float)
    dist = pd.to_numeric(df["mutation_distance"], errors="coerce").to_numpy(dtype=float)
    dist = np.where(np.isnan(dist), 0.0, dist)
    X[:, 0] = dist
    X[:, 1] = dist ** 2
    X[:, 2] = np.log1p(np.maximum(dist, 0.0))

    nota = df["mutation_notation"].astype(str) if "mutation_notation" in df.columns else pd.Series([""] * n)
    ds_ids = df["dataset_id"].astype(str) if "dataset_id" in df.columns else pd.Series([""] * n)
    if "wildtype_sequence" in df.columns:
        seq_lens = (
            df["wildtype_sequence"]
            .fillna("")
            .astype(str)
            .replace({"nan": "", "None": ""})
            .str.len()
            .replace(0, 1)
        )
    else:
        seq_lens = pd.Series([1] * n, index=df.index)

    for i in range(n):
        toks = parse_mutation_notation(nota.iloc[i])
        if not toks:
            continue
        ds_key = ds_ids.iloc[i]
        ds_fragile = fragile.get(ds_key, frozenset())
        wt_counts = np.zeros(20)
        mut_counts = np.zeros(20)
        positions: list[int] = []
        liability = 0
        in_fragile = 0
        pos_mean_vals: list[float] = []
        for tok in toks:
            try:
                pos = int(tok[1:-1])
            except ValueError:
                continue
            wt_aa = tok[0].upper()
            mut_aa = tok[-1].upper()
            if wt_aa in AA_TO_ID:
                wt_counts[AA_TO_ID[wt_aa]] += 1
            if mut_aa in AA_TO_ID:
                mut_counts[AA_TO_ID[mut_aa]] += 1
            positions.append(pos)
            if mut_aa in LIABILITY_AAS:
                liability += 1
            if pos in ds_fragile:
                in_fragile += 1
            v = pos_means.get((ds_key, pos))
            if v is not None and np.isfinite(v):
                pos_mean_vals.append(v)
        d = max(len(toks), 1)
        seq_raw = seq_lens.iloc[i]
        try:
            seq_len = max(int(seq_raw), 1)
        except (TypeError, ValueError):
            seq_len = 1
        X[i, 3:23] = wt_counts / d
        X[i, 23:43] = mut_counts / d
        if positions:
            X[i, 43] = float(np.mean(positions)) / seq_len
            X[i, 44] = float(np.std(positions)) / seq_len if len(positions) > 1 else 0.0
        X[i, 45] = in_fragile / d
        X[i, 46] = liability / d
        X[i, 47] = float(np.mean(pos_mean_vals)) if pos_mean_vals else 0.5
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

    def fit(
        self,
        train_df: pd.DataFrame,
        calibrate_df: pd.DataFrame | None = None,
        dms_context: pd.DataFrame | None = None,
    ):
        """Fit on ``train_df``; ``dms_context`` defaults to train_df itself.

        ``dms_context`` is the DMS table used to derive per-(dataset, position)
        single-mutant context features. It must contain only train-split data
        to avoid label leakage from val/test.
        """
        if "viable" not in train_df.columns or "fitness_norm" not in train_df.columns:
            raise ValueError("train_df must contain viable and fitness_norm columns")
        context = dms_context if dms_context is not None else train_df
        X = featurize_variants(train_df, dms_context=context)
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
            elif len(np.unique(y_cls)) > 1:
                self.classifier = LogisticRegression(max_iter=1000).fit(X, y_cls)
            else:
                # Single-class training data — no useful classifier; remember
                # the constant label and skip fitting an sklearn model.
                self.classifier = _ConstantClassifier(int(y_cls[0]))
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
                if len(np.unique(calibrate_df["viable"])) > 1:
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
        n = len(df)
        if hasattr(self.classifier, "predict_proba"):
            p = np.asarray(self.classifier.predict_proba(X))
            if p.ndim == 1:
                vals = p
            elif p.ndim == 2 and p.shape[1] >= 2:
                vals = p[:, 1]
            elif p.ndim == 2 and p.shape[1] == 1:
                only_class = float(getattr(self.classifier, "classes_", [0])[0])
                vals = np.full(n, only_class)
            else:
                vals = np.full(n, 0.5)
        else:
            vals = np.asarray(self.classifier.predict(X), dtype=float) if hasattr(self.classifier, "predict") else np.full(n, 0.5)
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


class _ConstantClassifier:
    """Predict a single-class label (used when training labels have one class)."""

    def __init__(self, label: int):
        self.label = int(label)
        self.classes_ = np.array([self.label])

    def predict_proba(self, X):
        n = len(X)
        return np.full(n, float(self.label))

    def predict(self, X):
        return np.full(len(X), self.label)


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
