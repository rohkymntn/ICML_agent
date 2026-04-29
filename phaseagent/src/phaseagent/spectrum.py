"""Single-mutant spectrum features for Spectral PhaseAgent."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .mutations import parse_mutation_notation

AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
AA_TO_ID = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


@dataclass(frozen=True)
class MutationToken:
    wt_aa: str
    position: int
    mut_aa: str


def _parse_single_token(mutation_notation) -> MutationToken | None:
    toks = parse_mutation_notation(mutation_notation)
    if len(toks) != 1:
        return None
    tok = toks[0]
    try:
        return MutationToken(tok[0].upper(), int(tok[1:-1]), tok[-1].upper())
    except ValueError:
        return None


def extract_single_mutant_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return canonical rows that correspond to exactly one substitution."""
    if "mutation_distance" in df.columns:
        out = df[df["mutation_distance"] == 1].copy()
    elif "mutation_notation" in df.columns:
        out = df[df["mutation_notation"].apply(lambda x: len(parse_mutation_notation(x)) == 1)].copy()
    else:
        return df.iloc[:0].copy()
    return out.reset_index(drop=True)


def compute_wt_fitness(
    df: pd.DataFrame,
    fitness_col: str = "fitness_norm",
    fallback: float | None = None,
) -> float:
    """Infer wild-type fitness from distance-zero rows or a conservative convention."""
    if fitness_col not in df.columns:
        raise ValueError(f"Missing fitness column: {fitness_col}")
    if "mutation_distance" in df.columns:
        wt = df[df["mutation_distance"] == 0]
        if len(wt) > 0:
            return float(pd.to_numeric(wt[fitness_col], errors="coerce").mean())
    if fallback is not None:
        return float(fallback)
    values = pd.to_numeric(df[fitness_col], errors="coerce").dropna()
    if len(values) == 0:
        return 0.0
    # Rank/minmax-normalized ProteinGym data lives on [0, 1]; use the existing
    # project convention that absent WT is a mid-fitness reference.
    if values.min() >= 0.0 and values.max() <= 1.0:
        return 0.5
    return 0.0


def compute_single_mutant_effects(
    df: pd.DataFrame,
    fitness_col: str = "fitness_norm",
    wt_fitness: float | None = None,
) -> pd.DataFrame:
    """Add mutation-token metadata and first-order effect ``delta_f``."""
    singles = extract_single_mutant_rows(df)
    if len(singles) == 0:
        return singles.assign(
            position=pd.Series(dtype=int),
            wt_aa=pd.Series(dtype=str),
            mut_aa=pd.Series(dtype=str),
            delta_f=pd.Series(dtype=float),
        )
    f0 = compute_wt_fitness(df, fitness_col=fitness_col, fallback=wt_fitness)
    parsed = singles["mutation_notation"].apply(_parse_single_token)
    singles = singles[parsed.notna()].copy()
    parsed = parsed[parsed.notna()]
    singles["position"] = [p.position for p in parsed]
    singles["wt_aa"] = [p.wt_aa for p in parsed]
    singles["mut_aa"] = [p.mut_aa for p in parsed]
    singles["wt_fitness"] = f0
    singles["delta_f"] = pd.to_numeric(singles[fitness_col], errors="coerce") - f0
    return singles.dropna(subset=["delta_f"]).reset_index(drop=True)


def build_spectrum_tokens(
    df: pd.DataFrame,
    fitness_col: str = "fitness_norm",
    wt_fitness: float | None = None,
) -> pd.DataFrame:
    """Build one token per observed single mutant."""
    effects = compute_single_mutant_effects(df, fitness_col=fitness_col, wt_fitness=wt_fitness)
    if len(effects) == 0:
        return pd.DataFrame(
            columns=[
                "dataset_id",
                "position",
                "pos_norm",
                "wt_aa",
                "wt_aa_id",
                "mut_aa",
                "mut_aa_id",
                "delta_f",
                "observed",
            ]
        )
    max_pos = max(float(effects["position"].max()), 1.0)
    out = pd.DataFrame()
    out["dataset_id"] = effects["dataset_id"].values if "dataset_id" in effects.columns else "dataset"
    out["position"] = effects["position"].astype(int).values
    out["pos_norm"] = effects["position"].astype(float).values / max_pos
    out["wt_aa"] = effects["wt_aa"].values
    out["wt_aa_id"] = effects["wt_aa"].map(AA_TO_ID).fillna(-1).astype(int).values
    out["mut_aa"] = effects["mut_aa"].values
    out["mut_aa_id"] = effects["mut_aa"].map(AA_TO_ID).fillna(-1).astype(int).values
    out["delta_f"] = effects["delta_f"].astype(float).values
    out["observed"] = 1
    return out


def _safe_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "effect_mean": np.nan,
            "effect_std": np.nan,
            "effect_skew": np.nan,
            "effect_kurtosis": np.nan,
            "effect_min": np.nan,
            "effect_max": np.nan,
        }
    std = float(values.std(ddof=0))
    centered = values - float(values.mean())
    skew = float(np.mean(centered**3) / (std**3)) if std > 0 else 0.0
    kurt = float(np.mean(centered**4) / (std**4) - 3.0) if std > 0 else 0.0
    return {
        "effect_mean": float(values.mean()),
        "effect_std": std,
        "effect_skew": skew,
        "effect_kurtosis": kurt,
        "effect_min": float(values.min()),
        "effect_max": float(values.max()),
    }


def _entropy(values: np.ndarray, bins: int = 30) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0
    hist, _ = np.histogram(values, bins=min(bins, max(2, values.size)))
    p = hist.astype(float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log(p)).sum())


def _gini(values: Iterable[float]) -> float:
    arr = np.sort(np.abs(np.asarray(list(values), dtype=float)))
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or np.all(arr == 0):
        return 0.0
    n = arr.size
    return float((2.0 * np.arange(1, n + 1) @ arr) / (n * arr.sum()) - (n + 1) / n)


def site_tolerance_profile(
    df: pd.DataFrame,
    deleterious_cutoff: float = -0.05,
    fitness_col: str = "fitness_norm",
) -> pd.DataFrame:
    """Summarize single-mutant tolerance at each position."""
    effects = compute_single_mutant_effects(df, fitness_col=fitness_col)
    if len(effects) == 0:
        return pd.DataFrame(
            columns=[
                "dataset_id",
                "position",
                "mean_delta_f",
                "median_delta_f",
                "deleterious_fraction",
                "beneficial_fraction",
                "n_single_mutants",
            ]
        )
    grouped = effects.groupby(["dataset_id", "position"] if "dataset_id" in effects.columns else ["position"])
    out = grouped["delta_f"].agg(mean_delta_f="mean", median_delta_f="median", n_single_mutants="size").reset_index()
    frac = grouped["delta_f"].agg(
        deleterious_fraction=lambda x: float(np.mean(np.asarray(x) <= deleterious_cutoff)),
        beneficial_fraction=lambda x: float(np.mean(np.asarray(x) > 0.0)),
    ).reset_index()
    return out.merge(frac, on=[c for c in out.columns if c in frac.columns and c not in {"mean_delta_f", "median_delta_f", "n_single_mutants"}])


def spectrum_summary_features(
    df: pd.DataFrame,
    deleterious_cutoff: float = -0.05,
    neutral_width: float = 0.05,
    fitness_col: str = "fitness_norm",
) -> dict[str, float]:
    """Return spectrum-level summary features for one dataset."""
    effects = compute_single_mutant_effects(df, fitness_col=fitness_col)
    values = effects["delta_f"].to_numpy(dtype=float) if len(effects) else np.array([], dtype=float)
    stats = _safe_stats(values)
    site_profile = site_tolerance_profile(df, deleterious_cutoff=deleterious_cutoff, fitness_col=fitness_col)
    site_fragility = (
        -site_profile["mean_delta_f"].to_numpy(dtype=float)
        if len(site_profile)
        else np.array([], dtype=float)
    )
    out = {
        **stats,
        "n_single_mutants": int(len(effects)),
        "n_sites_observed": int(effects["position"].nunique()) if len(effects) else 0,
        "beneficial_fraction": float(np.mean(values > 0.0)) if values.size else np.nan,
        "deleterious_tail_mass": float(np.mean(values <= deleterious_cutoff)) if values.size else np.nan,
        "neutral_fraction": float(np.mean(np.abs(values) <= neutral_width)) if values.size else np.nan,
        "spectrum_entropy": _entropy(values),
        "site_fragility_gini": _gini(site_fragility),
    }
    if "dataset_id" in df.columns and len(df):
        out["dataset_id"] = df["dataset_id"].iloc[0]
    return out


def mutation_class_matrix(
    df: pd.DataFrame,
    fitness_col: str = "fitness_norm",
) -> pd.DataFrame:
    """Return a 20x20 table of mean effects for WT amino acid -> mutant amino acid."""
    effects = compute_single_mutant_effects(df, fitness_col=fitness_col)
    mat = pd.DataFrame(np.nan, index=AMINO_ACIDS, columns=AMINO_ACIDS, dtype=float)
    if len(effects) == 0:
        mat.index.name = "wt_aa"
        return mat
    grouped = effects.groupby(["wt_aa", "mut_aa"])["delta_f"].mean()
    for (wt, mut), val in grouped.items():
        if wt in mat.index and mut in mat.columns:
            mat.loc[wt, mut] = float(val)
    mat.index.name = "wt_aa"
    return mat


def deleterious_tail_mass(effects: Iterable[float], cutoff: float = -0.05) -> float:
    arr = np.asarray(list(effects), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr <= cutoff)) if arr.size else np.nan


def beneficial_fraction(effects: Iterable[float]) -> float:
    arr = np.asarray(list(effects), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr > 0.0)) if arr.size else np.nan


def spectrum_entropy(effects: Iterable[float], bins: int = 30) -> float:
    return _entropy(np.asarray(list(effects), dtype=float), bins=bins)


def spectrum_gini_fragility(site_effects: Iterable[float]) -> float:
    return _gini(site_effects)


def empirical_mgf(effects: Iterable[float], lambdas: Iterable[float]) -> pd.DataFrame:
    """Estimate ``E[exp(lambda X)]`` from single-mutant effects."""
    x = np.asarray(list(effects), dtype=float)
    x = x[np.isfinite(x)]
    lam = np.asarray(list(lambdas), dtype=float)
    if x.size == 0:
        vals = np.full_like(lam, np.nan, dtype=float)
    else:
        vals = np.array([float(np.mean(np.exp(np.clip(l * x, -700, 700)))) for l in lam])
    return pd.DataFrame({"lambda": lam, "mgf": vals})
