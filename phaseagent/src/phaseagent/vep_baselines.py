"""Pre-computed VEP (variant effect predictor) rerank baselines.

ProteinGym v1.3 ships per-variant predictions from ~50 zero-shot models in
wide-format CSVs (one CSV per assay, one column per model). We use those
pre-computed scores as honest published baselines rather than re-implementing
or re-running the models ourselves.

Default headline picks: ``Tranception_L`` (autoregressive PLM with retrieval,
the standard ProteinGym-cited Tranception variant) and ``EVE_ensemble``
(MSA/evolutionary ensemble; the standard EVE benchmark variant). Other
columns are accessible via ``score_column``.

Each method joins the candidate pool to the zero-shot CSV by mutation
notation, ranks by the model's score, and returns the top-k. Selection is
restricted to candidates present in both the candidate pool and the
zero-shot table — coverage is reported per task so reviewers can audit
whether any method silently dropped variants.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .editing_tasks import EditingTask


HEADLINE_VEP_METHODS: dict[str, str] = {
    # method label → column name in the zero-shot CSV
    "tranception_l_rerank": "Tranception_L",
    "eve_ensemble_rerank": "EVE_ensemble",
}


def load_zero_shot_for_dataset(
    zero_shot_dir: Path,
    dataset_id: str,
) -> pd.DataFrame | None:
    """Return the per-assay zero-shot scores CSV, or None if missing."""
    path = Path(zero_shot_dir) / f"{dataset_id}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def vep_score_pool(
    pool: pd.DataFrame,
    zero_shot_df: pd.DataFrame,
    score_column: str,
    pool_key: str = "mutation_notation",
    zs_key: str = "mutant",
) -> pd.DataFrame:
    """Annotate ``pool`` with the named VEP column.

    Returns ``pool`` with an extra column ``vep_score`` (a copy of
    ``score_column``); rows whose key is missing from ``zero_shot_df`` get
    ``NaN`` and are dropped before ranking. The original notation is preserved
    so downstream metrics keep working.
    """
    if score_column not in zero_shot_df.columns:
        raise KeyError(
            f"score column {score_column!r} not in zero-shot CSV; available: "
            f"{sorted(zero_shot_df.columns)[:10]}…"
        )
    sub = zero_shot_df[[zs_key, score_column]].drop_duplicates(zs_key)
    sub = sub.rename(columns={zs_key: pool_key, score_column: "vep_score"})
    out = pool.merge(sub, on=pool_key, how="left")
    return out


def rerank_by_vep(
    pool: pd.DataFrame,
    zero_shot_df: pd.DataFrame,
    method_label: str,
    score_column: str,
    k: int,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """Score the pool with one VEP column and return the top-k.

    Selection drops rows missing a VEP score; the caller can compute
    coverage from ``len(returned) / len(pool)``.
    """
    scored = vep_score_pool(pool, zero_shot_df, score_column).dropna(subset=["vep_score"])
    if len(scored) == 0:
        return scored.assign(method=method_label)
    if higher_is_better:
        top = scored.nlargest(min(k, len(scored)), "vep_score")
    else:
        top = scored.nsmallest(min(k, len(scored)), "vep_score")
    out = top.copy()
    out["method"] = method_label
    return out.reset_index(drop=True)


def run_all_vep_methods(
    pool: pd.DataFrame,
    zero_shot_df: pd.DataFrame,
    k: int,
    methods: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run every VEP method in ``methods`` on the same pool.

    Returns ``{method_label: selection}``. Default methods are
    ``HEADLINE_VEP_METHODS``.
    """
    methods = methods or HEADLINE_VEP_METHODS
    out: dict[str, pd.DataFrame] = {}
    for label, column in methods.items():
        if column not in zero_shot_df.columns:
            continue
        out[label] = rerank_by_vep(pool, zero_shot_df, label, column, k)
    return out


def vep_coverage(pool: pd.DataFrame, zero_shot_df: pd.DataFrame, score_column: str) -> float:
    """Fraction of pool variants with a non-null VEP score."""
    if len(pool) == 0:
        return 0.0
    scored = vep_score_pool(pool, zero_shot_df, score_column)
    return float(scored["vep_score"].notna().mean())
