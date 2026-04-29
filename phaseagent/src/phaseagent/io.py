"""DMS CSV ingestion with column inference and manifest loading."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

_FITNESS_CANDIDATES = [
    "DMS_score",
    "fitness",
    "score",
    "selection",
    "log_enrichment",
    "log_fitness",
    "fit",
    "y",
]
_MUTATION_CANDIDATES = ["mutant", "mutation", "mutations", "variant", "mut"]
_SEQUENCE_CANDIDATES = ["mutated_sequence", "sequence", "seq", "aa_seq"]
_WT_CANDIDATES = ["wildtype_sequence", "target_seq", "wt_sequence", "wildtype"]


def _first_match(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def infer_fitness_column(df: pd.DataFrame) -> str:
    col = _first_match(df, _FITNESS_CANDIDATES)
    if col is None:
        numeric = df.select_dtypes("number")
        if numeric.shape[1] == 0:
            raise ValueError("No fitness column found and no numeric columns to fall back to.")
        col = numeric.columns[0]
    return col


def infer_mutation_column(df: pd.DataFrame) -> Optional[str]:
    return _first_match(df, _MUTATION_CANDIDATES)


def infer_sequence_column(df: pd.DataFrame) -> Optional[str]:
    return _first_match(df, _SEQUENCE_CANDIDATES)


def infer_wildtype_column(df: pd.DataFrame) -> Optional[str]:
    return _first_match(df, _WT_CANDIDATES)


def load_dms_csv(path: str, dataset_id: Optional[str] = None) -> pd.DataFrame:
    """Load a DMS CSV/TSV/Parquet file. Adds dataset_id if supplied."""
    p = Path(path)
    if p.suffix == ".tsv":
        df = pd.read_csv(p, sep="\t")
    elif p.suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    if dataset_id is not None:
        df["dataset_id"] = dataset_id
    return df


def load_dataset_manifest(config_path: str) -> List[Dict[str, Any]]:
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}
    return data.get("datasets", [])
