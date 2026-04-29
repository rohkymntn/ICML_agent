"""ProteinGym substitution benchmark loader.

Each per-assay CSV in ProteinGym's substitution archive contains columns
`mutant`, `mutated_sequence`, `DMS_score` (and often `DMS_score_bin`). We
funnel them through the canonical preprocessing pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from tqdm import tqdm

from .preprocessing import preprocess_dataset


def discover_proteingym_csvs(raw_dir: Path) -> list[Path]:
    csvs = sorted(Path(raw_dir).rglob("*.csv"))
    out = []
    for c in csvs:
        name = c.name.lower()
        if name.startswith("dms_substitutions") or "reference" in name:
            continue
        out.append(c)
    return out


def build_proteingym_dataset(
    raw_dir: Path,
    fitness_normalization: str = "rank",
    threshold_mode: str = "quantile",
    threshold_value: float = 0.75,
    n_datasets: Optional[int] = None,
    min_rows: int = 100,
    max_rows: int = 200_000,
    random_state: int = 0,
    min_distance_shells: int = 3,
    min_count_per_shell: int = 30,
    multi_mutant_only: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build the canonical parquet from a ProteinGym substitution archive.

    `multi_mutant_only`: drop assays whose variants live at a single mutation
    distance — the phase-boundary fit needs ≥3 well-populated distance shells.
    """
    csvs = discover_proteingym_csvs(Path(raw_dir))
    rows, summary = [], []
    config = {
        "fitness_normalization": fitness_normalization,
        "threshold_mode": threshold_mode,
        "threshold_value": threshold_value,
        "fitness_col": "DMS_score",
        "mutation_col": "mutant",
        "sequence_col": "mutated_sequence",
    }
    accepted = 0
    for path in tqdm(csvs, desc="datasets"):
        ds_id = path.stem
        if n_datasets is not None and accepted >= n_datasets:
            break
        try:
            raw = pd.read_csv(path)
            if len(raw) < min_rows:
                summary.append({"dataset_id": ds_id, "n": int(len(raw)), "status": "skipped_too_small", "source": str(path)})
                continue
            if len(raw) > max_rows:
                raw = raw.sample(max_rows, random_state=random_state).reset_index(drop=True)
            df = preprocess_dataset(raw, ds_id, config)
            if multi_mutant_only:
                shell_counts = df["mutation_distance"].value_counts()
                ok_shells = (shell_counts >= min_count_per_shell).sum()
                if ok_shells < min_distance_shells:
                    summary.append({
                        "dataset_id": ds_id,
                        "n": int(len(df)),
                        "status": f"skipped_single_distance:{ok_shells}_shells",
                        "source": str(path),
                    })
                    continue
            rows.append(df)
            summary.append({"dataset_id": ds_id, "n": int(len(df)), "status": "ok", "source": str(path)})
            accepted += 1
        except Exception as exc:
            summary.append({"dataset_id": ds_id, "n": 0, "status": f"error:{type(exc).__name__}:{exc}", "source": str(path)})
    full = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return full, pd.DataFrame(summary)
