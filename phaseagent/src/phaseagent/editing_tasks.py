"""Function-preserving protein editing task construction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .mutations import parse_mutation_notation


@dataclass(frozen=True)
class EditingTask:
    dataset_id: str
    objective: str
    edit_budget: int
    protected_positions: tuple[int, ...] = ()
    forbidden_residues: tuple[str, ...] = ()


def mutation_positions(mutation_notation) -> tuple[int, ...]:
    """Return sorted 1-indexed positions touched by a mutation notation."""
    positions = []
    for tok in parse_mutation_notation(mutation_notation):
        try:
            positions.append(int(tok[1:-1]))
        except ValueError:
            continue
    return tuple(sorted(set(positions)))


def mutation_alts(mutation_notation) -> tuple[str, ...]:
    """Return alternate amino acids used by a mutation notation."""
    return tuple(tok[-1].upper() for tok in parse_mutation_notation(mutation_notation))


def add_edit_annotations(df: pd.DataFrame) -> pd.DataFrame:
    """Add reusable edit-task annotations to a canonical DMS dataframe."""
    out = df.copy()
    out["edit_positions"] = out["mutation_notation"].apply(mutation_positions)
    out["n_edit_positions"] = out["edit_positions"].apply(len)
    out["edit_alts"] = out["mutation_notation"].apply(mutation_alts)
    return out


def infer_fragile_positions(
    df: pd.DataFrame,
    quantile: float = 0.25,
    min_single_mutants: int = 2,
) -> tuple[int, ...]:
    """Infer fragile positions from single-mutant DMS effects."""
    annotated = add_edit_annotations(df)
    singles = annotated[annotated["mutation_distance"] == 1].copy()
    if len(singles) == 0:
        return ()
    singles["position"] = singles["edit_positions"].apply(lambda p: p[0] if p else np.nan)
    per_pos = (
        singles.dropna(subset=["position"])
        .groupby("position")["fitness_norm"]
        .agg(mean_fitness="mean", n="size")
        .reset_index()
    )
    per_pos = per_pos[per_pos["n"] >= min_single_mutants]
    if len(per_pos) == 0:
        return ()
    cutoff = float(per_pos["mean_fitness"].quantile(quantile))
    return tuple(int(x) for x in per_pos[per_pos["mean_fitness"] <= cutoff]["position"])


def candidate_pool_for_task(
    df: pd.DataFrame,
    task: EditingTask,
) -> pd.DataFrame:
    """Return measured DMS variants that satisfy task constraints."""
    sub = df[df["dataset_id"] == task.dataset_id].copy() if "dataset_id" in df.columns else df.copy()
    if len(sub) == 0:
        return sub
    sub = add_edit_annotations(sub)
    sub = sub[sub["mutation_distance"] > 0]
    sub = sub[sub["mutation_distance"] <= task.edit_budget]
    if task.protected_positions:
        protected = set(task.protected_positions)
        sub = sub[sub["edit_positions"].apply(lambda ps: protected.isdisjoint(ps))]
    if task.forbidden_residues:
        forbidden = set(a.upper() for a in task.forbidden_residues)
        sub = sub[sub["edit_alts"].apply(lambda aas: forbidden.isdisjoint(aas))]
    return sub.reset_index(drop=True)


def objective_score(df: pd.DataFrame, objective: str) -> np.ndarray:
    """Compute a normalized user-objective score for candidate edits."""
    if len(df) == 0:
        return np.array([], dtype=float)
    objective = objective.lower()
    distance = df["mutation_distance"].to_numpy(dtype=float)
    if objective == "novelty":
        denom = max(float(np.nanmax(distance)), 1.0)
        return distance / denom
    if objective == "conservative":
        denom = max(float(np.nanmax(distance)), 1.0)
        return 1.0 - distance / denom
    if objective == "motif_avoidance":
        return np.ones(len(df), dtype=float)
    if objective == "fragility_aware":
        return np.ones(len(df), dtype=float)
    if objective == "humanization_like":
        # Placeholder proxy until a target human distribution is supplied: prefer
        # common non-aromatic/non-cysteine substitutions to avoid liability edits.
        alts = df.get("edit_alts", pd.Series([()] * len(df)))
        return alts.apply(lambda xs: float(all(a not in {"C", "W", "M"} for a in xs))).to_numpy()
    return np.zeros(len(df), dtype=float)


def constraint_satisfied(df: pd.DataFrame, task: EditingTask) -> np.ndarray:
    """Return per-candidate boolean constraint satisfaction."""
    if len(df) == 0:
        return np.array([], dtype=bool)
    annotated = add_edit_annotations(df) if "edit_positions" not in df.columns else df
    ok = annotated["mutation_distance"].to_numpy(dtype=float) <= task.edit_budget
    if task.protected_positions:
        protected = set(task.protected_positions)
        ok &= annotated["edit_positions"].apply(lambda ps: protected.isdisjoint(ps)).to_numpy(dtype=bool)
    if task.forbidden_residues:
        forbidden = set(a.upper() for a in task.forbidden_residues)
        ok &= annotated["edit_alts"].apply(lambda aas: forbidden.isdisjoint(aas)).to_numpy(dtype=bool)
    return ok


def build_editing_tasks(
    df: pd.DataFrame,
    budgets: Iterable[int] = (1, 2, 3, 5),
    objectives: Iterable[str] = ("novelty", "fragility_aware", "motif_avoidance"),
    min_candidates: int = 20,
    splits: "pd.DataFrame | None" = None,
    split_filter: "str | None" = None,
) -> pd.DataFrame:
    """Build an editing-task table from measured DMS assays.

    If ``splits`` is provided, every output row gets a ``split`` column. If
    ``split_filter`` is also provided, only datasets assigned to that split
    contribute rows — used to enforce held-out evaluation discipline.
    """
    if split_filter is not None and splits is None:
        raise ValueError("split_filter requires splits to be provided")
    if splits is not None:
        from .edit_splits import VALID_SPLITS, filter_by_split

        if split_filter is not None:
            df = filter_by_split(df, splits, split_filter)
        elif "dataset_id" not in df.columns:
            raise ValueError("df must have a 'dataset_id' column when splits is given")
        splits_map = dict(zip(splits["dataset_id"].astype(str), splits["split"].astype(str)))
        if split_filter is not None and split_filter not in VALID_SPLITS:
            raise ValueError(f"Unknown split_filter {split_filter!r}")
    else:
        splits_map = None
    rows = []
    for dataset_id, sub in df.groupby("dataset_id"):
        fragile = infer_fragile_positions(sub)
        for budget in budgets:
            for objective in objectives:
                protected = fragile if objective == "fragility_aware" else ()
                forbidden = ("C",) if objective == "motif_avoidance" else ()
                task = EditingTask(
                    dataset_id=str(dataset_id),
                    objective=str(objective),
                    edit_budget=int(budget),
                    protected_positions=protected,
                    forbidden_residues=forbidden,
                )
                pool = candidate_pool_for_task(df, task)
                if len(pool) < min_candidates:
                    continue
                row = {
                    "dataset_id": dataset_id,
                    "objective": objective,
                    "edit_budget": int(budget),
                    "protected_positions": ",".join(map(str, protected)),
                    "forbidden_residues": ",".join(forbidden),
                    "n_candidates": int(len(pool)),
                }
                if splits_map is not None:
                    row["split"] = splits_map.get(str(dataset_id), "unassigned")
                rows.append(row)
    return pd.DataFrame(rows)


def task_from_row(row: pd.Series | dict) -> EditingTask:
    protected_value = row.get("protected_positions", "")
    forbidden_value = row.get("forbidden_residues", "")
    protected_raw = "" if pd.isna(protected_value) else str(protected_value)
    forbidden_raw = "" if pd.isna(forbidden_value) else str(forbidden_value)
    protected = tuple(int(x) for x in protected_raw.split(",") if x.strip())
    forbidden = tuple(x.strip() for x in forbidden_raw.split(",") if x.strip())
    return EditingTask(
        dataset_id=str(row["dataset_id"]),
        objective=str(row["objective"]),
        edit_budget=int(row["edit_budget"]),
        protected_positions=protected,
        forbidden_residues=forbidden,
    )
