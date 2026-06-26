"""Megascale stability-editing tasks (3-tier protocol).

Per the field-mandated multi-mutant evaluation protocol (ThermoMPNN-D 2025,
MULTI-evolve critique 2026), we evaluate stability-editing methods in three
tiers:

- **T1 — single test**: Megascale ``dataset3_single`` test split (ThermoMPNN's
  published split). Pure single-mutant prediction sanity check; the additive
  baseline equals the model here by construction.
- **T2 — double mutants**: Megascale ``dataset2`` double-mutant subset.
  k=1 → k=2 generalization. Always reported alongside the additive baseline,
  with the headline being Δ(model − additive) on stabilizing-pair recall@k
  and Spearman in the |ΔΔG_epistasis| > 0.5 kcal/mol slice.
- **T3 — k≥2 extrapolation**: PTmul-NR (k≥2 multi-mutants from
  ProThermDB-derived sources). Loaded separately because the source is not
  on Hugging Face.

Editing tasks here are stability-specific: protect positions that are already
strongly stabilizing in single-mutant DMS (don't break what works), and
forbid the destabilizing-liability residues C/M/W/P (cysteine oxidation,
methionine oxidation, tryptophan/proline conformational issues). Each task
is parameterized by edit budget k and objective.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .editing_tasks import EditingTask, candidate_pool_for_task
from .mutations import parse_mutation_notation


STABILITY_LIABILITY_AAS = ("C", "M", "W", "P")
STABILITY_OBJECTIVES = (
    "stabilize",        # maximize predicted ΔΔG (most stabilizing top-k)
    "destab_avoid",     # minimize predicted destabilization (avoid worst losers)
    "stabilize_safe",   # stabilize while avoiding liability residues
)


@dataclass(frozen=True)
class MegascaleTaskConfig:
    """Configuration for one Megascale editing task."""

    dataset_id: str
    objective: str = "stabilize"
    edit_budget: int = 2
    protect_top_quartile_singles: bool = True
    forbid_liabilities: bool = True


def infer_strongly_stabilizing_positions(
    single_df: pd.DataFrame,
    quantile: float = 0.75,
    min_single_mutants: int = 2,
) -> tuple[int, ...]:
    """Return positions whose mean single-mutant stability is in the top quartile.

    These are the "don't break what works" positions — protected by default
    in stability-editing tasks because mutating away from them is more likely
    to lose stability than to gain it.
    """
    if "fitness_norm" not in single_df.columns:
        return ()
    sub = single_df[single_df.get("mutation_distance", 1) == 1].copy()
    if len(sub) == 0:
        return ()
    sub["position"] = sub["mutation_notation"].apply(
        lambda s: int(parse_mutation_notation(s)[0][1:-1])
        if parse_mutation_notation(s)
        and parse_mutation_notation(s)[0][1:-1].lstrip("-").isdigit()
        else np.nan
    )
    per_pos = (
        sub.dropna(subset=["position"])
        .groupby("position")["fitness_norm"]
        .agg(mean_fitness="mean", n="size")
        .reset_index()
    )
    per_pos = per_pos[per_pos["n"] >= min_single_mutants]
    if len(per_pos) == 0:
        return ()
    cutoff = float(per_pos["mean_fitness"].quantile(quantile))
    return tuple(int(x) for x in per_pos[per_pos["mean_fitness"] >= cutoff]["position"])


def build_megascale_task(
    config: MegascaleTaskConfig,
    single_df: pd.DataFrame,
) -> EditingTask:
    """Construct an EditingTask from a MegascaleTaskConfig and per-protein singles."""
    protected = ()
    forbidden = ()
    if config.protect_top_quartile_singles:
        protected = infer_strongly_stabilizing_positions(single_df)
    if config.forbid_liabilities or config.objective == "stabilize_safe":
        forbidden = STABILITY_LIABILITY_AAS
    return EditingTask(
        dataset_id=config.dataset_id,
        objective=config.objective,
        edit_budget=config.edit_budget,
        protected_positions=protected,
        forbidden_residues=forbidden,
    )


def build_megascale_editing_tasks(
    multi_df: pd.DataFrame,
    single_df: pd.DataFrame,
    *,
    budgets: Iterable[int] = (2, 3, 4, 5),
    objectives: Iterable[str] = STABILITY_OBJECTIVES,
    min_candidates: int = 20,
    max_proteins: int | None = None,
) -> pd.DataFrame:
    """Build the editing-task table for a Megascale tier.

    Each (protein, budget, objective) combination is one row, gated on having
    at least ``min_candidates`` measured multi-mutants in ``multi_df`` for
    that protein under the constraints.
    """
    rows = []
    proteins = multi_df["dataset_id"].unique() if "dataset_id" in multi_df.columns else []
    if max_proteins is not None:
        proteins = list(proteins)[:int(max_proteins)]
    for ds_id in proteins:
        ds_singles = single_df[single_df["dataset_id"] == ds_id] if "dataset_id" in single_df.columns else single_df
        for budget in budgets:
            for obj in objectives:
                cfg = MegascaleTaskConfig(
                    dataset_id=str(ds_id),
                    objective=str(obj),
                    edit_budget=int(budget),
                )
                task = build_megascale_task(cfg, ds_singles)
                pool = candidate_pool_for_task(multi_df, task)
                if len(pool) < min_candidates:
                    continue
                rows.append(
                    {
                        "dataset_id": str(ds_id),
                        "objective": str(obj),
                        "edit_budget": int(budget),
                        "protected_positions": ",".join(map(str, task.protected_positions)),
                        "forbidden_residues": ",".join(task.forbidden_residues),
                        "n_candidates": int(len(pool)),
                        "tier": "T2_doubles" if budget == 2 else "T3_high_k",
                    }
                )
    return pd.DataFrame(rows)


def stability_objective_score(
    df: pd.DataFrame,
    objective: str,
    additive_predictions: np.ndarray | None = None,
) -> np.ndarray:
    """Per-candidate objective score for stability-editing tasks.

    Always uses the *observed* fitness_norm column for the ground-truth
    ranking (since fitness_norm here is rank-normalized ΔΔG with positive =
    stabilizing). The optional ``additive_predictions`` argument is used by
    the ``additive_baseline`` method only — every other method ignores it.
    """
    if len(df) == 0:
        return np.array([], dtype=float)
    obj = objective.lower()
    obs = pd.to_numeric(df["fitness_norm"], errors="coerce").to_numpy(dtype=float)
    if obj == "stabilize" or obj == "stabilize_safe":
        return obs
    if obj == "destab_avoid":
        return 1.0 - obs  # Inverted: pick variants with lowest predicted destab.
    return np.zeros(len(df), dtype=float)
