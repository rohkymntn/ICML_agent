"""Baselines for function-preserving protein editing."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .editing_tasks import EditingTask, objective_score
from .editguard_prior import DMSFunctionPrior
from .editguard_sampling import select_guided_edits


def random_edits(candidates: pd.DataFrame, k: int, seed: int = 0) -> pd.DataFrame:
    if len(candidates) == 0 or k <= 0:
        return candidates.iloc[:0].copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(candidates.index.to_numpy(), size=min(k, len(candidates)), replace=False)
    out = candidates.loc[idx].copy()
    out["method"] = "random"
    return out.reset_index(drop=True)


def novelty_edits(candidates: pd.DataFrame, k: int) -> pd.DataFrame:
    out = candidates.sort_values("mutation_distance", ascending=False).head(k).copy()
    out["method"] = "novelty"
    return out.reset_index(drop=True)


def objective_edits(candidates: pd.DataFrame, task: EditingTask, k: int) -> pd.DataFrame:
    out = candidates.copy()
    out["objective_score"] = objective_score(out, task.objective)
    out = out.nlargest(min(k, len(out)), "objective_score")
    out["method"] = "objective_only"
    return out.reset_index(drop=True)


def dms_prior_rerank(candidates: pd.DataFrame, prior: DMSFunctionPrior, k: int) -> pd.DataFrame:
    out = candidates.copy()
    out["prior_function_prob"] = prior.predict_proba(out)
    out["prior_fitness"] = prior.predict_fitness(out)
    out = out.nlargest(min(k, len(out)), "prior_function_prob")
    out["method"] = "dms_prior_rerank"
    return out.reset_index(drop=True)


def editguard_guided(candidates: pd.DataFrame, task: EditingTask, prior: DMSFunctionPrior, k: int) -> pd.DataFrame:
    out = select_guided_edits(candidates, task, prior, k=k)
    out["method"] = "editguard_guided"
    return out.reset_index(drop=True)


def external_generator_rerank(
    generated: pd.DataFrame,
    task: EditingTask,
    prior: DMSFunctionPrior,
    k: int,
    generator_name: str = "proteinmpnn",
) -> pd.DataFrame:
    """Rerank externally generated candidates, e.g. ProteinMPNN outputs.

    The input table should already be mapped to canonical DMS-style columns
    (`dataset_id`, `mutation_notation`, `mutation_distance`) and can include
    measured `fitness_norm`/`viable` when available for evaluation.
    """
    if len(generated) == 0:
        return generated.iloc[:0].copy()
    sub = generated[generated["dataset_id"].astype(str) == task.dataset_id].copy()
    sub = sub[sub["mutation_distance"] <= task.edit_budget]
    if len(sub) == 0:
        return sub
    sub["prior_function_prob"] = prior.predict_proba(sub)
    sub["prior_fitness"] = prior.predict_fitness(sub)
    sub = sub.nlargest(min(k, len(sub)), "prior_function_prob")
    sub["method"] = f"{generator_name}_dms_rerank"
    return sub.reset_index(drop=True)


def run_edit_baselines(
    candidates: pd.DataFrame,
    task: EditingTask,
    prior: DMSFunctionPrior,
    k: int = 50,
    seed: int = 0,
    external_candidates: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Run lightweight editing baselines on a measured candidate pool."""
    baselines = {
        "random": random_edits(candidates, k, seed=seed),
        "novelty": novelty_edits(candidates, k),
        "objective_only": objective_edits(candidates, task, k),
        "dms_prior_rerank": dms_prior_rerank(candidates, prior, k),
        "editguard_guided": editguard_guided(candidates, task, prior, k),
    }
    if external_candidates is not None and len(external_candidates) > 0:
        baselines["proteinmpnn_dms_rerank"] = external_generator_rerank(
            external_candidates,
            task,
            prior,
            k,
            generator_name="proteinmpnn",
        )
    return baselines
