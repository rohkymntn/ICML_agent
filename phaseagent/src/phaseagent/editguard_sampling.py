"""Function-guided sampling for EditGuard."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .editing_tasks import EditingTask, constraint_satisfied, objective_score
from .editguard_prior import DMSFunctionPrior


def guided_energy(
    candidates: pd.DataFrame,
    task: EditingTask,
    prior: DMSFunctionPrior,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 2.0,
    kappa: float = 0.0,
) -> np.ndarray:
    """Lower is better energy for constrained function-preserving edits."""
    p = prior.predict_proba(candidates)
    obj = objective_score(candidates, task.objective)
    ok = constraint_satisfied(candidates, task)
    uncertainty = prior.uncertainty(candidates)
    penalty = np.where(ok, 0.0, 1.0)
    return -alpha * np.log(np.clip(p, 1e-6, 1.0)) - beta * obj + gamma * penalty + kappa * uncertainty


def guided_scores(
    candidates: pd.DataFrame,
    task: EditingTask,
    prior: DMSFunctionPrior,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 2.0,
    kappa: float = 0.0,
) -> pd.DataFrame:
    """Return candidates annotated with EditGuard guidance scores."""
    out = candidates.copy()
    out["prior_function_prob"] = prior.predict_proba(out)
    out["prior_fitness"] = prior.predict_fitness(out)
    out["prior_uncertainty"] = prior.uncertainty(out)
    out["objective_score"] = objective_score(out, task.objective)
    out["constraint_satisfied"] = constraint_satisfied(out, task)
    out["editguard_energy"] = guided_energy(
        out,
        task,
        prior,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        kappa=kappa,
    )
    out["editguard_score"] = -out["editguard_energy"]
    return out


def select_guided_edits(
    candidates: pd.DataFrame,
    task: EditingTask,
    prior: DMSFunctionPrior,
    k: int = 50,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 2.0,
    kappa: float = 0.0,
) -> pd.DataFrame:
    """Select top-k edits under the EditGuard guidance objective."""
    if len(candidates) == 0 or k <= 0:
        return candidates.iloc[:0].copy()
    scored = guided_scores(candidates, task, prior, alpha=alpha, beta=beta, gamma=gamma, kappa=kappa)
    return scored.nsmallest(min(k, len(scored)), "editguard_energy").reset_index(drop=True)


def softmax_sample_guided_edits(
    candidates: pd.DataFrame,
    task: EditingTask,
    prior: DMSFunctionPrior,
    k: int = 50,
    temperature: float = 1.0,
    seed: int = 0,
    **kwargs,
) -> pd.DataFrame:
    """Sample edits with probability proportional to guided score."""
    if len(candidates) == 0 or k <= 0:
        return candidates.iloc[:0].copy()
    rng = np.random.default_rng(seed)
    scored = guided_scores(candidates, task, prior, **kwargs)
    logits = -scored["editguard_energy"].to_numpy(dtype=float) / max(temperature, 1e-6)
    logits = logits - np.nanmax(logits)
    probs = np.exp(logits)
    probs = probs / probs.sum()
    n = min(k, len(scored))
    idx = rng.choice(scored.index.to_numpy(), size=n, replace=False, p=probs)
    return scored.loc[idx].sort_values("editguard_energy").reset_index(drop=True)
