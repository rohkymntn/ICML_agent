"""Prompted function-preserving editing interface."""
from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd

from .editing_tasks import EditingTask
from .edit_generators import (
    ProposalConfig,
    generate_many_then_rerank,
    guided_local_generation,
    infer_wildtype_sequence,
    join_generated_to_dms,
    observed_single_mutation_tokens,
    plm_masked_proposal_generation,
)
from .editguard_prior import DMSFunctionPrior


@dataclass(frozen=True)
class PromptedEditRequest:
    dataset_id: str
    prompt: str
    objective: str
    edit_budget: int
    forbidden_residues: tuple[str, ...] = ()
    protected_positions: tuple[int, ...] = ()
    n_candidates: int = 20
    coverage_mode: str = "dms_evaluable"


def parse_edit_prompt(
    dataset_id: str,
    prompt: str,
    edit_budget: int = 3,
    n_candidates: int = 20,
    protected_positions: tuple[int, ...] = (),
) -> PromptedEditRequest:
    """Convert a simple natural-language prompt into structured edit controls."""
    text = prompt.lower()
    objective = "novelty"
    forbidden: tuple[str, ...] = ()
    if "fragile" in text or "avoid sensitive" in text:
        objective = "fragility_aware"
    if "cysteine" in text or "cys" in text:
        objective = "motif_avoidance"
        forbidden = ("C",)
    if "humanize" in text or "humanise" in text:
        objective = "humanization_like"
    if "conservative" in text or "minimal" in text:
        objective = "conservative"
    m = re.search(r"(\d+)\s*(?:mutation|mutations|edit|edits)", text)
    budget = int(m.group(1)) if m else edit_budget
    return PromptedEditRequest(
        dataset_id=dataset_id,
        prompt=prompt,
        objective=objective,
        edit_budget=budget,
        forbidden_residues=forbidden,
        protected_positions=protected_positions,
        n_candidates=n_candidates,
    )


def generate_prompted_edits(
    dms_df: pd.DataFrame,
    prior: DMSFunctionPrior,
    request: PromptedEditRequest,
    method: str = "guided_local_generation",
    seed: int = 0,
) -> pd.DataFrame:
    """Generate prompted edits and attach measured DMS labels when available."""
    wt = infer_wildtype_sequence(dms_df, request.dataset_id)
    if wt is None:
        return pd.DataFrame()
    task = EditingTask(
        dataset_id=request.dataset_id,
        objective=request.objective,
        edit_budget=request.edit_budget,
        protected_positions=request.protected_positions,
        forbidden_residues=request.forbidden_residues,
    )
    allowed = observed_single_mutation_tokens(dms_df, task) if request.coverage_mode == "dms_evaluable" else None
    config = ProposalConfig(
        n_candidates=request.n_candidates,
        max_budget=request.edit_budget,
        seed=seed,
        coverage_mode=request.coverage_mode,
    )
    if method == "guided_local_generation":
        generated = guided_local_generation(request.dataset_id, wt, task, prior, config, allowed_tokens=allowed)
    elif method == "generate_many_then_rerank":
        generated = generate_many_then_rerank(
            request.dataset_id,
            wt,
            task,
            prior,
            n_generate=max(request.n_candidates * 10, request.n_candidates),
            k=request.n_candidates,
            seed=seed,
            allowed_tokens=allowed,
        )
    elif method == "esm2_masked_marginal_proxy":
        generated = plm_masked_proposal_generation(
            request.dataset_id,
            wt,
            task,
            n_candidates=request.n_candidates,
            seed=seed,
            allowed_tokens=allowed,
        )
    else:
        raise ValueError(f"Unknown prompted edit method: {method}")
    if len(generated) == 0:
        return generated
    generated = join_generated_to_dms(generated, dms_df)
    generated["prompt"] = request.prompt
    generated["structured_objective"] = request.objective
    generated["forbidden_residues"] = ",".join(request.forbidden_residues)
    generated["protected_positions"] = ",".join(map(str, request.protected_positions))
    generated["prior_function_prob"] = prior.predict_proba(generated)
    generated["prior_fitness"] = prior.predict_fitness(generated)
    generated["prior_uncertainty"] = prior.uncertainty(generated)
    return generated
