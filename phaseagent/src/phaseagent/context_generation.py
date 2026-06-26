"""Generation and baselines for DMS-conditioned protein diffusion."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from .dms_context_adapter import DMSContextAdapter
from .dms_context_adapter import parse_variant_target
from .dplm_backbone import AMINO_ACIDS, DPLMConfig, _allowed_aa_token_ids, _build_masked_sequence
from .editing_tasks import EditingTask, candidate_pool_for_task
from .edit_eval import evaluate_edit_selection, evaluate_generated_selection
from .editguard_dplm import join_to_dms_labels, propose_with_dplm, rerank_with_classifier_guidance
from .editguard_prior import DMSFunctionPrior
from .mutation_map_context import MutationMapContext
from .mutations import parse_mutation_notation
from .spectrum import AA_TO_ID


@dataclass(frozen=True)
class ContextGenerationConfig:
    n_mask_patterns: int = 16
    samples_per_pattern: int = 16
    temperature: float = 1.0
    context_scale: float = 1.0
    seed: int = 0
    batch_size: int = 8
    top_k: int = 50
    max_pool_candidates: int = 5000


def _sample_mask_pattern(
    wt_length: int,
    edit_budget: int,
    protected: frozenset[int],
    rng: np.random.Generator,
) -> tuple[int, ...]:
    editable = [i for i in range(1, wt_length + 1) if i not in protected]
    if len(editable) < edit_budget:
        return tuple(editable)
    return tuple(sorted(int(p) for p in rng.choice(editable, size=edit_budget, replace=False)))


def _forward_logits_hidden_batch(model, tokenizer, masked_sequences: list[str]):
    import torch

    inputs = tokenizer(list(masked_sequences), return_tensors="pt", padding=True)
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model(**inputs, output_hidden_states=True)
    hidden = out.hidden_states[-1].detach()
    return out.logits.detach(), hidden, inputs["input_ids"].detach()


def _masked_guided_logits_for_pattern(
    *,
    logits,
    hidden,
    adapter: DMSContextAdapter,
    context: MutationMapContext,
    pattern: tuple[int, ...],
    allowed_token_ids: list[int],
    forbidden_aas: frozenset[str],
    edit_budget: int,
    context_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    if not pattern:
        return np.zeros((0, 20)), np.zeros((0, 20)), np.zeros((0, 20))
    device = hidden.device
    pos = torch.tensor(list(pattern), dtype=torch.long, device=device)
    h = hidden.index_select(0, pos)
    ctx = torch.tensor(context.features_for_positions(pattern), dtype=torch.float32, device=device)
    budget = torch.full((len(pattern),), float(edit_budget), dtype=torch.float32, device=device)
    base = logits.index_select(0, pos)[:, allowed_token_ids].float()
    corr = adapter(h, ctx, budget)
    guided = base + float(context_scale) * corr
    # Hard constraints: no forbidden AAs and no WT self-substitution.
    for j, p in enumerate(pattern):
        wt_aa = context.wildtype_sequence[p - 1].upper()
        for aa, aa_idx in AA_TO_ID.items():
            if aa == wt_aa or aa in forbidden_aas:
                guided[j, aa_idx] = -1e9
    return (
        base.detach().cpu().numpy(),
        corr.detach().cpu().numpy(),
        guided.detach().cpu().numpy(),
    )


def _sample_variant_from_guided_logits(
    wt_sequence: str,
    pattern: tuple[int, ...],
    guided_logits: np.ndarray,
    base_logits: np.ndarray,
    rng: np.random.Generator,
    temperature: float,
) -> dict | None:
    chars = list(wt_sequence)
    toks: list[str] = []
    base_logps: list[float] = []
    guided_logps: list[float] = []
    for j, pos in enumerate(pattern):
        g = np.asarray(guided_logits[j], dtype=float) / max(float(temperature), 1e-6)
        if not np.isfinite(g).any() or np.nanmax(g) < -1e8:
            return None
        g = g - np.nanmax(g)
        probs = np.exp(g)
        probs[~np.isfinite(probs)] = 0.0
        if probs.sum() <= 0:
            return None
        probs = probs / probs.sum()
        aa_idx = int(rng.choice(len(AMINO_ACIDS), p=probs))
        alt = AMINO_ACIDS[aa_idx]
        ref = wt_sequence[pos - 1].upper()
        if alt == ref:
            continue
        chars[pos - 1] = alt
        toks.append(f"{ref}{pos}{alt}")
        guided_logps.append(float(np.log(max(probs[aa_idx], 1e-12))))
        b = np.asarray(base_logits[j], dtype=float)
        b = b - np.nanmax(b)
        bp = np.exp(b)
        bp = bp / max(float(bp.sum()), 1e-12)
        base_logps.append(float(np.log(max(bp[aa_idx], 1e-12))))
    if not toks:
        return None
    return {
        "mutation_notation": ",".join(toks),
        "mutated_sequence": "".join(chars),
        "mutation_distance": len(toks),
        "dplm_logp": float(np.mean(base_logps)) if base_logps else np.nan,
        "context_logp": float(np.mean(guided_logps)) if guided_logps else np.nan,
    }


def propose_with_context_adapter(
    wt_sequence: str,
    task: EditingTask,
    model,
    tokenizer,
    adapter: DMSContextAdapter,
    context: MutationMapContext,
    config: ContextGenerationConfig | None = None,
) -> pd.DataFrame:
    """Generate variants from frozen DPLM logits plus a trained DMS-context adapter."""
    config = config or ContextGenerationConfig()
    rng = np.random.default_rng(config.seed)
    protected = frozenset(int(p) for p in task.protected_positions)
    forbidden = frozenset(a.upper() for a in task.forbidden_residues)
    allowed_token_ids, allowed_aas = _allowed_aa_token_ids(tokenizer)
    if tuple(allowed_aas) != tuple(AMINO_ACIDS):
        raise RuntimeError("DPLM tokenizer AA order does not match context adapter AA order")

    patterns: list[tuple[int, ...]] = []
    masked: list[str] = []
    for _ in range(config.n_mask_patterns):
        pattern = _sample_mask_pattern(len(wt_sequence), int(task.edit_budget), protected, rng)
        if not pattern:
            continue
        patterns.append(pattern)
        masked.append(_build_masked_sequence(wt_sequence, pattern))
    rows: list[dict] = []
    if not patterns:
        return pd.DataFrame()
    adapter.eval()
    sample_rank = 0
    for start in range(0, len(masked), config.batch_size):
        chunk_patterns = patterns[start : start + config.batch_size]
        chunk_masked = masked[start : start + config.batch_size]
        logits, hidden, _ = _forward_logits_hidden_batch(model, tokenizer, chunk_masked)
        for b, pattern in enumerate(chunk_patterns):
            base, corr, guided = _masked_guided_logits_for_pattern(
                logits=logits[b],
                hidden=hidden[b],
                adapter=adapter,
                context=context,
                pattern=pattern,
                allowed_token_ids=allowed_token_ids,
                forbidden_aas=forbidden,
                edit_budget=int(task.edit_budget),
                context_scale=float(config.context_scale),
            )
            for s in range(config.samples_per_pattern):
                item = _sample_variant_from_guided_logits(
                    wt_sequence,
                    pattern,
                    guided,
                    base,
                    rng,
                    temperature=config.temperature,
                )
                if item is None:
                    continue
                item.update(
                    {
                        "dataset_id": task.dataset_id,
                        "proposal_source": "dms_context_adapter",
                        "method": "dms_context_adapter",
                        "mask_pattern": ":".join(map(str, pattern)),
                        "sample_idx": int(s),
                        "generated_rank": int(sample_rank),
                        "adapter_correction_mean": float(np.mean(corr)),
                    }
                )
                sample_rank += 1
                rows.append(item)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out = out.sort_values("context_logp", ascending=False).drop_duplicates(
        ["dataset_id", "mutation_notation"]
    )
    out["generated_rank"] = np.arange(len(out))
    return out.reset_index(drop=True)


def add_context_additive_scores(candidates: pd.DataFrame, context: MutationMapContext) -> pd.DataFrame:
    if len(candidates) == 0:
        return candidates.copy()
    out = candidates.copy()
    out["context_additive_score"] = out["mutation_notation"].astype(str).apply(context.additive_score)
    return out


def measured_pool_additive_oracle(
    dms_df: pd.DataFrame,
    task: EditingTask,
    context: MutationMapContext,
    k: int,
) -> pd.DataFrame:
    pool = candidate_pool_for_task(dms_df, task)
    if len(pool) == 0:
        return pool
    min_distance = min(2, int(task.edit_budget))
    pool = pool[pd.to_numeric(pool["mutation_distance"], errors="coerce") >= min_distance].copy()
    if len(pool) == 0:
        return pool
    out = add_context_additive_scores(pool, context)
    out = out.nlargest(min(k, len(out)), "context_additive_score").copy()
    out["method"] = "measured_pool_additive_oracle"
    return out.reset_index(drop=True)


def context_adapter_pool_rerank(
    *,
    dms_df: pd.DataFrame,
    task: EditingTask,
    wt_sequence: str,
    model,
    tokenizer,
    adapter: DMSContextAdapter,
    context: MutationMapContext,
    config: ContextGenerationConfig,
) -> pd.DataFrame:
    """Rank held-out measured candidates by frozen DPLM plus context-adapter log-prob.

    This is an offline ranking benchmark, not an open generator.  It answers a
    different reviewer-critical question: whether the trained DMS-context
    adapter assigns higher probability to functional measured multi-mutants
    than the frozen backbone alone.
    """
    import torch
    import torch.nn.functional as F

    pool = candidate_pool_for_task(dms_df, task)
    if len(pool) == 0:
        return pool
    min_distance = min(2, int(task.edit_budget))
    pool = pool[pd.to_numeric(pool["mutation_distance"], errors="coerce") >= min_distance].copy()
    if len(pool) == 0:
        return pool
    if config.max_pool_candidates > 0 and len(pool) > config.max_pool_candidates:
        pool = pool.sample(int(config.max_pool_candidates), random_state=int(config.seed)).reset_index(drop=True)
    allowed_token_ids, allowed_aas = _allowed_aa_token_ids(tokenizer)
    if tuple(allowed_aas) != tuple(AMINO_ACIDS):
        raise RuntimeError("DPLM tokenizer AA order does not match context adapter AA order")

    rows = []
    adapter.eval()
    for start in range(0, len(pool), int(config.batch_size)):
        chunk = pool.iloc[start : start + int(config.batch_size)].reset_index(drop=True)
        masked_sequences = []
        targets = []
        kept = []
        for _, row in chunk.iterrows():
            target = parse_variant_target(str(row["mutation_notation"]), wt_sequence)
            if target is None:
                continue
            masked_sequences.append(_build_masked_sequence(wt_sequence, target.positions))
            targets.append(target)
            kept.append(row)
        if not masked_sequences:
            continue
        logits, hidden, _ = _forward_logits_hidden_batch(model, tokenizer, masked_sequences)
        for b, (row, target) in enumerate(zip(kept, targets)):
            pos = torch.tensor(list(target.positions), dtype=torch.long, device=hidden.device)
            h = hidden[b].index_select(0, pos)
            ctx = torch.tensor(context.features_for_positions(target.positions), dtype=torch.float32, device=hidden.device)
            budget = torch.full((len(target.positions),), float(len(target.positions)), dtype=torch.float32, device=hidden.device)
            base = logits[b].index_select(0, pos)[:, allowed_token_ids].float()
            guided = base + float(config.context_scale) * adapter(h, ctx, budget)
            aa = torch.tensor(target.aa_ids, dtype=torch.long, device=hidden.device)
            base_lp = F.log_softmax(base, dim=-1).gather(1, aa[:, None]).squeeze(1)
            guided_lp = F.log_softmax(guided, dim=-1).gather(1, aa[:, None]).squeeze(1)
            out = row.to_dict()
            out["dplm_pool_logp"] = float(base_lp.mean().detach().cpu())
            out["context_adapter_pool_logp"] = float(guided_lp.mean().detach().cpu())
            rows.append(out)
    if not rows:
        return pool.iloc[:0].copy()
    out = pd.DataFrame(rows)
    out = out.nlargest(min(config.top_k, len(out)), "context_adapter_pool_logp").copy()
    out["method"] = "context_adapter_pool_rerank"
    return out.reset_index(drop=True)


def benchmark_context_generation_task(
    *,
    dms_df: pd.DataFrame,
    task: EditingTask,
    wt_sequence: str,
    model,
    tokenizer,
    adapter: DMSContextAdapter,
    context: MutationMapContext,
    prior: DMSFunctionPrior | None = None,
    config: ContextGenerationConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run same-task baselines and return ``(metrics, selections)``."""
    config = config or ContextGenerationConfig()
    metric_rows: list[dict] = []
    selection_rows: list[pd.DataFrame] = []

    def _record(method: str, selected: pd.DataFrame, generated: bool, elapsed: float):
        sel = selected.copy()
        sel["method"] = method
        sel["dataset_id"] = task.dataset_id
        sel["objective"] = task.objective
        sel["edit_budget"] = int(task.edit_budget)
        sel["seed"] = int(config.seed)
        selection_rows.append(sel)
        metrics = evaluate_generated_selection(sel, task) if generated else evaluate_edit_selection(sel, task)
        metrics.update(
            {
                "method": method,
                "dataset_id": task.dataset_id,
                "objective": task.objective,
                "edit_budget": int(task.edit_budget),
                "seed": int(config.seed),
                "runtime_sec": float(elapsed),
            }
        )
        metric_rows.append(metrics)

    # 1) Unguided DPLM candidate pool.
    t0 = perf_counter()
    dplm_cfg = DPLMConfig(
        n_mask_patterns=config.n_mask_patterns,
        samples_per_pattern=config.samples_per_pattern,
        temperature=config.temperature,
        seed=config.seed,
        batch_size=config.batch_size,
    )
    dplm_candidates = propose_with_dplm(wt_sequence, task, model, tokenizer, dplm_cfg)
    dplm_labeled = join_to_dms_labels(dplm_candidates, dms_df)
    _record(
        "dplm_unguided",
        dplm_labeled.head(config.top_k).copy(),
        generated=True,
        elapsed=perf_counter() - t0,
    )

    # 2) DPLM best-of-K by proposal log probability.
    t0 = perf_counter()
    best_k = dplm_labeled.nlargest(min(config.top_k, len(dplm_labeled)), "dplm_logp").copy()
    _record("dplm_best_of_k", best_k, generated=True, elapsed=perf_counter() - t0)

    # 3) DPLM + additive single-mutant context rerank.
    t0 = perf_counter()
    add = add_context_additive_scores(dplm_labeled, context)
    add = add.nlargest(min(config.top_k, len(add)), "context_additive_score").copy()
    _record("dplm_additive_context_rerank", add, generated=True, elapsed=perf_counter() - t0)

    # 4) DPLM + existing DMS prior rerank.
    if prior is not None and len(dplm_candidates) > 0:
        t0 = perf_counter()
        prior_ranked = rerank_with_classifier_guidance(
            dplm_candidates,
            prior=prior,
            beta=1.0,
            k=config.top_k,
            method_label="dplm_dms_prior_rerank",
        )
        prior_ranked = join_to_dms_labels(prior_ranked, dms_df)
        _record("dplm_dms_prior_rerank", prior_ranked, generated=True, elapsed=perf_counter() - t0)

    # 5) DMS-context adapter generation.
    t0 = perf_counter()
    ctx_generated = propose_with_context_adapter(
        wt_sequence,
        task,
        model,
        tokenizer,
        adapter,
        context,
        config=config,
    )
    ctx_generated = join_to_dms_labels(ctx_generated, dms_df)
    _record(
        "dms_context_adapter",
        ctx_generated.head(config.top_k).copy(),
        generated=True,
        elapsed=perf_counter() - t0,
    )

    # 6) Measured-pool additive oracle. This is not a deployable generator.
    t0 = perf_counter()
    oracle = measured_pool_additive_oracle(dms_df, task, context, k=config.top_k)
    _record("measured_pool_additive_oracle", oracle, generated=False, elapsed=perf_counter() - t0)

    # 7) Held-out measured-pool ranking by the trained adapter. This is not open
    # generation, but it is the cleanest likelihood/ranking test of the adapter.
    t0 = perf_counter()
    adapter_ranked = context_adapter_pool_rerank(
        dms_df=dms_df,
        task=task,
        wt_sequence=wt_sequence,
        model=model,
        tokenizer=tokenizer,
        adapter=adapter,
        context=context,
        config=config,
    )
    _record("context_adapter_pool_rerank", adapter_ranked, generated=False, elapsed=perf_counter() - t0)

    metrics = pd.DataFrame(metric_rows)
    selections = pd.concat(selection_rows, ignore_index=True) if selection_rows else pd.DataFrame()
    return metrics, selections
