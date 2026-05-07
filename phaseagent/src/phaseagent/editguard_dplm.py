"""DMS-guided DPLM sampler — the headline EditGuard generative method.

Architecture:

1. **Propose** with DPLM (masked LM): for a task with edit_budget K and
   protected positions P, sample multiple mask patterns (random K-sized
   subsets of editable positions), run a single DPLM forward pass per
   pattern, and decode each mask position by stochastic categorical
   sampling over the standard 20-AA alphabet (with task forbidden
   residues excluded). Each generated variant is annotated with its
   DPLM mean log-probability ``dplm_logp``.

2. **Rerank** with the DMS function prior at classifier-guidance weight
   ``beta``:

        combined = dplm_logp + beta * log p_phi(viable | variant)

   ``beta = 0`` is pure DPLM sampling; larger ``beta`` is DMS-guided
   denoising. Output is the top-k variants by ``combined``, joined to
   DMS labels for evaluation.

The classifier-guidance interpretation is "guide-by-rerank-after-each-step"
applied at the end of generation rather than per-denoising-step. This is
the standard approximation for masked discrete diffusion when the guide
predictor takes mutation tokens (not soft sequences) as input. A future
revision can add per-step guidance once we have a differentiable
soft-sequence path through the DMS prior.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dplm_backbone import (
    AMINO_ACIDS,
    DPLMConfig,
    _allowed_aa_token_ids,
    _build_masked_sequence,
    _forward_logits_batch,
    sample_from_masked_logits,
)
from .editing_tasks import EditingTask
from .editguard_prior import DMSFunctionPrior


@dataclass
class EditGuardDPLMResult:
    """Container returned by the sampler — kept separate from the metric
    DataFrame so the modal entrypoint can pick which view to write."""

    candidates: pd.DataFrame  # one row per generated unique variant
    proposal_log: pd.DataFrame  # raw per-(mask-pattern, sample) log


def _sample_mask_pattern(
    wt_length: int,
    edit_budget: int,
    protected: frozenset[int],
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """Sample ``edit_budget`` positions from [1..wt_length] excluding ``protected``."""
    editable = [i for i in range(1, wt_length + 1) if i not in protected]
    if len(editable) < edit_budget:
        return tuple(editable)
    chosen = rng.choice(editable, size=edit_budget, replace=False)
    return tuple(sorted(int(p) for p in chosen))


def _decode_token_position_to_seq_position(token_position: int) -> int:
    """ESM tokenizer adds BOS at index 0, so seq[i] corresponds to token[i+1]."""
    return token_position  # tokens already include CLS at 0; we record token_position
                            # and the caller subtracts 1 when mapping to 1-indexed sequence.


def propose_with_dplm(
    wt_sequence: str,
    task: EditingTask,
    model,
    tokenizer,
    config: DPLMConfig | None = None,
) -> pd.DataFrame:
    """Generate a candidate pool from DPLM masked-LM forward passes.

    Returns a DataFrame with columns: ``dataset_id``, ``mutation_notation``,
    ``mutated_sequence``, ``mutation_distance``, ``dplm_logp``,
    ``proposal_source``, ``mask_pattern``, ``sample_idx``. Variants are
    deduplicated by ``mutation_notation``; if multiple proposals collapse
    to the same variant, the highest ``dplm_logp`` is kept.
    """
    config = config or DPLMConfig()
    rng = np.random.default_rng(config.seed)
    protected = frozenset(int(p) for p in task.protected_positions)
    forbidden = frozenset(a.upper() for a in task.forbidden_residues)
    allowed_token_ids, allowed_aas = _allowed_aa_token_ids(tokenizer)
    if not allowed_token_ids:
        raise RuntimeError("No allowed AA token ids resolved from tokenizer")
    mask_token_id = tokenizer.mask_token_id

    edit_budget = max(1, int(task.edit_budget))
    L = len(wt_sequence)
    rows: list[dict] = []

    # Generate mask patterns and a corresponding masked sequence for each.
    patterns: list[tuple[int, ...]] = []
    masked_seqs: list[str] = []
    for _ in range(config.n_mask_patterns):
        pattern = _sample_mask_pattern(L, edit_budget, protected, rng)
        if not pattern:
            continue
        patterns.append(pattern)
        masked_seqs.append(_build_masked_sequence(wt_sequence, pattern))

    if not patterns:
        return pd.DataFrame()

    # Forward pass in batches.
    for batch_start in range(0, len(masked_seqs), config.batch_size):
        chunk_patterns = patterns[batch_start : batch_start + config.batch_size]
        chunk_seqs = masked_seqs[batch_start : batch_start + config.batch_size]
        logits, input_ids = _forward_logits_batch(model, tokenizer, chunk_seqs)
        # For each pattern, draw samples_per_pattern independent variants.
        for s in range(config.samples_per_pattern):
            sampled = sample_from_masked_logits(
                logits,
                input_ids,
                mask_token_id=mask_token_id,
                allowed_token_ids=allowed_token_ids,
                allowed_aas=allowed_aas,
                forbidden_aas=forbidden,
                rng=rng,
                temperature=config.temperature,
            )
            # Group sampled tokens by batch_idx → reconstruct a variant per batch row.
            by_batch: dict[int, list[dict]] = {}
            for tok in sampled:
                by_batch.setdefault(tok["batch_idx"], []).append(tok)
            for b, toks in by_batch.items():
                pattern = chunk_patterns[b]
                # Map token positions (which include CLS/EOS) to 1-indexed sequence positions.
                # ESM tokenizers prepend CLS at token_position=0; sequence position = token_position
                # because we asked for residue chars only and tokenizer returns [CLS, r1, ..., rL, EOS].
                # The 1-indexed sequence position is therefore token_position (since r1 is at token 1).
                chars = list(wt_sequence)
                tokens_in_order = sorted(toks, key=lambda x: x["token_position"])
                pattern_set = set(pattern)
                token_to_seq_pos = sorted(pattern_set)  # token positions correspond to sequence positions in order
                seq_changes: list[tuple[int, str, str]] = []
                # tokens_in_order has len(pattern) entries, in token_position order.
                # Each maps to one of the pattern positions in increasing order.
                for tok, seq_pos in zip(tokens_in_order, token_to_seq_pos):
                    seq_idx = seq_pos - 1
                    if 0 <= seq_idx < L:
                        ref = wt_sequence[seq_idx].upper()
                        alt = tok["aa"].upper()
                        if ref != alt:
                            seq_changes.append((seq_pos, ref, alt))
                            chars[seq_idx] = alt
                if not seq_changes:
                    continue  # No change relative to WT; skip
                notation = ",".join(f"{ref}{pos}{alt}" for pos, ref, alt in seq_changes)
                mutated_seq = "".join(chars)
                logp_mean = float(np.mean([t["logp"] for t in toks]))
                rows.append({
                    "dataset_id": task.dataset_id,
                    "mutation_notation": notation,
                    "mutated_sequence": mutated_seq,
                    "mutation_distance": len(seq_changes),
                    "dplm_logp": logp_mean,
                    "proposal_source": "editguard_dplm",
                    "mask_pattern": ":".join(map(str, pattern)),
                    "sample_idx": s,
                })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Deduplicate by mutation_notation; keep the highest-dplm_logp instance.
    df = df.sort_values("dplm_logp", ascending=False).drop_duplicates(
        ["dataset_id", "mutation_notation"]
    ).reset_index(drop=True)
    return df


def rerank_with_classifier_guidance(
    candidates: pd.DataFrame,
    prior: DMSFunctionPrior,
    beta: float,
    k: int,
    method_label: str = "editguard_diffusion_dplm",
) -> pd.DataFrame:
    """Combine DPLM logp with DMS prior log-probability at weight ``beta``.

    Returns the top-k candidates by ``combined_score = dplm_logp + beta * log p_dms``.
    """
    if len(candidates) == 0 or k <= 0:
        return candidates.iloc[:0].copy()
    out = candidates.copy()
    p_dms = prior.predict_proba(out)
    out["prior_function_prob"] = p_dms
    out["log_prior_function_prob"] = np.log(np.clip(p_dms, 1e-6, 1.0 - 1e-6))
    out["combined_score"] = out["dplm_logp"].astype(float) + float(beta) * out["log_prior_function_prob"]
    out = out.nlargest(min(k, len(out)), "combined_score").copy()
    out["method"] = method_label
    out["beta"] = float(beta)
    return out.reset_index(drop=True)


def join_to_dms_labels(generated: pd.DataFrame, dms_df: pd.DataFrame) -> pd.DataFrame:
    """Attach measured DMS labels by ``(dataset_id, mutation_notation)`` if present."""
    cols = [c for c in ("dataset_id", "mutation_notation", "fitness_norm", "viable", "fitness_raw") if c in dms_df.columns]
    labels = dms_df[cols].drop_duplicates(["dataset_id", "mutation_notation"])
    out = generated.merge(labels, on=["dataset_id", "mutation_notation"], how="left", indicator="dms_label_status")
    out["has_dms_label"] = out["dms_label_status"].eq("both")
    return out.drop(columns=["dms_label_status"])
