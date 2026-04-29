"""ESM2 pseudo-likelihood scoring on GPU.

This module is only imported inside the GPU Modal image — torch/fair-esm
are not part of the CPU dependency set.
"""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from .mutations import apply_substitutions


def _load_esm(model_name: str):
    import esm
    import torch

    fn = getattr(esm.pretrained, model_name)
    model, alphabet = fn()
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, alphabet


def pseudo_log_likelihood(
    seqs: list[str],
    model,
    alphabet,
    batch_size: int = 4,
    max_len: int = 1022,
) -> np.ndarray:
    """Mean per-residue log P(token | full-context) — fast naturalness proxy.

    Truncates to `max_len` residues (ESM-2 hard limit is 1024 incl. CLS/EOS).
    """
    import torch

    bc = alphabet.get_batch_converter()
    out = np.zeros(len(seqs), dtype=np.float32)
    device = next(model.parameters()).device
    with torch.inference_mode():
        for start in range(0, len(seqs), batch_size):
            chunk = seqs[start:start + batch_size]
            chunk = [s[:max_len] for s in chunk]
            batch = [(f"s{i}", s) for i, s in enumerate(chunk)]
            _, _, toks = bc(batch)
            toks = toks.to(device)
            logits = model(toks)["logits"]
            log_probs = torch.log_softmax(logits, dim=-1)
            true_lp = log_probs.gather(2, toks.unsqueeze(-1)).squeeze(-1)
            mask = (
                (toks != alphabet.padding_idx)
                & (toks != alphabet.cls_idx)
                & (toks != alphabet.eos_idx)
            )
            for j in range(true_lp.shape[0]):
                m = mask[j]
                if m.any():
                    out[start + j] = float(true_lp[j][m].mean().detach().cpu())
                else:
                    out[start + j] = float("nan")
    return out


def masked_token_log_probs(
    wt_sequence: str,
    mutation_tokens: list[str],
    model,
    alphabet,
    max_len: int = 1022,
    batch_size: int = 8,
) -> dict[str, float]:
    """Score substitution tokens by ESM masked marginal log probability.

    Each token is scored by masking the target position in the WT sequence and
    reading log P(mutant amino acid | masked WT context).
    """
    import torch

    if len(wt_sequence) > max_len:
        wt_sequence = wt_sequence[:max_len]
    bc = alphabet.get_batch_converter()
    device = next(model.parameters()).device
    token_to_idx = alphabet.tok_to_idx
    rows, valid_tokens = [], []
    for tok in mutation_tokens:
        try:
            pos = int(tok[1:-1])
            alt = tok[-1].upper()
        except Exception:
            continue
        if not (1 <= pos <= len(wt_sequence)) or alt not in token_to_idx:
            continue
        chars = list(wt_sequence)
        chars[pos - 1] = "<mask>"
        rows.append((tok, "".join(chars)))
        valid_tokens.append(tok)
    if not rows:
        return {}
    out: dict[str, float] = {}
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            chunk_tokens = valid_tokens[start:start + batch_size]
            _, _, toks = bc(chunk)
            toks = toks.to(device)
            logits = model(toks)["logits"]
            log_probs = torch.log_softmax(logits, dim=-1)
            for i, tok in enumerate(chunk_tokens):
                pos = int(tok[1:-1])
                alt = tok[-1].upper()
                out[tok] = float(log_probs[i, pos, token_to_idx[alt]].detach().cpu())
            del toks, logits, log_probs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return out


def score_generated_esm_masked(
    generated: pd.DataFrame,
    model_name: str = "esm2_t33_650M_UR50D",
    batch_size: int = 8,
) -> pd.DataFrame:
    """Annotate generated mutation candidates with ESM masked marginal scores."""
    from .mutations import parse_mutation_notation

    model, alphabet = _load_esm(model_name)
    rows = []
    for ds_id, sub in generated.groupby("dataset_id"):
        if "wildtype_sequence" not in sub.columns:
            continue
        wt_vals = sub["wildtype_sequence"].dropna().astype(str)
        if len(wt_vals) == 0:
            continue
        wt = wt_vals.iloc[0]
        all_tokens = sorted({tok for m in sub["mutation_notation"].astype(str) for tok in parse_mutation_notation(m)})
        token_scores = masked_token_log_probs(wt, all_tokens, model, alphabet, batch_size=batch_size)
        scored = sub.copy()
        vals = []
        for notation in scored["mutation_notation"].astype(str):
            toks = parse_mutation_notation(notation)
            if not toks or any(tok not in token_scores for tok in toks):
                vals.append(float("nan"))
            else:
                vals.append(float(np.mean([token_scores[tok] for tok in toks])))
        scored["esm_masked_score"] = vals
        rows.append(scored)
    return pd.concat(rows, ignore_index=True) if rows else generated.assign(esm_masked_score=np.nan)


def _resolve_sequences(sub: pd.DataFrame) -> list[str] | None:
    has_seq = (
        "mutated_sequence" in sub.columns
        and sub["mutated_sequence"].notna().any()
        and sub["mutated_sequence"].astype(str).str.len().gt(0).any()
    )
    if has_seq:
        return sub["mutated_sequence"].astype(str).tolist()
    if "wildtype_sequence" not in sub.columns:
        return None
    wt_vals = sub["wildtype_sequence"].dropna().unique()
    if len(wt_vals) == 0:
        return None
    wt = wt_vals[0]
    if "mutation_notation" not in sub.columns:
        return None
    return [apply_substitutions(wt, m) for m in sub["mutation_notation"].astype(str)]


def score_dataset_plm(
    df: pd.DataFrame,
    model_name: str = "esm2_t33_650M_UR50D",
    max_per_dataset: int = 5000,
    batch_size: int = 4,
    random_state: int = 0,
) -> pd.DataFrame:
    rows = []
    model = alphabet = None
    for ds_id, sub in df.groupby("dataset_id"):
        if len(sub) > max_per_dataset:
            sub = sub.sample(max_per_dataset, random_state=random_state).reset_index(drop=True)
        else:
            sub = sub.reset_index(drop=True)
        seqs = _resolve_sequences(sub)
        if seqs is None:
            print(f"[plm] skipping {ds_id}: no sequences available")
            continue
        if model is None:
            model, alphabet = _load_esm(model_name)
        scores = pseudo_log_likelihood(seqs, model, alphabet, batch_size=batch_size)
        sub_out = sub[["dataset_id", "variant_id", "mutation_distance", "fitness_norm", "viable"]].copy()
        sub_out["plm_score"] = scores
        rows.append(sub_out)
        print(f"[plm] {ds_id}: scored {len(sub_out)} variants")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
