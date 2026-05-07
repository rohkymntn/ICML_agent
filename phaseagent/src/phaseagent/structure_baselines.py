"""Structure-conditioned variant scoring with ESM-IF1 inverse folding.

Loads an AlphaFold PDB for the target protein, extracts the WT backbone
coordinates once, and scores variants by mean per-residue log-likelihood
under ``esm.pretrained.esm_if1_gvp4_t16_142M_UR50``: ``log P(seq | WT
backbone)``.

The structure conditioning is fixed to the WT backbone (we never have the
variant's true structure). This is the standard inverse-folding scoring
setup used in the ProteinGym structure-conditioned baselines.

For multi-mutant variants, we report the **average per-residue
log-likelihood over the variant's mutated positions only** (`per_position`
mode). The full-sequence average is also returned for comparison
(`full_sequence` mode), since published ESM-IF leaderboard numbers use
that variant.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .mutations import parse_mutation_notation


def _load_esm_if(model_name: str = "esm_if1_gvp4_t16_142M_UR50"):
    """Load ESM-IF1 inverse folding model. Returns (model, alphabet)."""
    import esm

    fn = getattr(esm.pretrained, model_name)
    model, alphabet = fn()
    model = model.eval()
    import torch

    if torch.cuda.is_available():
        model = model.cuda()
    return model, alphabet


def load_backbone_coords(pdb_path: str | Path, chain: str = "A"):
    """Return (coords, native_seq) from a PDB file via the ESM-IF utilities."""
    from esm.inverse_folding import util

    structure = util.load_structure(str(pdb_path), chain)
    coords, seq = util.extract_coords_from_structure(structure)
    return coords, seq


def per_residue_log_probs(model, alphabet, coords, sequence: str) -> np.ndarray:
    """Per-residue log-likelihood of the given sequence under the WT backbone.

    Returns a 1-D ``np.float32`` array of length ``L`` (one entry per residue
    in ``sequence``); positions with NaN backbone coordinates are masked out
    with ``np.nan``.
    """
    import torch
    import torch.nn.functional as F

    from esm.inverse_folding import util

    coords_b, confidence_b, strs_b, tokens_b, padding_mask_b = util.CoordBatchConverter(
        alphabet
    )([(coords, None, sequence)], device=next(model.parameters()).device)
    prev_output_tokens = tokens_b[:, :-1]
    target = tokens_b[:, 1:]
    target_padding_mask = target == alphabet.padding_idx
    with torch.inference_mode():
        logits, _ = model.forward(coords_b, padding_mask_b, confidence_b, prev_output_tokens)
        log_probs = F.log_softmax(logits, dim=1)  # (B, V, L)
        # Gather log_prob of the actual target token per position.
        per_pos = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)  # (B, L)
    per_pos = per_pos[0].detach().cpu().numpy().astype(np.float32)
    target_padding_mask = target_padding_mask[0].detach().cpu().numpy()
    coord_finite = np.all(np.all(np.isfinite(coords), axis=-1), axis=-1)
    # `coords` is (L, 3, 3) for N/CA/C; per_pos is L because of the shift.
    valid = (~target_padding_mask) & coord_finite[: len(per_pos)]
    out = np.full(len(per_pos), np.nan, dtype=np.float32)
    out[valid] = per_pos[valid]
    return out


def score_variant_per_position(
    per_pos_log_probs_wt: np.ndarray,  # not used today, kept for API symmetry
    per_pos_log_probs_var: np.ndarray,
    mutation_positions: list[int],
) -> tuple[float, float]:
    """Return (mean_log_prob_at_mutated_positions, mean_log_prob_full).

    Positions are 1-indexed (ProteinGym convention).
    """
    full = np.nanmean(per_pos_log_probs_var)
    if not mutation_positions:
        return float("nan"), float(full)
    idx = [p - 1 for p in mutation_positions if 0 <= p - 1 < len(per_pos_log_probs_var)]
    if not idx:
        return float("nan"), float(full)
    pos_vals = per_pos_log_probs_var[idx]
    pos_vals = pos_vals[np.isfinite(pos_vals)]
    if pos_vals.size == 0:
        return float("nan"), float(full)
    return float(np.mean(pos_vals)), float(full)


def score_variants_with_esm_if(
    candidates: pd.DataFrame,
    pdb_path: str | Path,
    model=None,
    alphabet=None,
    mutated_seq_col: str = "mutated_sequence",
    notation_col: str = "mutation_notation",
    chain: str = "A",
) -> pd.DataFrame:
    """Annotate each candidate with ESM-IF1 scores.

    Adds two columns:

    - ``esm_if_score`` — mean per-residue log-prob over the variant's mutated
      positions only (recommended; tracks "is this *substitution* plausible
      given the structure").
    - ``esm_if_score_full`` — mean per-residue log-prob over the full mutated
      sequence (matches the ProteinGym ESM-IF leaderboard convention).

    The model + alphabet are loaded if not provided. Coordinates are
    extracted once from the PDB.
    """
    if model is None or alphabet is None:
        model, alphabet = _load_esm_if()
    coords, native_seq = load_backbone_coords(pdb_path, chain=chain)
    out = candidates.copy().reset_index(drop=True)
    if len(out) == 0:
        out["esm_if_score"] = np.nan
        out["esm_if_score_full"] = np.nan
        return out
    if mutated_seq_col not in out.columns:
        raise KeyError(f"candidate dataframe must include {mutated_seq_col!r}")
    pos_scores: list[float] = []
    full_scores: list[float] = []
    cache: dict[str, np.ndarray] = {}
    n = len(out)
    for i, row in out.iterrows():
        seq = str(row[mutated_seq_col])
        if not seq or seq.lower() in {"nan", "none"}:
            pos_scores.append(float("nan"))
            full_scores.append(float("nan"))
            continue
        if len(seq) != len(native_seq):
            # Length mismatch — refuse to score (reviewers will ask).
            pos_scores.append(float("nan"))
            full_scores.append(float("nan"))
            continue
        if seq not in cache:
            cache[seq] = per_residue_log_probs(model, alphabet, coords, seq)
        per_pos = cache[seq]
        toks = parse_mutation_notation(str(row.get(notation_col, "")))
        positions = []
        for tok in toks:
            try:
                positions.append(int(tok[1:-1]))
            except ValueError:
                continue
        s_pos, s_full = score_variant_per_position(None, per_pos, positions)
        pos_scores.append(s_pos)
        full_scores.append(s_full)
        if (i + 1) % 100 == 0:
            print(f"[esm-if] scored {i + 1}/{n}")
    out["esm_if_score"] = pos_scores
    out["esm_if_score_full"] = full_scores
    return out


def rerank_by_esm_if(
    candidates: pd.DataFrame,
    pdb_path: str | Path,
    k: int,
    model=None,
    alphabet=None,
    score_col: str = "esm_if_score",
    method_label: str = "esm_if_rerank",
    chain: str = "A",
) -> pd.DataFrame:
    """Score candidates with ESM-IF1 and return the top-k by ``score_col``."""
    scored = score_variants_with_esm_if(
        candidates,
        pdb_path,
        model=model,
        alphabet=alphabet,
        chain=chain,
    )
    scored = scored.dropna(subset=[score_col])
    if len(scored) == 0:
        return scored.assign(method=method_label)
    top = scored.nlargest(min(k, len(scored)), score_col)
    out = top.copy()
    out["method"] = method_label
    return out.reset_index(drop=True)
