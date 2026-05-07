"""DPLM (Diffusion Protein Language Model) masked-LM backbone.

Loads ``airkingbd/dplm_650m`` as a HuggingFace ``EsmForMaskedLM``: the DPLM
checkpoint shares ESM-2 architecture, just trained with absorbing-state
discrete diffusion. We use it as a one-shot masked LM for inference —
mask the editable positions of the WT, single forward pass, sample
amino acids from the per-position softmax. This is the standard DPLM
inference pattern when you don't need iterative denoising.

The same module exposes ``score_sequence_logprob`` for ranking generated
candidates by DPLM mean log-probability — the proposal score under the
classifier-guidance framework.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


# 20 standard amino acids (ESM-2 alphabet ordering uses the full vocab,
# but we restrict sampling to these — DPLM should not propose B/U/Z/X).
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def _load_dplm(model_name: str = "airkingbd/dplm_650m"):
    """Return ``(model, tokenizer)`` on the available device."""
    import torch
    from transformers import AutoTokenizer, EsmForMaskedLM

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name)
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer


@dataclass
class DPLMConfig:
    n_mask_patterns: int = 10
    samples_per_pattern: int = 20
    temperature: float = 1.0
    seed: int = 0
    batch_size: int = 8
    forbid_non_standard: bool = True


def _build_masked_sequence(wt_sequence: str, mask_positions: Iterable[int]) -> str:
    """Return WT with ``<mask>`` at 1-indexed positions in ``mask_positions``."""
    chars = list(wt_sequence)
    for pos in mask_positions:
        if 1 <= pos <= len(chars):
            chars[pos - 1] = "<mask>"
    return "".join(chars)


def _forward_logits_batch(
    model,
    tokenizer,
    masked_sequences: Sequence[str],
):
    """Return ``logits`` (B, L, V) and ``input_ids`` (B, L) for a batch of
    sequences with ``<mask>`` placeholders. ``L`` includes BOS/EOS — the
    caller is responsible for skipping those positions."""
    import torch

    inputs = tokenizer(list(masked_sequences), return_tensors="pt", padding=True)
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model(**inputs)
    return out.logits.detach(), inputs["input_ids"].detach()


def _allowed_aa_token_ids(tokenizer) -> tuple[list[int], list[str]]:
    """Token ids and corresponding AA characters that we permit DPLM to emit."""
    ids, aas = [], []
    for aa in AMINO_ACIDS:
        token_id = tokenizer.convert_tokens_to_ids(aa)
        if token_id is None or token_id == tokenizer.unk_token_id:
            continue
        ids.append(token_id)
        aas.append(aa)
    return ids, aas


def sample_from_masked_logits(
    logits,
    input_ids,
    mask_token_id: int,
    allowed_token_ids: list[int],
    allowed_aas: list[str],
    forbidden_aas: frozenset[str],
    rng: np.random.Generator,
    temperature: float = 1.0,
) -> list[dict]:
    """Decode each masked position by categorical sampling over ``allowed_token_ids``,
    excluding ``forbidden_aas``. Returns one dict per (sample, position) with the
    sampled AA and its log probability under the original full-vocab softmax.
    """
    import torch
    import torch.nn.functional as F

    out = []
    for b in range(logits.shape[0]):
        positions = (input_ids[b] == mask_token_id).nonzero(as_tuple=False).flatten().tolist()
        for pos_idx in positions:
            l = logits[b, pos_idx].float()
            full_logp = F.log_softmax(l / max(temperature, 1e-6), dim=-1).cpu().numpy()
            mask = np.array([(aa not in forbidden_aas) for aa in allowed_aas], dtype=bool)
            sub_ids = [t for t, m in zip(allowed_token_ids, mask) if m]
            sub_aas = [a for a, m in zip(allowed_aas, mask) if m]
            if not sub_ids:
                continue
            sub_logp = full_logp[sub_ids]
            sub_logp -= sub_logp.max()
            probs = np.exp(sub_logp)
            probs = probs / probs.sum()
            choice = int(rng.choice(len(sub_ids), p=probs))
            chosen_aa = sub_aas[choice]
            chosen_token_id = sub_ids[choice]
            chosen_logp = float(full_logp[chosen_token_id])
            out.append({
                "batch_idx": int(b),
                "token_position": int(pos_idx),
                "aa": chosen_aa,
                "logp": chosen_logp,
            })
    return out


def score_sequence_logprob(
    model,
    tokenizer,
    sequences: Sequence[str],
    batch_size: int = 8,
) -> np.ndarray:
    """Mean per-residue log-probability of ``sequences`` under DPLM with NO masking.

    Used to score candidate variants by DPLM "naturalness" — useful as a
    secondary diagnostic, not for the headline ranking.
    """
    import torch
    import torch.nn.functional as F

    out = np.zeros(len(sequences), dtype=np.float32)
    for start in range(0, len(sequences), batch_size):
        chunk = list(sequences[start : start + batch_size])
        inputs = tokenizer(chunk, return_tensors="pt", padding=True)
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits
        log_probs = F.log_softmax(logits, dim=-1)
        per_token = log_probs.gather(2, inputs["input_ids"].unsqueeze(-1)).squeeze(-1)
        attn = inputs["attention_mask"].bool()
        cls = tokenizer.cls_token_id
        eos = tokenizer.eos_token_id
        ids = inputs["input_ids"]
        keep = attn & (ids != cls) & (ids != eos)
        for j in range(logits.shape[0]):
            m = keep[j]
            if m.any():
                out[start + j] = float(per_token[j][m].mean().item())
            else:
                out[start + j] = float("nan")
    return out
