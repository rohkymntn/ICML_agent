"""Trainable DMS-context adapter for protein diffusion logits.

The adapter keeps the DPLM backbone frozen.  For each masked edit position it
receives the DPLM hidden state and the target protein's single-mutant DMS map
features, then emits a 20-way amino-acid logit correction.  Training optimizes
masked-token likelihood on measured multi-mutants, with an optional pairwise
ranking loss at the variant level.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .mutation_map_context import CONTEXT_FEATURE_DIM, MutationMapContext
from .mutations import parse_mutation_notation
from .spectrum import AA_TO_ID

try:  # Imported lazily in environments without torch.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None
    _HAS_TORCH = False


@dataclass(frozen=True)
class DMSContextAdapterConfig:
    hidden_dim: int = 1280
    context_dim: int = CONTEXT_FEATURE_DIM
    adapter_dim: int = 256
    dropout: float = 0.10
    aa_vocab_size: int = 20
    edit_budget_scale: float = 10.0


@dataclass(frozen=True)
class VariantTarget:
    positions: tuple[int, ...]
    aa_ids: tuple[int, ...]
    mutation_notation: str


def parse_variant_target(mutation_notation: str, wt_sequence: str | None = None) -> VariantTarget | None:
    """Parse a multi-mutant notation into 1-indexed positions and AA ids."""
    toks = parse_mutation_notation(mutation_notation)
    if not toks:
        return None
    positions: list[int] = []
    aa_ids: list[int] = []
    seen_positions: set[int] = set()
    for tok in toks:
        try:
            pos = int(tok[1:-1])
        except ValueError:
            return None
        alt = tok[-1].upper()
        aa_id = AA_TO_ID.get(alt)
        if aa_id is None or pos in seen_positions:
            return None
        if wt_sequence is not None and not (1 <= pos <= len(wt_sequence)):
            return None
        positions.append(pos)
        aa_ids.append(int(aa_id))
        seen_positions.add(pos)
    order = np.argsort(positions)
    sorted_positions = tuple(int(positions[i]) for i in order)
    sorted_aa_ids = tuple(int(aa_ids[i]) for i in order)
    return VariantTarget(sorted_positions, sorted_aa_ids, ",".join(toks))


def make_training_frame(
    df: pd.DataFrame,
    *,
    min_distance: int = 2,
    max_distance: int = 8,
    fitness_col: str = "fitness_norm",
    require_viable: bool = False,
) -> pd.DataFrame:
    """Return parseable multi-mutant rows for adapter training/evaluation."""
    if "mutation_notation" not in df.columns:
        raise ValueError("training frame requires mutation_notation")
    if fitness_col not in df.columns:
        raise ValueError(f"training frame requires {fitness_col}")
    out = df.copy()
    if "mutation_distance" not in out.columns:
        out["mutation_distance"] = out["mutation_notation"].apply(lambda x: len(parse_mutation_notation(x)))
    dist = pd.to_numeric(out["mutation_distance"], errors="coerce")
    out = out[(dist >= min_distance) & (dist <= max_distance)].copy()
    out = out[out["mutation_notation"].apply(lambda x: parse_variant_target(str(x)) is not None)]
    out[fitness_col] = pd.to_numeric(out[fitness_col], errors="coerce")
    out = out.dropna(subset=[fitness_col])
    if require_viable and "viable" in out.columns:
        out = out[pd.to_numeric(out["viable"], errors="coerce") == 1]
    return out.reset_index(drop=True)


class DMSContextAdapter(nn.Module if _HAS_TORCH else object):
    """Small residual logit adapter conditioned on experimental mutation maps."""

    def __init__(self, config: DMSContextAdapterConfig):
        if not _HAS_TORCH:
            raise ImportError("DMSContextAdapter requires torch")
        super().__init__()
        self.config = config
        in_dim = int(config.hidden_dim + config.context_dim + 1)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, config.adapter_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.adapter_dim, config.adapter_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.adapter_dim, config.aa_vocab_size),
        )
        # Start near "do no harm" so early training behaves like frozen DPLM.
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, hidden_states, context_features, edit_budgets):
        """Return 20-way logit corrections for masked edit tokens.

        Parameters are token-level tensors:
        ``hidden_states`` [N, H], ``context_features`` [N, C],
        ``edit_budgets`` [N] or [N, 1].
        """
        if edit_budgets.ndim == 1:
            edit_budgets = edit_budgets[:, None]
        budget = edit_budgets.float() / max(float(self.config.edit_budget_scale), 1.0)
        x = torch.cat([hidden_states.float(), context_features.float(), budget], dim=-1)
        return self.net(x)


def token_context_for_variant(
    context: MutationMapContext,
    mutation_notation: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(positions, context_features)`` for one variant notation."""
    target = parse_variant_target(mutation_notation, context.wildtype_sequence)
    if target is None:
        return None
    positions = np.asarray(target.positions, dtype=np.int64)
    feats = context.features_for_positions(positions)
    return positions, feats


def adapter_loss(
    adapter: DMSContextAdapter,
    *,
    base_aa_logits,
    hidden_states,
    context_features,
    target_aa_ids,
    edit_budgets,
    token_weights=None,
    variant_offsets: Sequence[tuple[int, int]] | None = None,
    variant_fitness=None,
    pairwise_weight: float = 0.10,
    context_logit_scale: float = 1.0,
):
    """Compute weighted masked-token NLL plus optional variant ranking loss."""
    if not _HAS_TORCH:
        raise ImportError("adapter_loss requires torch")
    corrections = adapter(hidden_states, context_features, edit_budgets)
    guided_logits = base_aa_logits.float() + float(context_logit_scale) * corrections
    per_token = F.cross_entropy(guided_logits, target_aa_ids.long(), reduction="none")
    if token_weights is None:
        token_weights = torch.ones_like(per_token)
    token_weights = token_weights.float().clamp_min(0.0)
    nll = (per_token * token_weights).sum() / token_weights.sum().clamp_min(1.0)
    loss = nll
    rank_loss = torch.tensor(0.0, device=guided_logits.device)
    if (
        pairwise_weight > 0
        and variant_offsets is not None
        and variant_fitness is not None
        and len(variant_offsets) >= 2
    ):
        logp = F.log_softmax(guided_logits, dim=-1)
        target_lp = logp.gather(1, target_aa_ids.long()[:, None]).squeeze(1)
        scores = []
        for start, end in variant_offsets:
            if end <= start:
                scores.append(torch.tensor(0.0, device=guided_logits.device))
            else:
                scores.append(target_lp[start:end].mean())
        row_scores = torch.stack(scores)
        fit = torch.as_tensor(variant_fitness, dtype=torch.float32, device=guided_logits.device)
        pairs_i, pairs_j = torch.nonzero(fit[:, None] > fit[None, :] + 1e-6, as_tuple=True)
        if len(pairs_i) > 0:
            # Cap pairs to keep large batches cheap and deterministic.
            max_pairs = min(int(len(pairs_i)), 2048)
            pairs_i = pairs_i[:max_pairs]
            pairs_j = pairs_j[:max_pairs]
            rank_loss = F.softplus(-(row_scores[pairs_i] - row_scores[pairs_j])).mean()
            loss = loss + float(pairwise_weight) * rank_loss
    return {
        "loss": loss,
        "nll_loss": nll.detach(),
        "rank_loss": rank_loss.detach(),
        "guided_logits": guided_logits,
        "corrections": corrections,
    }


def save_adapter(
    adapter: DMSContextAdapter,
    path: str | Path,
    *,
    metadata: dict | None = None,
) -> None:
    if not _HAS_TORCH:
        raise ImportError("save_adapter requires torch")
    payload = {
        "state_dict": adapter.state_dict(),
        "config": asdict(adapter.config),
        "metadata": metadata or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_adapter(path: str | Path, map_location: str | None = None) -> DMSContextAdapter:
    if not _HAS_TORCH:
        raise ImportError("load_adapter requires torch")
    try:
        payload = torch.load(path, map_location=map_location or "cpu", weights_only=True)
    except TypeError:  # Older torch releases do not expose weights_only.
        payload = torch.load(path, map_location=map_location or "cpu")
    cfg = DMSContextAdapterConfig(**payload["config"])
    adapter = DMSContextAdapter(cfg)
    adapter.load_state_dict(payload["state_dict"])
    return adapter
