"""Real DPLM-650M wiring for the iterative twisted SMC sampler.

This module bridges the abstract ``DPLMBackbone`` / ``RewardModel`` protocols
in ``twisted_smc.py`` to the production DPLM-650M HuggingFace checkpoint
(``airkingbd/dplm_650m``) and the trained ``DMSFunctionPrior``.

Imports torch / transformers lazily so the SMC unit tests stay importable on
CPU-only laptops. Production use lives inside the GPU Modal entrypoint
(``modal_app_v2.py::run_twisted_smc_dplm``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .dplm_backbone import (
    AMINO_ACIDS,
    _allowed_aa_token_ids,
    _build_masked_sequence,
    _forward_logits_batch,
    _load_dplm,
)
from .editguard_prior import DMSFunctionPrior
from .phase_validation import (
    SpectrumStats,
    compute_spectrum_stats,
    predict_dc_alpha_from_spectrum,
    predict_V_at_distance,
    predict_V_at_distance_mc,
)
from .twisted_smc import DenoiseStep, DPLMBackbone, RewardModel


@dataclass
class RealDPLMBackbone(DPLMBackbone):
    """Production DPLM-650M backbone for Twisted SMC.

    Initialization randomly masks ``edit_budget`` positions of the WT.
    Each ``step()`` runs one masked-LM forward pass on the current sequence
    and unmasks one randomly-chosen masked position by sampling from the
    DPLM softmax over the 20 standard amino acids (forbidden residues
    optionally excluded).
    """

    model_name: str = "airkingbd/dplm_650m"
    temperature: float = 1.0
    forbidden_aas: frozenset[str] = frozenset()
    _model: object = None
    _tokenizer: object = None
    _allowed_ids: list = None
    _allowed_aas: list = None
    _mask_id: int = -1

    def _ensure_loaded(self):
        if self._model is None:
            model, tokenizer = _load_dplm(self.model_name)
            self._model = model
            self._tokenizer = tokenizer
            ids, aas = _allowed_aa_token_ids(tokenizer)
            self._allowed_ids = ids
            self._allowed_aas = aas
            self._mask_id = int(tokenizer.mask_token_id)

    def initialize(self, wt_seq: str, edit_budget: int, rng: np.random.Generator) -> tuple[str, np.ndarray]:
        positions = rng.choice(len(wt_seq), size=int(edit_budget), replace=False)
        mask = np.zeros(len(wt_seq), dtype=bool)
        mask[positions] = True
        return wt_seq, mask

    def step(
        self,
        seq: str,
        mask: np.ndarray,
        t: int,
        rng: np.random.Generator,
    ) -> DenoiseStep:
        if not mask.any():
            return DenoiseStep(new_sequence=seq, new_mask=mask, log_p_step=0.0)
        self._ensure_loaded()
        import torch
        import torch.nn.functional as F

        # Build the masked sequence using *all* currently masked positions so
        # DPLM sees the full context.
        masked_positions = [int(p) + 1 for p in np.flatnonzero(mask)]  # 1-indexed for builder
        masked_seq = _build_masked_sequence(seq, masked_positions)
        logits, input_ids = _forward_logits_batch(self._model, self._tokenizer, [masked_seq])
        # Pick one mask token at random (single-position unmask per step keeps the
        # outer SMC schedule comparable to the smc_ddm formulation).
        positions_in_tokens = (input_ids[0] == self._mask_id).nonzero(as_tuple=False).flatten().tolist()
        if not positions_in_tokens:
            return DenoiseStep(new_sequence=seq, new_mask=mask, log_p_step=0.0)
        pick_token_pos = int(rng.choice(positions_in_tokens))
        # Map token position → 0-indexed sequence position (subtract 1 for BOS).
        seq_pos = pick_token_pos - 1
        l = logits[0, pick_token_pos].float()
        full_logp = F.log_softmax(l / max(self.temperature, 1e-6), dim=-1).cpu().numpy()
        sub_mask = np.array([(aa not in self.forbidden_aas) for aa in self._allowed_aas], dtype=bool)
        sub_ids = [t for t, m in zip(self._allowed_ids, sub_mask) if m]
        sub_aas = [a for a, m in zip(self._allowed_aas, sub_mask) if m]
        if not sub_ids:
            return DenoiseStep(new_sequence=seq, new_mask=mask, log_p_step=0.0)
        sub_logp = full_logp[sub_ids]
        sub_logp = sub_logp - sub_logp.max()
        probs = np.exp(sub_logp)
        probs = probs / probs.sum()
        choice = int(rng.choice(len(sub_ids), p=probs))
        chosen_aa = sub_aas[choice]
        chosen_logp = float(full_logp[sub_ids[choice]])
        new_seq = seq[:seq_pos] + chosen_aa + seq[seq_pos + 1 :]
        new_mask = mask.copy()
        new_mask[seq_pos] = False
        return DenoiseStep(new_sequence=new_seq, new_mask=new_mask, log_p_step=chosen_logp)


@dataclass
class DMSPriorRewardModel(RewardModel):
    """Wrap a trained ``DMSFunctionPrior`` as a ``RewardModel``.

    Builds a one-row dataframe per sequence and reads ``predict_proba``,
    returning log P(viable) as the reward (higher is better).
    """

    prior: DMSFunctionPrior
    dataset_id: str
    wt_sequence: str
    add_pseudocount: float = 1e-3

    def reward(self, sequences: list[str], context: dict | None = None) -> np.ndarray:
        rows = []
        for s in sequences:
            mutations = []
            for i, (wt_aa, mut_aa) in enumerate(zip(self.wt_sequence, s), start=1):
                if wt_aa != mut_aa:
                    mutations.append(f"{wt_aa}{i}{mut_aa}")
            notation = ":".join(mutations) if mutations else "WT"
            rows.append(
                {
                    "dataset_id": self.dataset_id,
                    "mutation_notation": notation,
                    "mutation_distance": len(mutations),
                    "wildtype_sequence": self.wt_sequence,
                    "mutated_sequence": s,
                    "fitness_norm": 0.5,  # placeholder; not used by the prior
                    "viable": 0,
                }
            )
        df = pd.DataFrame(rows)
        proba = self.prior.predict_proba(df)
        return np.log(np.clip(proba, self.add_pseudocount, 1.0 - self.add_pseudocount))


def _hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(1 for x, y in zip(a, b) if x != y)


@dataclass
class PhaseCalibratedRewardModel(RewardModel):
    """Phase-calibrated reward: log phi(x) + beta * log V_g(d(x, x_0)).

    The phase prior V_g is the LDP-derived survival function from
    Theorem 1 of the paper -- it depends only on the per-protein
    single-mutant effect spectrum (m_g, sigma_g) and the viability gap
    delta_g, all of which are computed from the *single-mutant* DMS
    measurements alone.

    Parameters
    ----------
    base_reward
        The base reward model (e.g. DMSPriorRewardModel) producing
        log phi(x). The phase term is added on top.
    wt_sequence
        Wild-type sequence; used to compute Hamming distance.
    spectrum_stats
        First two moments (m_g, sigma_g) and third moment of the
        single-mutant damage distribution; produced by
        ``phase_validation.compute_spectrum_stats``.
    delta_g
        Viability gap WT_fitness - threshold > 0 (raw fitness units).
    beta
        Phase-prior weight. Set to 0 for ablation = base reward only.
    use_mc
        If True, use Monte-Carlo over the empirical effect distribution
        (more accurate at small d) rather than the Gaussian
        Berry-Esseen approximation. Requires ``effects`` to be set.
    effects
        Optional empirical single-mutant damage values. Required if
        ``use_mc=True``. If provided, V_g(d) is computed by MC; else by
        Berry-Esseen Gaussian.
    floor
        Numeric floor on V_g(d) to prevent log(0).
    """

    base_reward: RewardModel
    wt_sequence: str
    spectrum_stats: SpectrumStats
    delta_g: float
    beta: float = 1.0
    use_mc: bool = True
    effects: np.ndarray | None = None
    floor: float = 1e-6
    n_mc_samples: int = 50_000

    def reward(self, sequences: list[str], context: dict | None = None) -> np.ndarray:
        base = self.base_reward.reward(sequences, context=context)
        base = np.asarray(base, dtype=float)
        if self.beta == 0.0:
            return base
        phase = np.zeros(len(sequences), dtype=float)
        for i, s in enumerate(sequences):
            d = _hamming(s, self.wt_sequence)
            if d <= 0:
                # WT itself; assign V=1.0 (full viability).
                Vd = 1.0
            elif self.use_mc and self.effects is not None and len(self.effects) > 0:
                Vd = predict_V_at_distance_mc(
                    self.effects, d=int(d), delta_g=self.delta_g,
                    n_samples=self.n_mc_samples, seed=int(d),
                )
            else:
                Vd = predict_V_at_distance(
                    self.spectrum_stats, d=int(d), delta_g=self.delta_g,
                )
            Vd = float(np.clip(Vd, self.floor, 1.0))
            phase[i] = np.log(Vd)
        return base + self.beta * phase


def build_phase_reward_from_singles(
    base_reward: RewardModel,
    wt_sequence: str,
    single_effects: np.ndarray,
    delta_g: float,
    beta: float = 1.0,
    use_mc: bool = True,
) -> "PhaseCalibratedRewardModel":
    """Construct a PhaseCalibratedRewardModel from raw single-mutant damage values."""
    stats = compute_spectrum_stats(single_effects)
    return PhaseCalibratedRewardModel(
        base_reward=base_reward,
        wt_sequence=wt_sequence,
        spectrum_stats=stats,
        delta_g=float(delta_g),
        beta=float(beta),
        use_mc=bool(use_mc),
        effects=np.asarray(single_effects, dtype=float) if use_mc else None,
    )
