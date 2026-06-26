"""Iterative Twisted SMC sampler on top of DPLM-650M.

This is the stretch *method contribution* identified by the SMC research scout:
DPLM-650M is unclaimed in the SMC literature, and combining a twisted-SMC
inner loop (smc_ddm-style) with an evolutionary outer noise-then-denoise loop
(Reward-Guided Iterative Refinement, Uehara et al. ICML 2025) gives a clean
algorithmic delta.

Algorithm (high level):
1. Initialize K particles by sampling K noised wildtype-like sequences.
2. Run T denoising steps. At each step:
   a. Predict logits with DPLM at the current noise level.
   b. Sample a candidate transition for each particle via the standard
      DPLM mask-update.
   c. Compute the *twist* — a first-order Taylor approximation to the
      classifier reward for non-differentiable RF-DMS prior, à la Ou/Pani/Li
      2025 (smc_ddm).
   d. Importance-weight each particle by exp(beta * twist).
   e. Resample with replacement using systematic resampling.
3. After T steps, optionally re-noise back to t = T_outer/2 and repeat
   (Uehara RGIR outer loop).
4. Return the top-K particles by RF-DMS reward.

Compute parity: K=8 particles × T=128 steps × N_outer=2 outer iterations =
2048 forward passes, comparable to best-of-256 vanilla DPLM rerank.

This module is a clean *interface* + scaffolding. The DPLM forward pass and
RF-DMS reward computation are pluggable so the test suite can exercise the
control flow with a tiny mock backbone, and the production GPU run wires
the real DPLM and the trained ``DMSFunctionPrior`` in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import pandas as pd


# A particle here is a (sequence, partial-mask) pair plus a current weight.
@dataclass
class Particle:
    sequence: str
    mask: np.ndarray  # boolean per-position mask of currently-masked positions
    weight: float = 1.0
    score_log: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class TwistedSMCConfig:
    """Hyperparameters for one twisted-SMC run."""

    n_particles: int = 8
    n_denoise_steps: int = 128
    n_outer_iterations: int = 2
    outer_renoise_fraction: float = 0.5
    beta: float = 1.0  # twist temperature (β=0 reproduces unconditioned sampling)
    resample_threshold_ess: float = 0.5  # resample when effective sample size drops below ESS_th * K
    seed: int = 0


@dataclass
class DenoiseStep:
    """One denoising step result for a single particle."""

    new_sequence: str
    new_mask: np.ndarray
    log_p_step: float  # log P_DPLM(transition) — used for proposal correction


# Backbone protocols. Production wires DPLM-650M; tests wire a mock.
class DPLMBackbone:
    """Protocol for the DPLM masked-diffusion backbone."""

    def initialize(self, wt_seq: str, edit_budget: int, rng: np.random.Generator) -> tuple[str, np.ndarray]:
        """Return (initial_sequence, initial_mask) for one particle."""
        raise NotImplementedError

    def step(
        self,
        seq: str,
        mask: np.ndarray,
        t: int,
        rng: np.random.Generator,
    ) -> DenoiseStep:
        """One mask-update step. Returns the new sequence/mask + step log-prob."""
        raise NotImplementedError


class RewardModel:
    """Protocol for the classifier reward (RF-DMS prior in production)."""

    def reward(self, sequences: list[str], context: dict | None = None) -> np.ndarray:
        """Return per-sequence log P(viable) (or any non-positive log-reward)."""
        raise NotImplementedError


class MockDPLMBackbone(DPLMBackbone):
    """Stand-in DPLM backbone for unit tests.

    Initializes by random-masking ``edit_budget`` positions of the WT, and
    each step uniform-randomly fills one masked position with a random AA.
    """

    def __init__(self, alphabet: str = "ACDEFGHIKLMNPQRSTVWY"):
        self.alphabet = alphabet

    def initialize(self, wt_seq: str, edit_budget: int, rng: np.random.Generator) -> tuple[str, np.ndarray]:
        positions = rng.choice(len(wt_seq), size=int(edit_budget), replace=False)
        mask = np.zeros(len(wt_seq), dtype=bool)
        mask[positions] = True
        return wt_seq, mask

    def step(self, seq: str, mask: np.ndarray, t: int, rng: np.random.Generator) -> DenoiseStep:
        if not mask.any():
            return DenoiseStep(new_sequence=seq, new_mask=mask, log_p_step=0.0)
        masked_positions = np.flatnonzero(mask)
        pos = int(rng.choice(masked_positions))
        new_aa = self.alphabet[int(rng.integers(0, len(self.alphabet)))]
        new_mask = mask.copy()
        new_mask[pos] = False
        new_seq = seq[:pos] + new_aa + seq[pos + 1 :]
        log_p = -float(np.log(len(self.alphabet)))
        return DenoiseStep(new_sequence=new_seq, new_mask=new_mask, log_p_step=log_p)


class MockRewardModel(RewardModel):
    """Reward = -mean(non-A character index). Higher reward -> more 'A'-rich."""

    def reward(self, sequences: list[str], context: dict | None = None) -> np.ndarray:
        out = np.zeros(len(sequences), dtype=float)
        for i, seq in enumerate(sequences):
            score = sum(0.0 if c == "A" else -1.0 for c in seq) / max(1, len(seq))
            out[i] = score
        return out


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Standard systematic resampling. Returns indices of resampled particles."""
    K = len(weights)
    if K == 0:
        return np.array([], dtype=int)
    w = weights / max(weights.sum(), 1e-12)
    cum = np.cumsum(w)
    u = (rng.random() + np.arange(K)) / K
    idx = np.searchsorted(cum, u)
    return np.clip(idx, 0, K - 1)


def _effective_sample_size(weights: np.ndarray) -> float:
    if weights.sum() <= 0:
        return 0.0
    p = weights / weights.sum()
    return float(1.0 / np.sum(p ** 2))


def run_twisted_smc(
    wt_seq: str,
    edit_budget: int,
    backbone: DPLMBackbone,
    reward: RewardModel,
    config: TwistedSMCConfig,
    *,
    return_trajectory: bool = False,
) -> dict:
    """Run iterative twisted SMC and return the final particle set.

    Returns a dict with keys:
    - ``particles``: list[Particle] of length K (after final resample)
    - ``rewards``: np.ndarray of final per-particle rewards
    - ``trace``: optional per-step ESS / mean-reward trajectory if
      ``return_trajectory=True``.
    """
    rng = np.random.default_rng(config.seed)
    particles: list[Particle] = []
    for _ in range(config.n_particles):
        seq, mask = backbone.initialize(wt_seq, edit_budget, rng)
        particles.append(Particle(sequence=seq, mask=mask, weight=1.0))

    trace: list[dict] = []

    def _denoise_pass(particles: list[Particle], n_steps: int) -> list[Particle]:
        for t in range(n_steps):
            log_w = np.zeros(len(particles), dtype=float)
            new_seqs: list[str] = []
            for i, p in enumerate(particles):
                step = backbone.step(p.sequence, p.mask, t=t, rng=rng)
                p.sequence = step.new_sequence
                p.mask = step.new_mask
                # Twist: reward log-likelihood of the partially-denoised sequence.
                # Production swaps in the differentiable Taylor approximation.
                new_seqs.append(p.sequence)
                log_w[i] = p.score_log[-1] if p.score_log else 0.0
            r = reward.reward(new_seqs)
            # Twist update = beta * (new_reward - prev_reward).
            for i, p in enumerate(particles):
                prev = p.score_log[-1] if p.score_log else 0.0
                p.score_log.append(float(r[i]))
                log_w[i] = config.beta * (float(r[i]) - prev)
            weights = np.exp(log_w - log_w.max())
            for i, p in enumerate(particles):
                p.weight = float(p.weight * weights[i])
            ess = _effective_sample_size(np.array([p.weight for p in particles]))
            if ess < config.resample_threshold_ess * len(particles):
                idx = _systematic_resample(np.array([p.weight for p in particles]), rng)
                particles = [Particle(
                    sequence=particles[j].sequence,
                    mask=particles[j].mask.copy(),
                    weight=1.0,
                    score_log=list(particles[j].score_log),
                ) for j in idx]
            if return_trajectory:
                trace.append({
                    "t": t,
                    "ess": ess,
                    "mean_reward": float(np.mean(r)),
                })
        return particles

    # Outer iterations: denoise → re-noise to T*frac → denoise again.
    particles = _denoise_pass(particles, config.n_denoise_steps)
    for _outer in range(config.n_outer_iterations - 1):
        # Re-noise: mask a fraction of positions at random in each particle.
        n_renoise = int(round(edit_budget * config.outer_renoise_fraction))
        for p in particles:
            if n_renoise <= 0:
                continue
            unmasked = np.flatnonzero(~p.mask)
            if len(unmasked) == 0:
                continue
            renoise_idx = rng.choice(unmasked, size=min(n_renoise, len(unmasked)), replace=False)
            p.mask[renoise_idx] = True
        # Denoise again, but only for the renoised steps.
        particles = _denoise_pass(particles, max(1, config.n_denoise_steps // 2))

    final_rewards = reward.reward([p.sequence for p in particles])
    return {
        "particles": particles,
        "rewards": final_rewards,
        "trace": trace if return_trajectory else None,
    }


def best_particle(result: dict) -> Particle:
    """Return the highest-reward particle from a run_twisted_smc result."""
    rewards = result["rewards"]
    particles = result["particles"]
    return particles[int(np.argmax(rewards))]
