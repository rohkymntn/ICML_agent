"""EditGuard sampling backbones.

The guidance framework is pluggable: ``DMSPoolGuidedSampler`` selects from a
measured DMS candidate pool (a strong selection baseline, not a generative
model); ``DPLMEditBackbone`` (Phase 2 of the implementation plan) generates
sequences with a real masked discrete diffusion backbone and the same
classifier-guidance interface.

Method labels emitted by this module are deliberately specific: the DMS-pool
sampler reports ``method = "dms_pool_guided"`` and the (future) DPLM-backed
sampler reports ``method = "editguard_diffusion_dplm"``. The bare label
``editguard_diffusion`` is reserved for the real DPLM-backed sampler and is
no longer emitted by the DMS-pool surrogate.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .editing_tasks import EditingTask
from .editguard_prior import DMSFunctionPrior
from .editguard_sampling import softmax_sample_guided_edits


@dataclass
class DiffusionSampleConfig:
    n_samples: int = 50
    temperature: float = 1.0
    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 2.0
    kappa: float = 0.0
    seed: int = 0


class DPLMEditBackbone:
    """Adapter contract for a DPLM-style masked discrete diffusion backbone."""

    def supports_guidance_hooks(self) -> bool:
        """Whether denoising logits/probabilities can be modified at sampling time."""
        return False

    def sample(self, wt_sequence: str, edit_mask, objective_context: dict, n_samples: int = 50):
        raise NotImplementedError(
            "Wire a public DPLM implementation here. Required hooks: masked "
            "infilling, fixed-position constraints, and denoising logits."
        )


class DMSPoolGuidedSampler:
    """Softmax sampling from a measured DMS pool, scored by a DMS function prior.

    This is a *selection* method, not a generative model. It samples from the
    measured candidate pool with probability proportional to the EditGuard
    guided score (function prior × objective × constraints). It is included
    in headline benchmarks as a strong selection baseline that has full
    access to measured DMS pool labels.
    """

    def __init__(self, prior: DMSFunctionPrior):
        self.prior = prior

    def sample(
        self,
        candidates: pd.DataFrame,
        task: EditingTask,
        config: DiffusionSampleConfig | None = None,
    ) -> pd.DataFrame:
        config = config or DiffusionSampleConfig()
        out = softmax_sample_guided_edits(
            candidates,
            task,
            self.prior,
            k=config.n_samples,
            temperature=config.temperature,
            seed=config.seed,
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma,
            kappa=config.kappa,
        )
        out["method"] = "dms_pool_guided"
        out["diffusion_backbone"] = "measured_pool"
        return out


# Backwards-compatibility alias. New code should use DMSPoolGuidedSampler.
MeasuredPoolEditDiffusion = DMSPoolGuidedSampler
