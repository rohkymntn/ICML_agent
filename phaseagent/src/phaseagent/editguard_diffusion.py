"""DPLM-style discrete edit diffusion interfaces for EditGuard.

This module keeps the diffusion backbone pluggable. The first implementation is
a measured-pool simulator used for reproducible DMS evaluation; a real DPLM
adapter can implement the same ``sample`` interface once checkpoints are wired.
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


class MeasuredPoolEditDiffusion:
    """DMS-measured candidate pool used as a controlled diffusion surrogate.

    This is not claimed as the final generative model. It lets the whole
    EditGuard guidance/evaluation stack run against true DMS labels while the
    DPLM adapter is integrated.
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
        out["method"] = "editguard_diffusion"
        out["diffusion_backbone"] = "measured_pool_surrogate"
        return out
