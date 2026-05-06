"""Baseline registry for ICML-level EditGuard comparisons."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import pandas as pd


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    tier: str
    role: str
    uses_dms_train: bool
    uses_dms_rerank: bool
    uses_structure: bool
    uses_msa: bool
    generates_directly: bool
    status: str


BASELINE_REGISTRY = (
    BaselineSpec("random", "weak", "uniform random selection from measured DMS pool", False, False, False, False, False, "implemented"),
    BaselineSpec("novelty", "weak", "highest mutation distance first", False, False, False, False, False, "implemented"),
    BaselineSpec("objective_only", "weak", "task objective score, no DMS guidance", False, False, False, False, False, "implemented"),
    BaselineSpec("aa_frequency_proposal", "weak", "swissprot AA-frequency baseline (NOT a PLM)", False, False, False, False, True, "implemented"),
    BaselineSpec("random_direct_generation", "weak", "WT-anchored random substitution generator", False, False, False, False, True, "implemented"),
    BaselineSpec("generate_many_then_rerank", "core", "random WT-anchored generator + DMS prior rerank", True, True, False, False, True, "implemented"),
    BaselineSpec("guided_local_generation", "core", "DMS-guided WT-anchored generation", True, False, False, False, True, "implemented"),
    BaselineSpec("dms_prior_rerank", "core", "DMS function prior rerank of measured pool", True, True, False, False, False, "implemented"),
    BaselineSpec("editguard_guided", "ours", "guided selection (function prior + objective + constraints)", True, False, False, False, False, "implemented"),
    BaselineSpec("dms_pool_guided", "ours", "softmax-sampled selection from measured pool with prior guidance", True, False, False, False, False, "implemented"),
    BaselineSpec("esm2_masked_marginal", "sequence_generation", "real ESM-2 masked-marginal proposal (GPU)", False, False, False, False, True, "implemented"),
    BaselineSpec("esm2_masked_marginal_dms_rerank", "sequence_generation", "real ESM-2 proposal + DMS prior rerank", True, True, False, False, True, "implemented"),
    BaselineSpec("editguard_diffusion_dplm", "ours", "DMS-guided DPLM denoising (Phase 2 of plan)", True, False, False, False, True, "adapter_pending"),
    BaselineSpec("proteinmpnn", "structure_conditioned", "inverse folding baseline", False, False, True, False, True, "phase2_planned"),
    BaselineSpec("tranception_rerank", "vep_ranker", "Tranception score rerank (precomputed by ProteinGym)", False, True, False, False, False, "phase2_planned"),
    BaselineSpec("eve_rerank", "vep_ranker", "EVE score rerank (precomputed by ProteinGym)", False, True, False, True, False, "phase2_planned"),
)


def baseline_registry_frame() -> pd.DataFrame:
    """Return baseline metadata for reporting and reproducibility."""
    return pd.DataFrame([asdict(x) for x in BASELINE_REGISTRY])
