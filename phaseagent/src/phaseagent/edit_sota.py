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
    BaselineSpec("random_direct_generation", "weak", "direct generator sanity check", False, False, False, False, True, "implemented"),
    BaselineSpec("generate_many_then_rerank", "core", "post-hoc DMS reranking baseline", True, True, False, False, True, "implemented"),
    BaselineSpec("guided_local_generation", "ours", "DMS guidance during generation", True, False, False, False, True, "implemented"),
    BaselineSpec("esm2_masked_marginal", "sequence_generation", "PLM masked edit proposal", False, False, False, False, True, "proxy_implemented"),
    BaselineSpec("esm2_masked_marginal_dms_rerank", "sequence_generation", "PLM proposal plus DMS rerank", True, True, False, False, True, "proxy_implemented"),
    BaselineSpec("dplm_masked_infilling", "diffusion_generation", "DPLM masked infilling", False, False, False, False, True, "adapter_pending"),
    BaselineSpec("dplm_dms_rerank", "diffusion_generation", "DPLM proposal plus DMS rerank", True, True, False, False, True, "adapter_pending"),
    BaselineSpec("editguard_guided_dplm", "ours", "DMS-guided DPLM denoising", True, False, False, False, True, "adapter_pending"),
    BaselineSpec("proteinmpnn", "structure_conditioned", "inverse folding baseline", False, False, True, False, True, "external_optional"),
    BaselineSpec("proteinmpnn_dms_rerank", "structure_conditioned", "ProteinMPNN plus DMS rerank", True, True, True, False, True, "hook_implemented"),
    BaselineSpec("ligandmpnn", "structure_conditioned", "ligand-aware inverse folding baseline", False, False, True, False, True, "external_optional"),
    BaselineSpec("esm_if1", "structure_conditioned", "inverse folding baseline", False, False, True, False, True, "external_optional"),
    BaselineSpec("tranception", "vep_ranker", "autoregressive VEP/ranking baseline", False, False, False, False, False, "external_optional"),
    BaselineSpec("eve_evmutation_gemme", "vep_ranker", "MSA/evolutionary VEP baseline", False, False, False, True, False, "external_optional"),
    BaselineSpec("adalead", "optimization", "local sequence optimization baseline", True, False, False, False, True, "planned"),
    BaselineSpec("bayesian_optimization", "optimization", "mutation-token optimizer", True, False, False, False, True, "planned"),
)


def baseline_registry_frame() -> pd.DataFrame:
    """Return baseline metadata for reporting and reproducibility."""
    return pd.DataFrame([asdict(x) for x in BASELINE_REGISTRY])
