"""Experimental / scaffolded PhaseAgent components.

Modules in this package are not part of the EditGuard scientific story and
are not loaded by the EditGuard pipeline. They are scaffolds for the future
PhaseAgent submission (neural Spectral Transformer survival residual model,
invariant structure-aware GNN, PDB/contact-map utilities). They were moved
here to make it visually obvious that they are not wired into any
EditGuard claim.

Importing anything from this package emits a warning so reviewers and
future contributors can immediately tell that they're touching unverified
scaffolding.
"""
import warnings

warnings.warn(
    "phaseagent._experimental: scaffolded modules not used by EditGuard. "
    "These are placeholders for future PhaseAgent work; no EditGuard claim "
    "depends on them.",
    stacklevel=2,
)
