"""Lightweight structure/plausibility sanity checks for EditGuard demos."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .mutations import parse_mutation_notation


HYDROPHOBIC = set("AILMFWYV")
CHARGED = set("DEKRH")
POLAR = set("STNQCY")


def sequence_plausibility_features(generated: pd.DataFrame) -> pd.DataFrame:
    """Add lightweight sequence sanity features before heavier structure jobs."""
    out = generated.copy()
    seqs = out["mutated_sequence"].astype(str) if "mutated_sequence" in out.columns else pd.Series([""] * len(out))
    lengths = seqs.str.len().replace(0, np.nan)
    out["sequence_length"] = lengths
    out["frac_hydrophobic"] = seqs.apply(lambda s: sum(a in HYDROPHOBIC for a in s) / max(len(s), 1))
    out["frac_charged"] = seqs.apply(lambda s: sum(a in CHARGED for a in s) / max(len(s), 1))
    out["frac_polar"] = seqs.apply(lambda s: sum(a in POLAR for a in s) / max(len(s), 1))
    out["frac_cysteine"] = seqs.apply(lambda s: s.count("C") / max(len(s), 1))
    out["has_stop"] = seqs.str.contains(r"\*", regex=True)
    out["structure_sanity_ready"] = out["mutated_sequence"].notna() if "mutated_sequence" in out.columns else False
    return out


def mutation_position_table(generated: pd.DataFrame) -> pd.DataFrame:
    """Return one row per generated mutation token for structure mapping."""
    rows = []
    for idx, row in generated.iterrows():
        for tok in parse_mutation_notation(row.get("mutation_notation", "")):
            try:
                pos = int(tok[1:-1])
            except ValueError:
                continue
            rows.append(
                {
                    "candidate_index": int(idx),
                    "dataset_id": row.get("dataset_id"),
                    "mutation_notation": row.get("mutation_notation"),
                    "token": tok,
                    "position": pos,
                    "wt_aa": tok[0].upper(),
                    "mut_aa": tok[-1].upper(),
                }
            )
    return pd.DataFrame(rows)


def structure_sanity_stub(generated: pd.DataFrame) -> pd.DataFrame:
    """Create a report table with placeholders for ESMFold/ProteinMPNN outputs."""
    out = sequence_plausibility_features(generated)
    for col in [
        "esmfold_plddt",
        "esmfold_ptm",
        "structure_similarity_tm",
        "proteinmpnn_score",
    ]:
        if col not in out.columns:
            out[col] = np.nan
    return out
