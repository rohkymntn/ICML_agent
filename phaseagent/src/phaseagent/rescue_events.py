"""Extract rescue-event training pairs from multi-mutant fitness data.

A *rescue event* is the central data primitive of the EditGuard-Clin atlas:
given a damaging first mutation ``m1`` (single-mutant fitness < threshold),
identify second-site mutations ``m2`` such that the combined variant ``(m1, m2)``
has higher fitness than ``m1`` alone — or ideally restored fitness above the
viability threshold.

Three flavors of "rescue":

- **partial**: combined fitness > single fitness (any improvement)
- **full**: combined fitness >= viability threshold (restored to functional)
- **super-compensatory**: combined fitness >= WT-equivalent (over-rescued)

We extract rescue events from two sources:

1. **Megascale doubles** (RosettaCommons/MegaScale, dataset2): folding
   stability rescues. ``rescue`` here means the second mutation buys back
   ΔΔG that the first mutation lost.
2. **ProteinGym multi-mutant assays** (e.g. BRCA1, PTEN, TPMT, GFP, P53):
   functional rescues in disease-relevant proteins.

The output is a uniform schema each row of which is one (m1, m2, observed
combined effect, rescue_type) record, with the per-row protein/dataset
context preserved so downstream models can condition on it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .mutations import parse_mutation_notation


# Default thresholds (tunable per assay; defaults match standard practice).
DEFAULT_DAMAGING_QUANTILE = 0.25       # bottom-quartile fitness = "damaging"
DEFAULT_VIABLE_QUANTILE = 0.50         # above-median fitness = "functional"
DEFAULT_WT_EQUIVALENT_QUANTILE = 0.75  # top-quartile = "WT-equivalent"


@dataclass(frozen=True)
class RescueThresholds:
    """Per-protein thresholds derived from the single-mutant fitness distribution."""

    damaging: float
    viable: float
    wt_equivalent: float
    n_singles: int


def _parse_double_notation(notation: str) -> tuple[str, str] | None:
    """Return the two single-mutation tokens of a double-mutant notation.

    Notation is colon- or comma-separated 1-letter style (e.g. ``"P123A:K456E"``).
    Returns None if the notation does not parse to exactly two single mutations.
    """
    tokens = parse_mutation_notation(notation.replace(",", ":"))
    if len(tokens) != 2:
        return None
    return tokens[0], tokens[1]


def derive_thresholds(
    single_df: pd.DataFrame,
    score_col: str = "fitness_norm",
    damaging_q: float = DEFAULT_DAMAGING_QUANTILE,
    viable_q: float = DEFAULT_VIABLE_QUANTILE,
    wt_q: float = DEFAULT_WT_EQUIVALENT_QUANTILE,
) -> RescueThresholds:
    """Compute the three rescue-classification thresholds for one protein."""
    s = pd.to_numeric(single_df[score_col], errors="coerce").dropna()
    if len(s) < 4:
        # Truly insufficient data; fall back to permissive thresholds.
        return RescueThresholds(
            damaging=float("-inf"), viable=0.5, wt_equivalent=0.75, n_singles=int(len(s))
        )
    return RescueThresholds(
        damaging=float(s.quantile(damaging_q)),
        viable=float(s.quantile(viable_q)),
        wt_equivalent=float(s.quantile(wt_q)),
        n_singles=int(len(s)),
    )


def _classify(score: float, thr: RescueThresholds) -> str:
    if not np.isfinite(score):
        return "unknown"
    if score >= thr.wt_equivalent:
        return "wt_equivalent"
    if score >= thr.viable:
        return "viable"
    if score <= thr.damaging:
        return "damaging"
    return "intermediate"


def _classify_rescue(
    m1_score: float,
    combined_score: float,
    thr: RescueThresholds,
) -> str:
    """Return one of: ``no_rescue``, ``partial``, ``full``, ``super``."""
    if not (np.isfinite(m1_score) and np.isfinite(combined_score)):
        return "unknown"
    if combined_score >= thr.wt_equivalent and m1_score < thr.viable:
        return "super"
    if combined_score >= thr.viable and m1_score < thr.viable:
        return "full"
    if combined_score > m1_score + 0.05:  # at least 5pt fitness bump
        return "partial"
    return "no_rescue"


RESCUE_EVENT_COLUMNS = (
    "dataset_id", "m1_notation", "m2_notation",
    "m1_pos", "m2_pos", "m1_aa", "m2_aa",
    "m1_score", "m2_score", "combined_score",
    "rescue_class", "m1_class", "m2_class",
    "delta_rescue", "protein_n_singles",
    "threshold_damaging", "threshold_viable",
)


def _empty_rescue_frame() -> pd.DataFrame:
    """Return an empty rescue-events DataFrame with the expected column schema."""
    return pd.DataFrame({col: pd.Series(dtype=object) for col in RESCUE_EVENT_COLUMNS})


def extract_rescue_events(
    multi_df: pd.DataFrame,
    single_df: pd.DataFrame,
    *,
    score_col: str = "fitness_norm",
    only_damaging_first: bool = True,
    require_both_singles_measured: bool = True,
    damaging_q: float = DEFAULT_DAMAGING_QUANTILE,
    viable_q: float = DEFAULT_VIABLE_QUANTILE,
    wt_q: float = DEFAULT_WT_EQUIVALENT_QUANTILE,
) -> pd.DataFrame:
    """Build a rescue-event table from per-protein single + double mutant data.

    Each output row is one (m1, m2) ordered pair where m1 is the *damaging*
    variant being rescued and m2 is the *candidate rescuer* (or partial
    rescuer). Both orderings of the underlying double are emitted when both
    halves are damaging — those rows describe mutual rescue.

    Output columns:
        dataset_id, m1_notation, m2_notation, m1_pos, m2_pos, m1_aa, m2_aa,
        m1_score, m2_score, combined_score,
        rescue_class (no_rescue / partial / full / super / unknown),
        m1_class, m2_class,
        delta_rescue (combined - m1),
        protein_n_singles, threshold_damaging, threshold_viable.
    """
    if "dataset_id" not in multi_df.columns or "dataset_id" not in single_df.columns:
        raise ValueError("both multi_df and single_df must include dataset_id")
    if len(multi_df) == 0:
        return _empty_rescue_frame()

    out_rows = []
    for ds_id, multi_sub in multi_df.groupby("dataset_id"):
        ds_singles = single_df[single_df["dataset_id"] == ds_id]
        if len(ds_singles) == 0:
            continue
        thr = derive_thresholds(
            ds_singles, score_col=score_col,
            damaging_q=damaging_q, viable_q=viable_q, wt_q=wt_q,
        )
        # Per-protein single-mutation lookup: notation -> score.
        singles_lookup: dict[str, float] = {}
        for notation, score in zip(
            ds_singles["mutation_notation"].astype(str),
            pd.to_numeric(ds_singles[score_col], errors="coerce"),
        ):
            tokens = parse_mutation_notation(notation)
            if len(tokens) == 1 and np.isfinite(score):
                singles_lookup[tokens[0]] = float(score)

        for _, row in multi_sub.iterrows():
            notation = str(row.get("mutation_notation", ""))
            pair = _parse_double_notation(notation)
            if pair is None:
                continue
            tok_a, tok_b = pair
            score_a = singles_lookup.get(tok_a)
            score_b = singles_lookup.get(tok_b)
            if require_both_singles_measured and (score_a is None or score_b is None):
                continue
            combined = pd.to_numeric(row.get(score_col), errors="coerce")
            if not np.isfinite(combined):
                continue
            for first, second, score_first, score_second in (
                (tok_a, tok_b, score_a, score_b),
                (tok_b, tok_a, score_b, score_a),
            ):
                if score_first is None:
                    continue
                if only_damaging_first and score_first > thr.damaging:
                    continue
                rescue_class = _classify_rescue(score_first, float(combined), thr)
                out_rows.append(
                    {
                        "dataset_id": str(ds_id),
                        "m1_notation": first,
                        "m2_notation": second,
                        "m1_pos": int(first[1:-1]) if first[1:-1].lstrip("-").isdigit() else -1,
                        "m2_pos": int(second[1:-1]) if second[1:-1].lstrip("-").isdigit() else -1,
                        "m1_aa": first[-1].upper(),
                        "m2_aa": second[-1].upper(),
                        "m1_score": float(score_first),
                        "m2_score": float(score_second) if score_second is not None else float("nan"),
                        "combined_score": float(combined),
                        "rescue_class": rescue_class,
                        "m1_class": _classify(score_first, thr),
                        "m2_class": _classify(score_second, thr) if score_second is not None else "unknown",
                        "delta_rescue": float(combined) - float(score_first),
                        "protein_n_singles": thr.n_singles,
                        "threshold_damaging": thr.damaging,
                        "threshold_viable": thr.viable,
                    }
                )
    if not out_rows:
        return _empty_rescue_frame()
    return pd.DataFrame(out_rows)


def summarize_rescue_events(events: pd.DataFrame) -> pd.DataFrame:
    """Per-protein summary of rescue events for sanity-checking the extractor."""
    if len(events) == 0:
        return pd.DataFrame()
    agg = (
        events.groupby("dataset_id")
        .agg(
            n_pairs=("m1_notation", "size"),
            n_unique_m1=("m1_notation", "nunique"),
            n_partial=("rescue_class", lambda x: int((x == "partial").sum())),
            n_full=("rescue_class", lambda x: int((x == "full").sum())),
            n_super=("rescue_class", lambda x: int((x == "super").sum())),
            n_no_rescue=("rescue_class", lambda x: int((x == "no_rescue").sum())),
            mean_delta=("delta_rescue", "mean"),
            max_delta=("delta_rescue", "max"),
        )
        .reset_index()
    )
    agg["rescue_rate"] = (
        (agg["n_partial"] + agg["n_full"] + agg["n_super"]) / agg["n_pairs"]
    )
    agg["full_or_super_rate"] = (agg["n_full"] + agg["n_super"]) / agg["n_pairs"]
    return agg.sort_values("n_pairs", ascending=False).reset_index(drop=True)
