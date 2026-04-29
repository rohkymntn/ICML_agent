"""Helpers for picking viability thresholds."""
from __future__ import annotations

import numpy as np
import pandas as pd


def quantile_threshold(values, q: float) -> float:
    return float(np.quantile(np.asarray(values), q))


def absolute_threshold(value: float) -> float:
    return float(value)


def viability_cutoff(df: pd.DataFrame, mode: str, value: float) -> float:
    if mode == "quantile":
        return quantile_threshold(df["fitness_norm"], value)
    if mode == "absolute":
        return absolute_threshold(value)
    raise ValueError(f"Unknown threshold mode: {mode}")
