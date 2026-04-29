"""Misc helpers."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]
