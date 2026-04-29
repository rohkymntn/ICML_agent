"""Local CLI: placeholder trainer for neural survival models.

This command intentionally fails with guidance until enough real/synthetic
training examples are assembled. Deterministic baselines are available through
``run_large_deviation.py`` and ``eval_survival_model.py``.
"""
from __future__ import annotations

import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spectra")
    p.add_argument("--targets")
    p.add_argument("--out")
    p.parse_args()
    raise SystemExit(
        "Neural training is scaffolded in phaseagent.survival_model, but this "
        "CLI needs a curated target table of combinatorial survival curves. "
        "Run scripts/run_large_deviation.py first for deterministic baselines."
    )


if __name__ == "__main__":
    main()
