"""Local CLI: fit phase boundaries on the canonical parquet."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phaseagent.phase import (  # noqa: E402
    bootstrap_phase_boundary,
    classify_phase_regime,
    compute_viability_by_distance,
    fit_phase_boundary,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--min-count-per-distance", type=int, default=10)
    args = p.parse_args()

    df = pd.read_parquet(args.data)
    boundaries, v_long, boots = [], [], []
    for ds_id, sub in df.groupby("dataset_id"):
        v = compute_viability_by_distance(sub, min_count_per_distance=args.min_count_per_distance)
        fit = fit_phase_boundary(v)
        regime = classify_phase_regime(fit["dc"], fit["alpha"], fit.get("r2"), v)
        v["dataset_id"] = ds_id
        v["dc"] = fit["dc"]
        v_long.append(v)
        boundaries.append({"dataset_id": ds_id, "regime": regime, **fit})
        if args.bootstrap > 0:
            b = bootstrap_phase_boundary(
                sub, n_boot=args.bootstrap, min_count_per_distance=args.min_count_per_distance
            )
            b["dataset_id"] = ds_id
            boots.append(b)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(boundaries).to_csv(out_path, index=False)
    pd.concat(v_long, ignore_index=True).to_csv(out_path.parent / "viability_by_distance.csv", index=False)
    if boots:
        pd.concat(boots, ignore_index=True).to_csv(
            out_path.parent / "bootstrap_boundaries.csv", index=False
        )
    print(f"Wrote {len(boundaries)} boundary rows to {out_path}")


if __name__ == "__main__":
    main()
