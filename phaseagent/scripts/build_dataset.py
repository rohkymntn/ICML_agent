"""Local CLI: build the canonical parquet from a directory of ProteinGym CSVs.

For Modal-native execution, prefer:
    modal run modal_app.py::build_dataset
This script is provided so the same logic can be exercised on a small local
slice (e.g. tests/fixtures or a partial download) without the Modal harness.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is importable when running as `python scripts/build_dataset.py`.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phaseagent.proteingym import build_proteingym_dataset  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", required=True, help="Directory of per-assay CSVs.")
    p.add_argument("--out", required=True, help="Output parquet path.")
    p.add_argument("--summary", default="outputs/tables/dataset_summary.csv")
    p.add_argument("--fitness-normalization", default="rank")
    p.add_argument("--threshold-mode", default="quantile")
    p.add_argument("--threshold-value", type=float, default=0.75)
    p.add_argument("--n-datasets", type=int, default=None)
    p.add_argument("--min-rows", type=int, default=100)
    p.add_argument("--max-rows", type=int, default=200_000)
    args = p.parse_args()

    full, summary = build_proteingym_dataset(
        raw_dir=Path(args.raw_dir),
        fitness_normalization=args.fitness_normalization,
        threshold_mode=args.threshold_mode,
        threshold_value=args.threshold_value,
        n_datasets=args.n_datasets,
        min_rows=args.min_rows,
        max_rows=args.max_rows,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out, index=False)
    sum_path = Path(args.summary)
    sum_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(sum_path, index=False)
    print(f"Wrote {len(full)} rows from {len(summary)} datasets to {out}")


if __name__ == "__main__":
    main()
