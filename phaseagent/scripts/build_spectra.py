"""Local CLI: build single-mutant spectrum tables."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phaseagent.spectrum import (  # noqa: E402
    build_spectrum_tokens,
    compute_single_mutant_effects,
    spectrum_summary_features,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    df = pd.read_parquet(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries, tokens, effects = [], [], []
    for ds_id, sub in df.groupby("dataset_id"):
        summaries.append(spectrum_summary_features(sub))
        tok = build_spectrum_tokens(sub)
        eff = compute_single_mutant_effects(sub)
        tok["dataset_id"] = ds_id
        eff["dataset_id"] = ds_id
        tokens.append(tok)
        effects.append(eff)

    pd.DataFrame(summaries).to_csv(out_dir / "spectrum_summary.csv", index=False)
    pd.concat(tokens, ignore_index=True).to_parquet(out_dir / "spectrum_tokens.parquet", index=False)
    pd.concat(effects, ignore_index=True).to_parquet(out_dir / "single_mutant_effects.parquet", index=False)
    print(f"Wrote spectra for {len(summaries)} datasets to {out_dir}")


if __name__ == "__main__":
    main()
