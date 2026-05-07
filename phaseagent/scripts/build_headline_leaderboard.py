"""Build the publication-ready test-split leaderboard with bootstrap CIs and
paired Wilcoxon p-values.

Reads every ``outputs/editguard/*_metrics.csv`` produced by Phase 1/2 runs,
asserts they only contain test-split datasets, and writes:

- ``outputs/editguard/headline_leaderboard.csv``: one row per method with
  mean / median / 5th / 95th percentile of functional_hit_rate and
  joint_success_rate (90% bootstrap CI, n_boot=1000).
- ``outputs/editguard/headline_pairwise_wilcoxon.csv``: pairwise Wilcoxon
  signed-rank tests between every method, paired by (dataset_id, objective,
  edit_budget, seed). Holm-Bonferroni corrected p-values across the family
  of comparisons against the headline reference method.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phaseagent.edit_eval import bootstrap_ci, paired_method_test  # noqa: E402
from phaseagent.edit_splits import assert_tasks_in_split  # noqa: E402


HEADLINE_METRICS = [
    "outputs/editguard/edit_baseline_metrics.csv",
    "outputs/editguard/dms_pool_guided_metrics.csv",
    "outputs/editguard/esm2_masked_marginal_metrics.csv",
    "outputs/editguard/vep_metrics.csv",
    "outputs/editguard/esm_if_metrics.csv",
]


def holm_bonferroni(pvals: list[float]) -> list[float]:
    """Holm step-down correction over a family of p-values."""
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    corrected = [0.0] * n
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = pvals[idx] * (n - rank)
        adj = min(adj, 1.0)
        running_max = max(running_max, adj)
        corrected[idx] = running_max
    return corrected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-files", nargs="+", default=HEADLINE_METRICS)
    ap.add_argument("--splits", default="outputs/editguard/edit_splits.csv")
    ap.add_argument("--require-split", default="test")
    ap.add_argument("--leaderboard-out", default="outputs/editguard/headline_leaderboard.csv")
    ap.add_argument("--wilcoxon-out", default="outputs/editguard/headline_pairwise_wilcoxon.csv")
    ap.add_argument("--reference-method", default="dms_prior_rerank")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--ci-confidence", type=float, default=0.90)
    args = ap.parse_args()

    frames = []
    for f in args.metrics_files:
        path = Path(f)
        if not path.exists():
            print(f"[lb] missing {path}, skipping")
            continue
        sub = pd.read_csv(path)
        if len(sub) == 0:
            continue
        frames.append(sub)
    if not frames:
        raise SystemExit("no metrics CSVs found")
    metrics = pd.concat(frames, ignore_index=True, sort=False)
    print(f"[lb] loaded {len(metrics)} metric rows across {metrics['method'].nunique()} methods")

    splits = pd.read_csv(args.splits)
    if args.require_split:
        # Build a fake "tasks" frame that just has dataset_ids — assert_tasks_in_split
        # only reads the dataset_id column.
        check = metrics[["dataset_id"]].drop_duplicates()
        assert_tasks_in_split(check, splits, args.require_split)
        print(f"[lb] split discipline OK: all {len(check)} datasets in split={args.require_split!r}")

    rows = []
    for method, sub in metrics.groupby("method"):
        for metric in ("functional_hit_rate", "joint_success_rate", "constraint_satisfaction_rate"):
            if metric not in sub.columns:
                continue
            vals = sub[metric].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            lo, hi = bootstrap_ci(
                vals, n_boot=args.n_boot, confidence=args.ci_confidence, seed=0
            )
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "std": float(np.std(vals)),
                    "n": int(vals.size),
                }
            )
    leaderboard = pd.DataFrame(rows)
    out_lb = Path(args.leaderboard_out)
    out_lb.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(out_lb, index=False)
    print(f"\n[lb] wrote {out_lb}")
    pivot = leaderboard.pivot_table(
        index="method",
        columns="metric",
        values="mean",
    ).sort_values("functional_hit_rate", ascending=False)
    print(pivot.round(3).to_string())

    # Pairwise Wilcoxon vs the reference method.
    methods_present = sorted(metrics["method"].unique())
    if args.reference_method not in methods_present:
        print(f"[lb] reference method {args.reference_method!r} not present; skipping pairwise tests")
        return
    others = [m for m in methods_present if m != args.reference_method]
    rows = []
    raw_p = []
    for other in others:
        for metric in ("functional_hit_rate", "joint_success_rate"):
            r = paired_method_test(metrics, args.reference_method, other, metric=metric)
            rows.append(
                {
                    "method_a": args.reference_method,
                    "method_b": other,
                    "metric": metric,
                    **r,
                }
            )
            if metric == "functional_hit_rate":
                raw_p.append(r["p_value"])
    holm_p_hit = holm_bonferroni([float(p) for p in raw_p])
    j = 0
    for r in rows:
        if r["metric"] == "functional_hit_rate":
            r["p_holm_within_metric"] = holm_p_hit[j]
            j += 1
        else:
            r["p_holm_within_metric"] = float("nan")

    pw = pd.DataFrame(rows)
    out_pw = Path(args.wilcoxon_out)
    pw.to_csv(out_pw, index=False)
    print(f"\n[lb] wrote {out_pw}")
    print(pw.sort_values(["metric", "mean_delta"], ascending=[True, False]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
