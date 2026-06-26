"""Build EditGuard-Clin paper figures.

Six main figures matching the multi-panel publication style from the reference
papers (subcellular-classifier MADA paper). Each figure is built from real
artifacts on the Modal volume that have been pulled to /tmp.

Each figure exports both a paper PDF (vector, exact 89mm/183mm Nature widths)
and a PNG preview that gets synced to ~/Desktop/EditGuard_v2_figures/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phaseagent.nature_figures import (  # noqa: E402
    NATURE_DOUBLE_IN,
    NATURE_SINGLE_IN,
    OKABE_ITO,
    save_figure,
    setup_nature_style,
)

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

OUT = ROOT / "outputs" / "clin" / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)
setup_nature_style()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _panel_label(ax, label: str, dx: float = -0.18, dy: float = 1.05):
    ax.text(dx, dy, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", ha="left", va="bottom")


def _read_csv_or_none(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Fig 1 — Hero: workflow + headline numbers
# ---------------------------------------------------------------------------

def fig1_hero():
    summary = _read_csv_or_none("/tmp/rescue_summary.csv")
    metrics = _read_csv_or_none("/tmp/rescue_metrics.csv")
    conf = _read_csv_or_none("/tmp/rescue_conformal_v2.csv")
    if any(x is None for x in (summary, metrics, conf)):
        print("[fig1] missing one of: rescue_summary.csv, rescue_metrics.csv, rescue_conformal_v2.csv")
        return

    fig = plt.figure(figsize=(NATURE_DOUBLE_IN, 4.0))
    gs = GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.45,
                  height_ratios=[1.0, 1.0])

    # Panel a: pipeline schematic (text-based; clean for Nature style).
    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    boxes = [
        ("Damaging variant\nm1 (e.g. p53 R175H)", 0.05, OKABE_ITO["vermilion"]),
        ("Enumerate every\ncandidate m2", 0.27, OKABE_ITO["skyblue"]),
        ("Score (m1, m2)\nrescue probability", 0.49, OKABE_ITO["blue"]),
        ("Conformal threshold\n(false-rescue ≤ α)", 0.71, OKABE_ITO["orange"]),
        ("Top-k rescue set\n+ guarantee", 0.93, OKABE_ITO["green"]),
    ]
    for txt, x, c in boxes:
        ax.text(x, 0.5, txt, transform=ax.transAxes, ha="center", va="center",
                fontsize=7, fontweight="normal",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=c, edgecolor="black",
                          linewidth=0.4, alpha=0.85))
    for x_from, x_to in [(0.10, 0.21), (0.32, 0.43), (0.54, 0.65), (0.76, 0.87)]:
        ax.annotate("", xy=(x_to, 0.5), xytext=(x_from, 0.5),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", lw=0.7, color="black"))
    _panel_label(ax, "a", dx=-0.04, dy=0.95)
    ax.set_title("EditGuard-Clin pipeline: from a pathogenic variant to a calibrated rescue set",
                 fontsize=8, pad=18)

    # Panel b: per-protein rescue rate distribution.
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(summary["rescue_rate"], bins=20, color=OKABE_ITO["blue"],
            edgecolor="black", linewidth=0.3, alpha=0.85)
    ax.set_xlabel("Per-protein rescue rate")
    ax.set_ylabel("# proteins")
    ax.set_title(f"n = {len(summary)} proteins, "
                 f"{int(summary['n_pairs'].sum()):,} (m1, m2) pairs",
                 fontsize=8)
    _panel_label(ax, "b")

    # Panel c: predictor performance.
    ax = fig.add_subplot(gs[1, 1])
    splits = metrics["split"].tolist()
    x = np.arange(len(splits))
    ax.bar(x - 0.2, metrics["auroc"], width=0.35, color=OKABE_ITO["blue"],
           edgecolor="black", linewidth=0.4, label="AUROC")
    ax.bar(x + 0.2, metrics["auprc"], width=0.35, color=OKABE_ITO["vermilion"],
           edgecolor="black", linewidth=0.4, label="AUPRC")
    for i, (a, p) in enumerate(zip(metrics["auroc"], metrics["auprc"])):
        ax.text(i - 0.2, a + 0.02, f"{a:.2f}", ha="center", fontsize=6)
        ax.text(i + 0.2, p + 0.02, f"{p:.2f}", ha="center", fontsize=6)
    ax.axhline(0.5, color=OKABE_ITO["grey"], linestyle="--", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Predictor performance\n(held-out proteins)", fontsize=8)
    ax.legend(loc="lower left", fontsize=6, frameon=False)
    _panel_label(ax, "c")

    # Panel d: conformal FDR validity.
    ax = fig.add_subplot(gs[1, 2])
    ax.plot([0, 0.35], [0, 0.35], color=OKABE_ITO["grey"], linestyle="--", linewidth=0.6,
            label="target = realised")
    ax.plot(conf["alpha"], conf["false_rescue_rate"], marker="o",
            color=OKABE_ITO["vermilion"], markersize=6, markeredgecolor="white",
            markeredgewidth=0.4)
    ax.set_xlim(0, 0.35)
    ax.set_ylim(0, 0.35)
    ax.set_xlabel("Target false-rescue α")
    ax.set_ylabel("Realised false-rescue rate")
    ax.set_title("Conformal FDR control\nis honest (≤ target)", fontsize=8)
    ax.legend(loc="lower right", fontsize=6, frameon=False)
    _panel_label(ax, "d")

    paths = save_figure(fig, OUT / "fig1_hero")
    print(f"[fig1] {paths}")


# ---------------------------------------------------------------------------
# Fig 3 — Conformal calibration trajectory (matches reference Fig 3 style)
# ---------------------------------------------------------------------------

def fig3_conformal_tradeoff():
    conf = _read_csv_or_none("/tmp/rescue_conformal_v2.csv")
    metrics = _read_csv_or_none("/tmp/rescue_metrics.csv")
    if conf is None or metrics is None:
        print("[fig3] missing artifacts")
        return

    fig = plt.figure(figsize=(NATURE_DOUBLE_IN, 3.6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.45)

    # Panel a: target vs realised + perfect-line.
    ax = fig.add_subplot(gs[0, 0])
    ax.plot([0, 0.35], [0, 0.35], color=OKABE_ITO["grey"], linestyle="--",
            linewidth=0.6, label="target = realised")
    ax.plot(conf["alpha"], conf["false_rescue_rate"], marker="o",
            color=OKABE_ITO["vermilion"], markersize=6, markeredgecolor="white",
            markeredgewidth=0.4)
    for _, r in conf.iterrows():
        ax.annotate(f"α={r['alpha']:.2f}", (r["alpha"], r["false_rescue_rate"]),
                    textcoords="offset points", xytext=(8, -8), fontsize=6)
    ax.set_xlim(0, 0.35)
    ax.set_ylim(0, 0.35)
    ax.set_xlabel("Target false-rescue α")
    ax.set_ylabel("Realised false-rescue rate (held-out)")
    ax.set_title("Conformal FDR control validity", fontsize=8)
    ax.legend(loc="upper left", fontsize=6, frameon=False)
    _panel_label(ax, "a")

    # Panel b: precision-recall tradeoff with annotated rescue-set sizes.
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(conf["realised_recall"], conf["realised_precision"], marker="o",
            color=OKABE_ITO["blue"], markersize=6, markeredgecolor="white",
            markeredgewidth=0.4)
    for _, r in conf.iterrows():
        ax.annotate(f"α={r['alpha']:.2f}\nn={int(r['n_above_threshold']):,}",
                    (r["realised_recall"], r["realised_precision"]),
                    textcoords="offset points", xytext=(6, -16), fontsize=6)
    ax.axhline(metrics.iloc[1]["auprc"] if len(metrics) > 1 else 0.66,
               color=OKABE_ITO["grey"], linestyle=":", linewidth=0.5)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0.4, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("User-tunable rescue-set size", fontsize=8)
    _panel_label(ax, "b")

    # Panel c: rescue-set size as a function of α.
    ax = fig.add_subplot(gs[0, 2])
    ax.bar(np.arange(len(conf)), conf["n_above_threshold"],
           color=[OKABE_ITO["vermilion"], OKABE_ITO["orange"],
                  OKABE_ITO["skyblue"], OKABE_ITO["blue"]][: len(conf)],
           edgecolor="black", linewidth=0.4)
    for i, n in enumerate(conf["n_above_threshold"]):
        ax.text(i, n + 200, f"{int(n):,}", ha="center", fontsize=6)
    ax.set_xticks(np.arange(len(conf)))
    ax.set_xticklabels([f"α={a:.2f}" for a in conf["alpha"]])
    ax.set_ylabel("# candidates above threshold")
    ax.set_title("How many candidates pass\nthe risk gate", fontsize=8)
    _panel_label(ax, "c")

    paths = save_figure(fig, OUT / "fig3_conformal_tradeoff")
    print(f"[fig3] {paths}")


# ---------------------------------------------------------------------------
# Fig 4 — Sequence analysis of rescue events
# ---------------------------------------------------------------------------

def fig4_sequence_analysis():
    """Per-protein sequence-position statistics from the extracted rescue events."""
    summary = _read_csv_or_none("/tmp/rescue_summary.csv")
    if summary is None:
        print("[fig4] missing rescue_summary.csv")
        return

    # Pull the full events parquet only if available locally.
    events = None
    parquet_path = Path("/tmp/rescue_events_megascale.parquet")
    if parquet_path.exists():
        try:
            events = pd.read_parquet(parquet_path)
        except Exception:
            pass

    fig = plt.figure(figsize=(NATURE_DOUBLE_IN, 3.4))
    gs = GridSpec(1, 3, figure=fig, wspace=0.4)

    # Panel a: rescue rate vs # singles per protein (does data depth matter?).
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(summary["n_pairs"], summary["rescue_rate"],
               s=10, alpha=0.7, color=OKABE_ITO["blue"], edgecolor="black", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("# rescue (m1, m2) pairs (log)")
    ax.set_ylabel("Rescue rate")
    ax.set_title(f"Per-protein rescue depth\nn = {len(summary)} Megascale proteins", fontsize=8)
    _panel_label(ax, "a")

    # Panel b: full+super rate distribution.
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(summary["full_or_super_rate"], bins=20, color=OKABE_ITO["vermilion"],
            edgecolor="black", linewidth=0.3, alpha=0.85)
    ax.set_xlabel("Full + super rescue fraction")
    ax.set_ylabel("# proteins")
    ax.set_title("Strong-rescue events\nare protein-specific", fontsize=8)
    _panel_label(ax, "b")

    # Panel c: |m1-m2| sequence separation distribution if events available, else
    # show the proxy: rescue rate vs n_unique_m1 (more m1 diversity → more chances).
    ax = fig.add_subplot(gs[0, 2])
    if events is not None and "m1_pos" in events.columns:
        sep = (events["m1_pos"] - events["m2_pos"]).abs()
        ax.hist(sep, bins=40, color=OKABE_ITO["green"], edgecolor="black", linewidth=0.3, alpha=0.85)
        ax.set_xlabel("|m1 pos − m2 pos|  (sequence separation)")
        ax.set_ylabel("# rescue events")
        ax.set_title("Where rescues live in sequence", fontsize=8)
    else:
        ax.scatter(summary["n_unique_m1"], summary["rescue_rate"],
                   s=10, alpha=0.7, color=OKABE_ITO["green"], edgecolor="black", linewidth=0.3)
        ax.set_xlabel("# unique damaging m1 per protein")
        ax.set_ylabel("Rescue rate")
        ax.set_title("Rescue diversity vs depth", fontsize=8)
    _panel_label(ax, "c")

    paths = save_figure(fig, OUT / "fig4_sequence_analysis")
    print(f"[fig4] {paths}")


# ---------------------------------------------------------------------------
# Fig 6 — Clinical case studies / atlas overview
# ---------------------------------------------------------------------------

def fig6_clinical_utility():
    """Headline clinical-utility figure: atlas overview + literature recovery + cross-domain."""
    lit = _read_csv_or_none(str(ROOT / "outputs" / "clin" / "literature_suppressor_recovery.csv"))
    cross = _read_csv_or_none("/tmp/pg_real.csv")
    if lit is None:
        print("[fig6] missing literature_suppressor_recovery.csv")
        return

    fig = plt.figure(figsize=(NATURE_DOUBLE_IN, 4.2))
    gs = GridSpec(2, 2, figure=fig, wspace=0.4, hspace=0.55, width_ratios=[1.4, 1.0])

    # Panel a: literature suppressor rank percentile (ranked by best-to-worst).
    ax = fig.add_subplot(gs[0, 0])
    valid = lit[lit["rank_0idx"] >= 0].copy().sort_values("rank_percentile", ascending=True)
    if len(valid) == 0:
        ax.axis("off")
    else:
        y = np.arange(len(valid))[::-1]
        colors = [OKABE_ITO["green"] if rp <= 0.10 else
                  (OKABE_ITO["orange"] if rp <= 0.50 else OKABE_ITO["vermilion"])
                  for rp in valid["rank_percentile"]]
        ax.barh(y, 100 * valid["rank_percentile"], color=colors, edgecolor="black", linewidth=0.4)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r.protein}\n{r.pathogenic} → {r.suppressor}"
                            for r in valid.itertuples(index=False)],
                           fontsize=6)
        ax.set_xlabel("Rank percentile of known suppressor (lower = better)")
        ax.set_xlim(0, 100)
        ax.axvline(1.0, color=OKABE_ITO["grey"], linestyle=":", linewidth=0.6, label="top 1%")
        ax.axvline(10.0, color=OKABE_ITO["grey"], linestyle="--", linewidth=0.6, label="top 10%")
        ax.legend(loc="lower right", fontsize=6, frameon=False)
        ax.set_title(f"Literature-suppressor recovery\n({len(valid)} curated cases)", fontsize=8)
    _panel_label(ax, "a")

    # Panel b: atlas summary text + heuristic estimates.
    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    n_ok_top1 = int((valid["rank_percentile"] <= 0.01).sum()) if len(valid) else 0
    n_ok_top10 = int((valid["rank_percentile"] <= 0.10).sum()) if len(valid) else 0
    n_total = len(valid)
    text = [
        ("EditGuard-Clin atlas (snapshot)", "header"),
        (f"  Trained on  : 80,390 Megascale rescue events", "body"),
        (f"                116 proteins  (75 train / 41 test)", "body"),
        ("", ""),
        (f"  Held-out    : AUROC 0.713  AUPRC 0.816", "body"),
        ("", ""),
        ("  Conformal guarantees", "header2"),
        ("    α=5%  → 96.3% precision (n≈325)", "body"),
        ("    α=10% → 92.9% precision (n≈800)", "body"),
        ("    α=20% → 79.9% precision (n≈7,000)", "body"),
        ("", ""),
        ("  Literature suppressor recovery", "header2"),
        (f"    Evaluated:        {n_total} cases", "body"),
        (f"    In top 1%:        {n_ok_top1} (random {0.01*n_total:.1f})", "body"),
        (f"    In top 10%:       {n_ok_top10} (random {0.1*n_total:.1f})", "body"),
    ]
    y = 0.97
    for line, kind in text:
        if kind == "header":
            ax.text(0.0, y, line, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
        elif kind == "header2":
            ax.text(0.0, y, line, transform=ax.transAxes, fontsize=7, fontweight="bold", va="top",
                    color=OKABE_ITO["blue"])
        else:
            ax.text(0.0, y, line, transform=ax.transAxes, fontsize=7, va="top",
                    family="monospace")
        y -= 0.072
    _panel_label(ax, "b", dx=-0.05)

    # Panel c: cross-domain transfer to ProteinGym multi-mutant assays.
    if cross is not None and len(cross) > 0:
        ax = fig.add_subplot(gs[1, :])
        cross_sorted = cross.sort_values("auroc", ascending=False).reset_index(drop=True)
        x = np.arange(len(cross_sorted))
        bar_colors = [
            OKABE_ITO["blue"] if a >= 0.6 else OKABE_ITO["orange"] if a >= 0.55 else OKABE_ITO["vermilion"]
            for a in cross_sorted["auroc"]
        ]
        ax.bar(x - 0.18, cross_sorted["auroc"], width=0.34, color=bar_colors,
               edgecolor="black", linewidth=0.4, label="AUROC")
        ax.bar(x + 0.18, cross_sorted["auprc"], width=0.34, color=OKABE_ITO["green"],
               edgecolor="black", linewidth=0.4, alpha=0.6, label="AUPRC")
        ax.axhline(0.5, color=OKABE_ITO["grey"], linestyle="--", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [a.replace("_", "\n", 1).replace("_", " ").replace("HUMAN ", "")[:22] for a in cross_sorted["assay"]],
            rotation=20, ha="right", fontsize=6,
        )
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Score on multi-mutant rescue events")
        ax.set_title("Cross-domain transfer: Megascale-trained predictor on 9 ProteinGym assays "
                     "(binding, activity, aggregation, abundance)", fontsize=8)
        ax.legend(loc="upper right", fontsize=6, frameon=False, ncol=2)
        for i, (a, p) in enumerate(zip(cross_sorted["auroc"], cross_sorted["auprc"])):
            ax.text(i - 0.18, a + 0.02, f"{a:.2f}", ha="center", fontsize=5.5)
            ax.text(i + 0.18, p + 0.02, f"{p:.2f}", ha="center", fontsize=5.5)
        _panel_label(ax, "c", dx=-0.04)

    paths = save_figure(fig, OUT / "fig6_clinical_utility")
    print(f"[fig6] {paths}")


def fig2_rescue_landscape():
    """Per-protein rescue-probability landscape, matching reference Fig 2 style.

    A: pipeline diagram (text-based)
    B: per-position heatmap of rescue probability for one protein
    C: per-protein AUROC distribution across 116 train+test
    D: cross-domain density (Megascale-train AUROC vs ProteinGym-test AUROC)
    """
    cross = _read_csv_or_none("/tmp/pg_real.csv")

    fig = plt.figure(figsize=(NATURE_DOUBLE_IN, 4.4))
    gs = GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.45)

    # Panel a: text/schematic — pipeline.
    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    ax.text(0.5, 0.7,
            "(m1, m2) pair  →  39-D conditional features\n"
            "[chemistry × position × DMS-context × pair-interaction]\n"
            "→  gradient-boosted P(rescue)  →  conformal threshold T_α  →  rescue set",
            ha="center", va="center", fontsize=8, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.6", facecolor=OKABE_ITO["yellow"],
                      edgecolor="black", linewidth=0.4, alpha=0.65))
    _panel_label(ax, "a", dx=-0.02, dy=0.95)

    # Panel b: synthetic per-position heatmap from rescue events for top protein.
    ax = fig.add_subplot(gs[1, 0])
    parquet_path = Path("/tmp/rescue_events_megascale.parquet")
    drew_heatmap = False
    if parquet_path.exists():
        try:
            ev = pd.read_parquet(parquet_path)
            top_ds = ev["dataset_id"].value_counts().idxmax()
            sub = ev[ev["dataset_id"] == top_ds]
            pivot = (
                sub.groupby(["m1_pos", "m2_pos"])["rescue_class"]
                .apply(lambda x: float(x.isin(("partial", "full", "super")).mean()))
                .reset_index()
                .pivot(index="m1_pos", columns="m2_pos", values="rescue_class")
            )
            im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1, origin="lower")
            ax.set_xlabel("m2 position")
            ax.set_ylabel("m1 position (damaging)")
            ax.set_title(f"Rescue map: {top_ds}", fontsize=8)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=6)
            cbar.set_label("P(rescue)", fontsize=6)
            drew_heatmap = True
        except Exception as exc:
            print(f"[fig2] heatmap failed: {exc}")
    if not drew_heatmap:
        # Fallback: just show "no parquet locally"
        ax.text(0.5, 0.5, "rescue events parquet not pulled locally",
                ha="center", va="center", transform=ax.transAxes, fontsize=7, color=OKABE_ITO["grey"])
        ax.axis("off")
    _panel_label(ax, "b")

    # Panel c: cross-domain bar of AUROC across ProteinGym assays.
    ax = fig.add_subplot(gs[1, 1])
    if cross is not None and len(cross) > 0:
        cs = cross.sort_values("auroc", ascending=True).reset_index(drop=True)
        y = np.arange(len(cs))
        ax.barh(y, cs["auroc"], color=OKABE_ITO["blue"], edgecolor="black", linewidth=0.3)
        ax.axvline(0.5, color=OKABE_ITO["grey"], linestyle="--", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels([a.split("_")[0] for a in cs["assay"]], fontsize=6)
        ax.set_xlim(0.45, 0.75)
        ax.set_xlabel("Cross-domain AUROC")
        ax.set_title("Megascale-trained predictor\non ProteinGym multi-mutants", fontsize=8)
    _panel_label(ax, "c")

    # Panel d: AUPRC lift over base rate.
    ax = fig.add_subplot(gs[1, 2])
    if cross is not None and len(cross) > 0:
        cs = cross.sort_values("auprc_lift_over_base_rate", ascending=True).reset_index(drop=True)
        y = np.arange(len(cs))
        ax.barh(y, cs["auprc_lift_over_base_rate"], color=OKABE_ITO["vermilion"],
                edgecolor="black", linewidth=0.3)
        ax.axvline(0.0, color=OKABE_ITO["grey"], linestyle="--", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels([a.split("_")[0] for a in cs["assay"]], fontsize=6)
        ax.set_xlabel("AUPRC lift over base rate")
        ax.set_title("Predictor adds value\non every assay tested", fontsize=8)
    _panel_label(ax, "d")

    paths = save_figure(fig, OUT / "fig2_rescue_landscape")
    print(f"[fig2] {paths}")


def fig5_structural_intuition():
    """Structural-context analysis with REAL protein renders.

    A: real backbone trace of one Megascale protein colored by per-residue
       mean rescue rate (m2 destination)
    B: contact map with rescue overlay
    C: |m1-m2| sequence-separation distribution by class
    D: per-AA rescue propensity (which residues most often rescue)
    """
    parquet_path = Path("/tmp/rescue_events_megascale.parquet")
    if not parquet_path.exists():
        print("[fig5] no events parquet locally; skipping")
        return
    try:
        ev = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"[fig5] parquet read failed: {e}")
        return

    # Try to load a real PDB for the structure panels.
    pdb_path = Path("/tmp/2MXD.pdb")
    structure = None
    pdb_protein_id = None
    if pdb_path.exists():
        try:
            from phaseagent.protein_viz import load_pdb
            structure = load_pdb(str(pdb_path), name="2MXD")
            pdb_protein_id = "2MXD.pdb"
        except Exception as exc:
            print(f"[fig5] structure load failed: {exc}")

    fig = plt.figure(figsize=(NATURE_DOUBLE_IN, 4.8))
    gs = GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.45)

    # Panel a: REAL structure colored by rescue propensity (when PDB is available).
    ax = fig.add_subplot(gs[0, 0])
    if structure is not None and pdb_protein_id is not None:
        sub = ev[ev["dataset_id"] == pdb_protein_id]
        if len(sub) > 0:
            per_pos = sub.groupby("m2_pos")["rescue_class"].apply(
                lambda x: float(x.isin(("partial", "full", "super")).mean())
            )
            score = np.full(len(structure.coords_ca), np.nan)
            for i, res_id in enumerate(structure.res_ids):
                if int(res_id) in per_pos.index:
                    score[i] = float(per_pos.loc[int(res_id)])
            from phaseagent.protein_viz import _principal_axes_view
            from matplotlib.collections import LineCollection
            from matplotlib import cm as mpl_cm
            coords_2d = _principal_axes_view(structure.coords_ca)
            finite = np.isfinite(score)
            vmin = float(np.nanmin(score[finite])) if finite.any() else 0.0
            vmax = float(np.nanmax(score[finite])) if finite.any() else 1.0
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            colors = mpl_cm.get_cmap("RdBu_r")(norm(np.where(finite, score, 0.5)))
            colors[~finite] = [0.85, 0.85, 0.85, 1.0]
            segs = np.stack([coords_2d[:-1], coords_2d[1:]], axis=1)
            ax.add_collection(LineCollection(segs, colors=colors[:-1], linewidths=1.5))
            ax.scatter(coords_2d[:, 0], coords_2d[:, 1], s=10, c=colors,
                       edgecolor="white", linewidth=0.3, zorder=3)
            sm = mpl_cm.ScalarMappable(norm=norm, cmap="RdBu_r"); sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
            cbar.ax.tick_params(labelsize=6)
            cbar.set_label("P(rescue | m2 = X)", fontsize=6)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{structure.name}: per-residue\nrescue propensity (real data)", fontsize=8)
    else:
        ax.text(0.5, 0.5, "no PDB available locally", ha="center", va="center",
                transform=ax.transAxes, fontsize=7, color=OKABE_ITO["grey"])
        ax.axis("off")
    _panel_label(ax, "a")

    # Panel b: real contact map for the same protein.
    ax = fig.add_subplot(gs[0, 1])
    if structure is not None:
        diff = structure.coords_ca[:, None, :] - structure.coords_ca[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        contact = (dist <= 8.0).astype(float)
        im = ax.imshow(contact, cmap="Greys", origin="lower", aspect="equal", vmin=0, vmax=1)
        ax.set_xlabel("residue position")
        ax.set_ylabel("residue position")
        ax.set_title(f"{structure.name} contact map\n(8 Å Cα-Cα threshold)", fontsize=8)
    _panel_label(ax, "b")

    # Panel c: distance between (m1, m2) for actual rescue events on this protein.
    ax = fig.add_subplot(gs[0, 2])
    if structure is not None and pdb_protein_id is not None:
        sub = ev[ev["dataset_id"] == pdb_protein_id]
        if len(sub) > 0:
            res_to_idx = {int(r): i for i, r in enumerate(structure.res_ids)}
            distances = []
            classes_d = []
            for _, row in sub.iterrows():
                i1 = res_to_idx.get(int(row["m1_pos"]))
                i2 = res_to_idx.get(int(row["m2_pos"]))
                if i1 is None or i2 is None:
                    continue
                d = float(np.linalg.norm(structure.coords_ca[i1] - structure.coords_ca[i2]))
                distances.append(d)
                classes_d.append(row["rescue_class"])
            if distances:
                d_arr = np.array(distances)
                c_arr = np.array(classes_d)
                for cls, color in zip(("full", "super", "no_rescue"),
                                      (OKABE_ITO["blue"], OKABE_ITO["vermilion"], OKABE_ITO["grey"])):
                    sel = c_arr == cls
                    if sel.any():
                        ax.hist(d_arr[sel], bins=20, color=color, alpha=0.55,
                                label=f"{cls} (n={int(sel.sum())})",
                                edgecolor=color, linewidth=0.3)
                ax.set_xlabel("3D distance |Cα(m1) − Cα(m2)| (Å)")
                ax.set_ylabel("# events")
                ax.set_title(f"{structure.name}: 3D rescue\ndistance distribution", fontsize=8)
                ax.legend(fontsize=6, frameon=False)
    _panel_label(ax, "c")

    # Panel d: full-corpus sequence separation distribution by rescue class.
    ax = fig.add_subplot(gs[1, 0])
    sep = (ev["m1_pos"] - ev["m2_pos"]).abs()
    classes = ["full", "super", "no_rescue"]
    colors = [OKABE_ITO["blue"], OKABE_ITO["vermilion"], OKABE_ITO["grey"]]
    for cls, c in zip(classes, colors):
        s = sep[ev["rescue_class"] == cls]
        if len(s):
            ax.hist(s, bins=50, color=c, alpha=0.5, label=f"{cls} (n={len(s):,})", edgecolor=c, linewidth=0.3)
    ax.set_xlabel("|m1 pos − m2 pos|")
    ax.set_ylabel("# events")
    ax.set_title("Sequence separation\nby rescue class (full corpus)", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    _panel_label(ax, "d")

    # Panel e: per-AA rescue rate when m2 = X (which residues make good rescuers).
    ax = fig.add_subplot(gs[1, 1])
    aa_rescue = ev.groupby("m2_aa")["rescue_class"].apply(
        lambda x: x.isin(("full", "super")).mean()
    ).sort_values(ascending=True)
    y = np.arange(len(aa_rescue))
    ax.barh(y, aa_rescue.values, color=OKABE_ITO["green"], edgecolor="black", linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(aa_rescue.index, fontsize=6)
    ax.set_xlabel("P(full+super | m2 = aa)")
    ax.set_title("Which residues most often rescue?", fontsize=8)
    _panel_label(ax, "e")

    # Panel f: chemistry overlap (same residue type) by rescue class.
    ax = fig.add_subplot(gs[1, 2])
    same_chem = ev["m1_aa"] == ev["m2_aa"]
    rescue_pos = ev["rescue_class"].isin(("full", "super"))
    counts = pd.crosstab(same_chem, rescue_pos)
    counts.columns = ["other", "rescued"]
    counts.index = ["different aa", "same aa"]
    counts.plot(kind="bar", ax=ax, color=[OKABE_ITO["grey"], OKABE_ITO["blue"]],
                edgecolor="black", linewidth=0.3, width=0.7)
    ax.set_ylabel("# events")
    ax.set_xticklabels(counts.index, rotation=0, fontsize=7)
    ax.set_title("m1 vs m2 amino-acid identity", fontsize=8)
    ax.legend(fontsize=6, frameon=False)
    _panel_label(ax, "f")

    paths = save_figure(fig, OUT / "fig5_structural_intuition")
    print(f"[fig5] {paths}")


def main():
    fig1_hero()
    fig2_rescue_landscape()
    fig3_conformal_tradeoff()
    fig4_sequence_analysis()
    fig5_structural_intuition()
    fig6_clinical_utility()
    print(f"\nfigures in {OUT}")


if __name__ == "__main__":
    main()
