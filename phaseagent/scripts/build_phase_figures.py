"""Build figures for the Phase-Calibrated Steering paper.

Reads real Modal outputs from /tmp and renders publication-ready PDFs/SVGs
into outputs/phase/figures/.

Required CSVs (pulled from Modal volume to /tmp first):
  /tmp/megascale_validation.csv         -- Megascale theorem validation
  /tmp/pg_validation.csv                -- ProteinGym theorem validation
  /tmp/pg_curves.csv                    -- Per-distance V(d) curves on PG
  /tmp/phase_smc_metrics.csv            -- Phase-calibrated SMC results

Renders:
  fig1_theorem_validation.pdf  -- the headline theorem-vs-data figure
  fig2_phase_smc_pareto.pdf    -- viability/novelty Pareto across betas
  fig3_hardness.pdf            -- per-protein hardness H_g vs performance
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phaseagent.nature_figures import OKABE_ITO, save_figure, setup_nature_style  # noqa: E402

OUT = ROOT / "outputs" / "phase" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
setup_nature_style()


def fig1_theorem_validation(
    megascale_csv: Path = Path("/tmp/megascale_validation.csv"),
    pg_csv: Path = Path("/tmp/pg_validation.csv"),
    pg_curves_csv: Path = Path("/tmp/pg_curves.csv"),
):
    """Hero figure: theorem prediction vs observation on real data.

    Panels:
      A) V(1) prediction vs observation (sanity check: should be exact)
      B) V(2) prediction vs observation (Megascale doubles; deviation = epistasis)
      C) Per-protein V(d) curve overlay for a representative ProteinGym assay
      D) (d_c_pred, d_c_obs) scatter across ProteinGym multi-distance assays
    """
    import matplotlib.pyplot as plt

    if not megascale_csv.exists():
        print(f"[fig1] missing {megascale_csv}; skipping")
        return
    ms = pd.read_csv(megascale_csv).dropna(subset=["V1_pred_MC", "V1_obs", "V2_pred_MC", "V2_obs"])
    pg = pd.read_csv(pg_csv) if pg_csv.exists() else pd.DataFrame()
    curves = pd.read_csv(pg_curves_csv) if pg_curves_csv.exists() else pd.DataFrame()

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.0))
    (axA, axB), (axC, axD) = axes

    # --- Panel A: V(1) sanity check
    axA.scatter(
        ms["V1_pred_MC"], ms["V1_obs"],
        s=14, color=OKABE_ITO["blue"], alpha=0.7, edgecolor="black", linewidth=0.3,
    )
    axA.plot([0, 1], [0, 1], color=OKABE_ITO["grey"], linestyle="--", linewidth=0.6)
    r1 = float(np.corrcoef(ms["V1_pred_MC"], ms["V1_obs"])[0, 1])
    axA.set_xlim(0, 1)
    axA.set_ylim(0, 1)
    axA.set_xlabel(r"$V_{\rm pred}(1)$ from singles spectrum")
    axA.set_ylabel(r"$V_{\rm obs}(1)$ measured")
    axA.set_title(f"A) Single-mutant V(1) sanity\nr = {r1:.3f}, n = {len(ms)}", fontsize=8)

    # --- Panel B: V(2) -- the meaningful test
    axB.scatter(
        ms["V2_pred_MC"], ms["V2_obs"],
        s=16, color=OKABE_ITO["vermilion"], alpha=0.75, edgecolor="black", linewidth=0.3,
    )
    axB.plot([0, 1], [0, 1], color=OKABE_ITO["grey"], linestyle="--", linewidth=0.6,
             label="additive null")
    r2 = float(np.corrcoef(ms["V2_pred_MC"], ms["V2_obs"])[0, 1])
    median_dev = float((ms["V2_obs"] - ms["V2_pred_MC"]).median())
    sign = "negative" if median_dev < 0 else "positive"
    axB.set_xlim(0, 1)
    axB.set_ylim(0, 1)
    axB.set_xlabel(r"$V_{\rm pred}(2)$ -- additive null (theorem)")
    axB.set_ylabel(r"$V_{\rm obs}(2)$ -- observed Megascale doubles")
    axB.set_title(
        f"B) V(2): $V_{{\\rm obs}} - V_{{\\rm null}}$ measures epistasis\n"
        f"median deviation = {median_dev:+.3f} ({sign} epistasis)\n"
        f"n = {len(ms)} proteins",
        fontsize=8,
    )
    axB.legend(loc="lower right", fontsize=6, frameon=False)

    # --- Panel C: V(d) curve for a representative ProteinGym assay
    if len(curves) and len(pg):
        # Pick the assay with the largest n_total and high r2.
        if "obs_fit_r2" in pg.columns:
            chosen = pg.sort_values("n_total", ascending=False).iloc[0]
        else:
            chosen = pg.iloc[0]
        sub = curves[curves["dataset_id"] == chosen["dataset_id"]].sort_values("mutation_distance")
        # Subsample to reasonable max d for plotting.
        sub = sub[sub["mutation_distance"] <= min(15, int(sub["mutation_distance"].max()))]
        axC.plot(sub["mutation_distance"], sub["V_obs"], "o-",
                 color=OKABE_ITO["vermilion"], markeredgecolor="black",
                 markeredgewidth=0.4, label="observed")
        axC.plot(sub["mutation_distance"], sub["V_pred_MC"], "s--",
                 color=OKABE_ITO["blue"], markeredgecolor="black",
                 markeredgewidth=0.4, label="LDP predicted")
        axC.axvline(float(chosen.get("dc_obs", np.nan)),
                    color=OKABE_ITO["vermilion"], linestyle=":", linewidth=0.6, alpha=0.6)
        axC.axvline(float(chosen.get("dc_pred", np.nan)),
                    color=OKABE_ITO["blue"], linestyle=":", linewidth=0.6, alpha=0.6)
        axC.set_xlabel("mutation distance d")
        axC.set_ylabel("V(d) viability")
        axC.set_title(f"C) {chosen['dataset_id'][:24]}\nV(d) curve: theorem vs observation", fontsize=8)
        axC.set_ylim(-0.02, 1.02)
        axC.legend(loc="best", fontsize=6, frameon=False)
    else:
        axC.text(0.5, 0.5, "(curves CSV missing)", transform=axC.transAxes, ha="center")

    # --- Panel D: dc_pred vs dc_obs across ProteinGym assays
    if len(pg) and "dc_pred" in pg.columns and "dc_obs" in pg.columns:
        valid = pg.dropna(subset=["dc_pred", "dc_obs"])
        axD.scatter(valid["dc_pred"], valid["dc_obs"], s=24,
                    color=OKABE_ITO["green"], alpha=0.85, edgecolor="black", linewidth=0.3)
        lim_max = float(max(valid["dc_pred"].max(), valid["dc_obs"].max()) * 1.1)
        axD.plot([0, lim_max], [0, lim_max], color=OKABE_ITO["grey"], linestyle="--", linewidth=0.6)
        if len(valid) >= 3:
            r_dc = float(np.corrcoef(valid["dc_pred"], valid["dc_obs"])[0, 1])
        else:
            r_dc = float("nan")
        axD.set_xlabel(r"$d_c^{\rm pred}$ from singles (theorem)")
        axD.set_ylabel(r"$d_c^{\rm obs}$ from sigmoid fit")
        axD.set_title(f"D) Phase-boundary location across ProteinGym\nr = {r_dc:.3f}, n = {len(valid)}",
                      fontsize=8)
        axD.set_xlim(0, lim_max)
        axD.set_ylim(0, lim_max)
    else:
        axD.text(0.5, 0.5, "(PG CSV missing)", transform=axD.transAxes, ha="center")

    fig.tight_layout()
    save_figure(fig, OUT / "fig1_theorem_validation")
    print(f"[fig1] -> {OUT / 'fig1_theorem_validation.pdf'}")


def fig2_phase_smc_pareto(
    smc_csv: Path = Path("/tmp/phase_smc_metrics.csv"),
):
    """Viability/novelty Pareto across phase betas.

    The phase-calibrated SMC at beta=0.0 reproduces the standard Twisted
    SMC baseline; larger beta increasingly weights the phase prior.
    """
    import matplotlib.pyplot as plt

    if not smc_csv.exists():
        print(f"[fig2] missing {smc_csv}; skipping")
        return
    df = pd.read_csv(smc_csv)
    if not len(df):
        print("[fig2] no rows in smc CSV; skipping")
        return

    # Aggregate per (phase_beta, dataset_id, objective) by best across seeds.
    df["best_total_reward"] = pd.to_numeric(df["best_total_reward"], errors="coerce")
    df["best_base_reward"] = pd.to_numeric(df["best_base_reward"], errors="coerce")
    df["best_hamming"] = pd.to_numeric(df["best_hamming"], errors="coerce")
    grp = df.groupby(["phase_beta", "dataset_id", "objective", "edit_budget"]).agg(
        best_base_reward=("best_base_reward", "max"),
        mean_hamming=("best_hamming", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    betas = sorted(grp["phase_beta"].unique())
    palette = {
        0.0: OKABE_ITO["grey"],
        0.5: OKABE_ITO["skyblue"],
        1.0: OKABE_ITO["blue"],
        2.0: OKABE_ITO["vermilion"],
    }
    for beta in betas:
        sub = grp[grp["phase_beta"] == beta]
        # Mean over the slice
        x = float(sub["mean_hamming"].mean())
        y = float(sub["best_base_reward"].mean())
        x_err = float(sub["mean_hamming"].std() / max(1, np.sqrt(len(sub))))
        y_err = float(sub["best_base_reward"].std() / max(1, np.sqrt(len(sub))))
        ax.errorbar(x, y, xerr=x_err, yerr=y_err,
                    fmt="o", color=palette.get(beta, OKABE_ITO["green"]),
                    markersize=8, markeredgecolor="black", markeredgewidth=0.5,
                    capsize=3, label=fr"$\beta_{{\rm phase}} = {beta}$")
    ax.set_xlabel("mean Hamming distance to WT (novelty)")
    ax.set_ylabel("base log $\\phi(x)$ reward (viability)")
    ax.set_title("Phase calibration: novelty/viability Pareto", fontsize=8)
    ax.legend(loc="best", fontsize=6, frameon=False)
    fig.tight_layout()
    save_figure(fig, OUT / "fig2_phase_smc_pareto")
    print(f"[fig2] -> {OUT / 'fig2_phase_smc_pareto.pdf'}")


def fig3_hardness(
    megascale_csv: Path = Path("/tmp/megascale_validation.csv"),
):
    """Per-protein hardness H_g and its predictive power."""
    import matplotlib.pyplot as plt

    if not megascale_csv.exists():
        print(f"[fig3] missing; skipping")
        return
    df = pd.read_csv(megascale_csv).dropna(subset=["hardness_H", "V2_obs", "V2_pred_MC"])
    if not len(df):
        return
    # Hardness vs |V2_obs - V2_pred|: high hardness should mean smaller deviation
    df["epistasis_signal"] = df["V2_obs"] - df["V2_pred_MC"]

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))
    axA, axB = axes
    axA.hist(df["hardness_H"], bins=24, color=OKABE_ITO["skyblue"], edgecolor="black", linewidth=0.3)
    axA.set_xlabel(r"hardness $H_g = \sigma_g / m_g$")
    axA.set_ylabel("# Megascale proteins")
    axA.set_title(f"A) per-protein hardness distribution\nmedian H = {df['hardness_H'].median():.2f}", fontsize=8)

    axB.scatter(df["hardness_H"], df["epistasis_signal"],
                s=14, color=OKABE_ITO["vermilion"], alpha=0.7, edgecolor="black", linewidth=0.3)
    r = float(np.corrcoef(df["hardness_H"], df["epistasis_signal"])[0, 1])
    axB.axhline(0, color=OKABE_ITO["grey"], linestyle="--", linewidth=0.6)
    axB.set_xlabel(r"hardness $H_g$")
    axB.set_ylabel(r"$V_{\rm obs}(2) - V_{\rm pred}(2)$ (epistasis)")
    axB.set_title(f"B) hardness vs epistasis signal\nr = {r:.3f}", fontsize=8)

    fig.tight_layout()
    save_figure(fig, OUT / "fig3_hardness")
    print(f"[fig3] -> {OUT / 'fig3_hardness.pdf'}")


if __name__ == "__main__":
    fig1_theorem_validation()
    fig3_hardness()
    fig2_phase_smc_pareto()
    print("done")
