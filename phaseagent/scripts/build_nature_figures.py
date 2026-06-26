"""Nature-publication-quality figure suite for the PhaseSMC paper.

Each panel saved as a separate JPG (300 dpi) with naming
``fig<N>_<L>.jpg`` (e.g. ``fig1_A.jpg``). Companion PDFs written
alongside each JPG for journal submission.

Conventions:
  * Single-column width 89mm = 3.5in; double-column 183mm = 7.2in.
  * Sans-serif (Arial / Helvetica fallback).
  * Okabe-Ito colorblind-safe palette.
  * 300 dpi minimum; lines 0.6-1.2pt; axes off-the-top-and-right.
  * No shadows, no 3D effects, no decorative grids.
  * Per-panel sample sizes annotated in lower-right.

Required CSVs at /tmp/:
  - megascale_validation.csv   (153 proteins, V(1)/V(2) + spectrum)
  - pg_validation.csv          (6 ProteinGym multi-distance assays)
  - pg_curves.csv              (per-d V(d) curves on those assays)
  - phase_smc_metrics.csv      (128 phase-cal SMC runs)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ============================================================================
# Nature publication style
# ============================================================================

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "patch.linewidth": 0.4,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OKABE = {
    "black":     "#000000",
    "orange":    "#E69F00",
    "skyblue":   "#56B4E9",
    "green":     "#009E73",
    "yellow":    "#F0E442",
    "blue":      "#0072B2",
    "vermilion": "#D55E00",
    "purple":    "#CC79A7",
    "grey":      "#999999",
    "lightgrey": "#CCCCCC",
}

# Output directory
OUT = ROOT / "outputs" / "phase" / "figures_nature"
OUT.mkdir(parents=True, exist_ok=True)

W_S = 3.5   # single column (89mm)
W_D = 7.2   # double column (183mm)
H_S = 2.6   # default panel height for single panel single column

def save_panel(fig, name: str, also_pdf: bool = True):
    """Save panel as 300dpi JPG (and matching PDF for archival)."""
    jpg = OUT / f"{name}.jpg"
    fig.savefig(jpg, dpi=300, format="jpg", bbox_inches="tight", pad_inches=0.04)
    if also_pdf:
        pdf = OUT / f"{name}.pdf"
        fig.savefig(pdf, format="pdf", bbox_inches="tight", pad_inches=0.04)
    print(f"  [saved] {name} -> {jpg.name}")
    plt.close(fig)


def annotate_n(ax, n, loc="lower right"):
    """Stamp the sample size in a corner of the axis."""
    pos = {
        "lower right": (0.96, 0.05, "right", "bottom"),
        "lower left":  (0.04, 0.05, "left", "bottom"),
        "upper right": (0.96, 0.95, "right", "top"),
        "upper left":  (0.04, 0.95, "left", "top"),
    }[loc]
    ax.text(pos[0], pos[1], f"n = {n}", transform=ax.transAxes,
            ha=pos[2], va=pos[3], fontsize=5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5))


# ============================================================================
# Data loading
# ============================================================================

ms_df = pd.read_csv("/tmp/megascale_validation.csv")
pg_df = pd.read_csv("/tmp/pg_validation.csv")
pg_curves = pd.read_csv("/tmp/pg_curves.csv")
smc_df = pd.read_csv("/tmp/phase_smc_metrics.csv")

# ============================================================================
# Figure 1 — Theorem validation
# ============================================================================

def fig1_A():
    """V(1) sanity check: predicted vs observed across 153 Megascale proteins."""
    fig, ax = plt.subplots(figsize=(W_S * 0.55, W_S * 0.55))
    df = ms_df.dropna(subset=["V1_pred_MC", "V1_obs"])
    ax.scatter(df["V1_pred_MC"], df["V1_obs"],
               s=10, color=OKABE["blue"], alpha=0.65,
               edgecolor="black", linewidth=0.25)
    ax.plot([0, 1], [0, 1], color=OKABE["grey"], linestyle="--", linewidth=0.5)
    r = float(np.corrcoef(df["V1_pred_MC"], df["V1_obs"])[0, 1])
    ax.text(0.04, 0.96, f"r = {r:.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=6, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
    annotate_n(ax, len(df))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(r"$V_{\mathrm{pred}}(1)$ — singles spectrum")
    ax.set_ylabel(r"$V_{\mathrm{obs}}(1)$ — measured")
    ax.set_aspect("equal")
    save_panel(fig, "fig1_A")


def fig1_B():
    """V(2) vs additive null: epistasis residual on Megascale doubles."""
    fig, ax = plt.subplots(figsize=(W_S * 0.55, W_S * 0.55))
    df = ms_df.dropna(subset=["V2_pred_MC", "V2_obs"])
    ax.scatter(df["V2_pred_MC"], df["V2_obs"],
               s=12, color=OKABE["vermilion"], alpha=0.7,
               edgecolor="black", linewidth=0.25)
    ax.plot([0, 1], [0, 1], color=OKABE["grey"], linestyle="--", linewidth=0.5,
            label="additive null")
    r = float(np.corrcoef(df["V2_pred_MC"], df["V2_obs"])[0, 1])
    md = float((df["V2_obs"] - df["V2_pred_MC"]).median())
    ax.text(0.04, 0.96, f"r = {r:.2f}\nΔ = {md:+.2f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=6, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
    annotate_n(ax, len(df))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(r"$V_{\mathrm{pred}}(2)$ — additive null")
    ax.set_ylabel(r"$V_{\mathrm{obs}}(2)$ — Megascale doubles")
    ax.legend(loc="lower right", fontsize=5, frameon=False, handlelength=1.5)
    ax.set_aspect("equal")
    save_panel(fig, "fig1_B")


def fig1_C():
    """V(d) curve overlay: theorem prediction vs observed for one assay."""
    if not len(pg_curves):
        return
    # Pick the assay with the most distance points
    chosen_id = pg_curves.groupby("dataset_id")["mutation_distance"].max().idxmax()
    sub = pg_curves[pg_curves["dataset_id"] == chosen_id].sort_values("mutation_distance")
    sub = sub[sub["mutation_distance"] <= 15]
    chosen_meta = pg_df[pg_df["dataset_id"] == chosen_id].iloc[0] if (pg_df["dataset_id"] == chosen_id).any() else None

    fig, ax = plt.subplots(figsize=(W_S * 0.55, W_S * 0.55))
    ax.plot(sub["mutation_distance"], sub["V_obs"], "o-",
            color=OKABE["vermilion"], markersize=3.5, markeredgewidth=0.3,
            markeredgecolor="black", linewidth=1.0, label="observed")
    ax.plot(sub["mutation_distance"], sub["V_pred_MC"], "s--",
            color=OKABE["blue"], markersize=3, markeredgewidth=0.3,
            markeredgecolor="black", linewidth=1.0, label="theorem (singles only)")
    if chosen_meta is not None:
        if np.isfinite(chosen_meta.get("dc_obs", np.nan)):
            ax.axvline(chosen_meta["dc_obs"], color=OKABE["vermilion"],
                       linestyle=":", linewidth=0.5, alpha=0.6)
        if np.isfinite(chosen_meta.get("dc_pred", np.nan)):
            ax.axvline(chosen_meta["dc_pred"], color=OKABE["blue"],
                       linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_xlabel("edit distance d")
    ax.set_ylabel("V(d) viability")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="best", fontsize=5, frameon=False)
    short_id = chosen_id[:14]
    ax.text(0.04, 0.96, short_id, transform=ax.transAxes,
            ha="left", va="top", fontsize=5.5, style="italic")
    save_panel(fig, "fig1_C")


def fig1_D():
    """d_c_pred vs d_c_obs scatter across ProteinGym multi-distance assays."""
    df = pg_df.dropna(subset=["dc_pred", "dc_obs"])
    fig, ax = plt.subplots(figsize=(W_S * 0.55, W_S * 0.55))
    lim = max(df["dc_pred"].max(), df["dc_obs"].max()) * 1.1
    ax.scatter(df["dc_pred"], df["dc_obs"],
               s=24, color=OKABE["green"], alpha=0.85,
               edgecolor="black", linewidth=0.4)
    ax.plot([0, lim], [0, lim], color=OKABE["grey"], linestyle="--", linewidth=0.5)
    r = float(np.corrcoef(df["dc_pred"], df["dc_obs"])[0, 1])
    ax.text(0.04, 0.96, f"r = {r:.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=6, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
    annotate_n(ax, len(df))
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(r"$d_c^{\mathrm{pred}}$ — theorem")
    ax.set_ylabel(r"$d_c^{\mathrm{obs}}$ — sigmoid fit")
    ax.set_aspect("equal")
    save_panel(fig, "fig1_D")


# ============================================================================
# Figure 2 — Spectrum-derived parameter distributions
# ============================================================================

def fig2_A():
    """Distribution of m_g (mean damage per random mutation) across cohort."""
    df = ms_df.dropna(subset=["m_g"])
    fig, ax = plt.subplots(figsize=(W_S * 0.55, W_S * 0.5))
    ax.hist(df["m_g"], bins=24, color=OKABE["skyblue"],
            edgecolor="black", linewidth=0.3)
    ax.axvline(df["m_g"].median(), color=OKABE["vermilion"],
               linestyle="--", linewidth=0.7,
               label=f"median = {df['m_g'].median():.2f}")
    annotate_n(ax, len(df))
    ax.set_xlabel(r"mean damage $m_g$ (kcal/mol)")
    ax.set_ylabel("# proteins")
    ax.legend(loc="upper right", fontsize=5, frameon=False)
    save_panel(fig, "fig2_A")


def fig2_B():
    """Distribution of sigma_g."""
    df = ms_df.dropna(subset=["sigma_g"])
    fig, ax = plt.subplots(figsize=(W_S * 0.55, W_S * 0.5))
    ax.hist(df["sigma_g"], bins=24, color=OKABE["green"],
            edgecolor="black", linewidth=0.3)
    ax.axvline(df["sigma_g"].median(), color=OKABE["vermilion"],
               linestyle="--", linewidth=0.7,
               label=f"median = {df['sigma_g'].median():.2f}")
    annotate_n(ax, len(df))
    ax.set_xlabel(r"spectrum SD $\sigma_g$ (kcal/mol)")
    ax.set_ylabel("# proteins")
    ax.legend(loc="upper right", fontsize=5, frameon=False)
    save_panel(fig, "fig2_B")


def fig2_C():
    """Distribution of predicted phase boundary d_c_pred."""
    df = ms_df.dropna(subset=["dc_pred"])
    fig, ax = plt.subplots(figsize=(W_S * 0.55, W_S * 0.5))
    ax.hist(df["dc_pred"], bins=24, color=OKABE["blue"],
            edgecolor="black", linewidth=0.3)
    ax.axvline(df["dc_pred"].median(), color=OKABE["vermilion"],
               linestyle="--", linewidth=0.7,
               label=fr"median = {df['dc_pred'].median():.2f}")
    annotate_n(ax, len(df))
    ax.set_xlabel(r"phase boundary $d_c^{\mathrm{pred}}$")
    ax.set_ylabel("# proteins")
    ax.legend(loc="upper right", fontsize=5, frameon=False)
    save_panel(fig, "fig2_C")


def fig2_D():
    """Joint scatter: (d_c_pred, alpha_pred) coloured by hardness H_g."""
    df = ms_df.dropna(subset=["dc_pred", "alpha_pred", "hardness_H"])
    fig, ax = plt.subplots(figsize=(W_S * 0.6, W_S * 0.55))
    sc = ax.scatter(df["dc_pred"], df["alpha_pred"],
                    c=df["hardness_H"], cmap="viridis",
                    s=12, alpha=0.85, edgecolor="black", linewidth=0.2)
    cb = plt.colorbar(sc, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label(r"hardness $H_g = \sigma_g/m_g$", fontsize=6)
    cb.ax.tick_params(labelsize=5)
    annotate_n(ax, len(df))
    ax.set_xlabel(r"phase boundary $d_c^{\mathrm{pred}}$")
    ax.set_ylabel(r"sharpness $\alpha^{\mathrm{pred}}$")
    save_panel(fig, "fig2_D")


# ============================================================================
# Figure 3 — Theory illustrations: Berry-Esseen, Edgeworth, Cramer
# ============================================================================

def fig3_A():
    """Berry-Esseen rate vs d for representative proteins.

    Shows |R_g(d)| <= C_KS * rho_g / (sigma_g^3 * sqrt(d)) for several
    proteins picked across the hardness range.
    """
    df = ms_df.dropna(subset=["sigma_g", "third_abs_moment"]).copy()
    df["BE_rate_at_d"] = df["third_abs_moment"] / (df["sigma_g"] ** 3)
    pick = df.sort_values("hardness_H").iloc[[0, len(df)//4, len(df)//2,
                                                3*len(df)//4, -1]].reset_index(drop=True)
    C_KS = 0.4748

    fig, ax = plt.subplots(figsize=(W_S * 0.55, W_S * 0.55))
    d_range = np.arange(1, 21)
    cmap = mpl.cm.viridis(np.linspace(0.0, 0.9, len(pick)))
    for i, row in pick.iterrows():
        bound = C_KS * row["BE_rate_at_d"] / np.sqrt(d_range)
        ax.plot(d_range, bound, "-", linewidth=1.0, color=cmap[i],
                label=fr"$H_g$ = {row['hardness_H']:.2f}")
    ax.set_xlabel(r"edit distance $d$")
    ax.set_ylabel(r"Berry–Esseen bound $|R_g(d)|$")
    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=5, frameon=False, ncol=1)
    ax.text(0.04, 0.04, r"$|R_g(d)|\leq C_{\mathrm{KS}}\rho_g/(\sigma_g^3\sqrt{d})$",
            transform=ax.transAxes, fontsize=6, ha="left", va="bottom")
    save_panel(fig, "fig3_A")


def fig3_B():
    """Edgeworth correction term for representative skewness values.

    Shows the order-1/sqrt(d) correction (gamma_1/6 sqrt(d))(1-u^2)phi(u).
    """
    fig, ax = plt.subplots(figsize=(W_S * 0.55, W_S * 0.55))
    u = np.linspace(-3, 3, 400)
    skewnesses = [-0.5, 0.0, 0.5, 1.0]
    cmap = mpl.cm.coolwarm(np.linspace(0.15, 0.85, len(skewnesses)))
    d = 4
    for i, gamma1 in enumerate(skewnesses):
        # Edgeworth correction: -gamma_1/(6 sqrt(d)) * (1 - u^2) * phi(u)
        corr = -gamma1 / (6 * np.sqrt(d)) * (1 - u**2) * sstats.norm.pdf(u)
        ax.plot(u, corr, "-", linewidth=1.0, color=cmap[i],
                label=fr"$\gamma_1$ = {gamma1:+.1f}")
    ax.axhline(0, color=OKABE["grey"], linewidth=0.4, linestyle="--")
    ax.set_xlabel(r"standardized $u = (\Delta_g - dm_g)/(\sigma_g\sqrt{d})$")
    ax.set_ylabel(r"Edgeworth correction at $d=4$")
    ax.legend(loc="upper right", fontsize=5, frameon=False)
    save_panel(fig, "fig3_B")


def fig3_C():
    """Cramer rate function I_g(a) for representative spectra."""
    fig, ax = plt.subplots(figsize=(W_S * 0.55, W_S * 0.55))
    a_grid = np.linspace(-2.5, 2.5, 200)
    # Construct synthetic spectra with realistic moments
    # m=0.8, vary sigma in the empirical range
    spectra = [
        ("low H (sharp cliff)",  0.8, 0.6),
        ("median H",             0.8, 1.0),
        ("high H (gentle cliff)", 0.8, 1.6),
    ]
    cmap = mpl.cm.viridis(np.linspace(0.1, 0.85, len(spectra)))
    for i, (label, m, s) in enumerate(spectra):
        # Gaussian rate function: I(a) = (a-m)^2 / (2 sigma^2)
        I = np.maximum((a_grid - m) ** 2 / (2 * s ** 2), 0)
        ax.plot(a_grid, I, "-", linewidth=1.0, color=cmap[i],
                label=label)
    ax.set_xlabel(r"empirical mean $a = S_d / d$")
    ax.set_ylabel(r"Cramér rate $I_g(a)$")
    ax.legend(loc="upper center", fontsize=5, frameon=False)
    ax.text(0.96, 0.04, r"$V_g(d)\sim e^{-dI_g(\Delta_g/d)}/\sqrt{d}$",
            transform=ax.transAxes, fontsize=6, ha="right", va="bottom")
    save_panel(fig, "fig3_C")


def fig3_D():
    """Berry-Esseen vs Edgeworth vs MC ground truth on one synthetic spectrum.

    Shows that Edgeworth corrects BE in the small-d regime.
    """
    rng = np.random.default_rng(42)
    skewness = 0.7  # typical Megascale skewness magnitude
    # Skew-normal-like spectrum with realistic moments
    n = 50000
    # A simple approach: shift+scale a chi-square then standardize
    raw = rng.gamma(shape=2.0, scale=1.0, size=n) - 2.0
    Y = 0.8 + 1.0 * (raw - raw.mean()) / raw.std()  # m=0.8, sigma=1.0, positive skew
    m_g, sig_g = Y.mean(), Y.std()
    gamma1 = sstats.skew(Y)

    fig, ax = plt.subplots(figsize=(W_S * 0.55, W_S * 0.55))
    d_range = np.arange(1, 11)
    delta = 1.0  # viability gap

    V_mc = []
    V_be = []
    V_edge = []
    M = 200000
    for d in d_range:
        # MC
        draws = rng.choice(Y, size=(M, d), replace=True)
        sums = draws.sum(axis=1)
        V_mc.append((sums <= delta).mean())
        # Berry-Esseen Gaussian
        u_d = (delta - d * m_g) / (sig_g * np.sqrt(d))
        V_be.append(sstats.norm.cdf(u_d))
        # Edgeworth correction
        edge_corr = -gamma1 / (6 * np.sqrt(d)) * (1 - u_d**2) * sstats.norm.pdf(u_d)
        V_edge.append(sstats.norm.cdf(u_d) + edge_corr)

    ax.plot(d_range, V_mc, "o-", color=OKABE["vermilion"], label="MC truth",
            markersize=4, markeredgewidth=0.3, markeredgecolor="black", linewidth=1.0)
    ax.plot(d_range, V_be, "s--", color=OKABE["blue"], label="Berry–Esseen",
            markersize=3.5, markeredgewidth=0.3, markeredgecolor="black", linewidth=0.8)
    ax.plot(d_range, V_edge, "^-.", color=OKABE["green"], label="Edgeworth",
            markersize=3.5, markeredgewidth=0.3, markeredgecolor="black", linewidth=0.8)
    ax.set_xlabel("d")
    ax.set_ylabel("V(d)")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="upper right", fontsize=5, frameon=False)
    ax.text(0.04, 0.04, fr"$\gamma_1\!=\!{gamma1:.2f}$,  $\Delta_g\!=\!1$",
            transform=ax.transAxes, fontsize=6, ha="left", va="bottom")
    save_panel(fig, "fig3_D")


# ============================================================================
# Figure 4 — Phase-calibrated reward mechanism
# ============================================================================

def fig4_A():
    """Reward landscape: log V_g(d) vs d for representative proteins."""
    fig, ax = plt.subplots(figsize=(W_S * 0.55, W_S * 0.55))
    df = ms_df.dropna(subset=["dc_pred", "alpha_pred"]).copy()
    pick = df.sort_values("hardness_H").iloc[[0, len(df)//2, -1]].reset_index(drop=True)

    d_range = np.linspace(0, 8, 200)
    cmap = mpl.cm.viridis(np.linspace(0.1, 0.85, len(pick)))
    for i, row in pick.iterrows():
        # sigmoid V(d) = 1 / (1 + exp(alpha*(d-dc)))
        V_d = 1.0 / (1.0 + np.exp(row["alpha_pred"] * (d_range - row["dc_pred"])))
        log_V = np.log(np.clip(V_d, 1e-6, 1.0))
        ax.plot(d_range, log_V, "-", linewidth=1.2, color=cmap[i],
                label=fr"$d_c$ = {row['dc_pred']:.1f}, $H$ = {row['hardness_H']:.2f}")
    ax.set_xlabel("Hamming distance d(x, x₀)")
    ax.set_ylabel(r"$\log\hat{V}_g(d)$ (phase-prior reward term)")
    ax.legend(loc="lower left", fontsize=5, frameon=False)
    save_panel(fig, "fig4_A")


def fig4_B():
    """Combined reward log φ + β log V(d) at different β."""
    fig, ax = plt.subplots(figsize=(W_S * 0.55, W_S * 0.55))
    d_range = np.linspace(0, 8, 200)
    dc, alpha = 1.5, 1.5  # representative protein

    V_d = 1.0 / (1.0 + np.exp(alpha * (d_range - dc)))
    log_V = np.log(np.clip(V_d, 1e-6, 1.0))
    log_phi = -0.05 * d_range  # toy increasing-with-novelty viability proxy

    betas = [0.0, 0.5, 1.0, 2.0]
    cmap = [OKABE["grey"], OKABE["skyblue"], OKABE["blue"], OKABE["vermilion"]]
    for i, beta in enumerate(betas):
        r = log_phi + beta * log_V
        ax.plot(d_range, r, "-", linewidth=1.2, color=cmap[i],
                label=fr"$\beta_{{\mathrm{{phase}}}}$ = {beta}")
    ax.axvline(dc, color=OKABE["grey"], linestyle=":", linewidth=0.5)
    ax.text(dc, ax.get_ylim()[1] * 0.95, r"$d_c$", fontsize=6,
            ha="center", va="top")
    ax.set_xlabel("Hamming distance")
    ax.set_ylabel(r"$\log\phi(x) + \beta_{\mathrm{phase}}\log\hat{V}_g(d)$")
    ax.legend(loc="lower left", fontsize=5, frameon=False)
    save_panel(fig, "fig4_B")


def fig4_C():
    """Reward annealing schedule β_t across SMC steps."""
    fig, ax = plt.subplots(figsize=(W_S * 0.55, W_S * 0.5))
    T = 64
    t_range = np.arange(0, T + 1)
    gammas = [0.5, 1.0, 2.0]
    cmap = mpl.cm.plasma(np.linspace(0.1, 0.8, len(gammas)))
    beta_max = 1.0
    for i, gamma in enumerate(gammas):
        beta_t = beta_max * (1 - t_range / T) ** gamma
        ax.plot(t_range, beta_t, "-", linewidth=1.2, color=cmap[i],
                label=fr"$\gamma$ = {gamma}")
    ax.set_xlabel("denoising step t")
    ax.set_ylabel(r"phase weight $\beta_t$")
    ax.legend(loc="upper right", fontsize=5, frameon=False)
    save_panel(fig, "fig4_C")


# ============================================================================
# Figure 5 — Phase-calibrated SMC empirical results
# ============================================================================

def fig5_A():
    """Pareto frontier across phase betas (real SMC data, n=128 runs)."""
    df = smc_df.copy()
    df["best_base_reward"] = pd.to_numeric(df["best_base_reward"], errors="coerce")
    df["best_hamming"] = pd.to_numeric(df["best_hamming"], errors="coerce")

    fig, ax = plt.subplots(figsize=(W_S * 0.55, W_S * 0.55))
    betas = sorted(df["phase_beta"].unique())
    palette = {0.0: OKABE["grey"], 0.5: OKABE["skyblue"],
               1.0: OKABE["blue"], 2.0: OKABE["vermilion"]}
    for beta in betas:
        sub = df[df["phase_beta"] == beta]
        x = sub["best_hamming"].mean()
        y = sub["best_base_reward"].mean()
        xerr = sub["best_hamming"].sem()
        yerr = sub["best_base_reward"].sem()
        ax.errorbar(x, y, xerr=xerr, yerr=yerr,
                    fmt="o", color=palette.get(beta, OKABE["green"]),
                    markersize=7, markeredgecolor="black", markeredgewidth=0.4,
                    capsize=2.5, capthick=0.5, elinewidth=0.5,
                    label=fr"$\beta$ = {beta}")
    ax.set_xlabel("mean Hamming dist. to WT (novelty)")
    ax.set_ylabel(r"base log $\phi(x)$ reward (viability)")
    ax.legend(loc="upper right", fontsize=5, frameon=False)
    annotate_n(ax, len(df))
    save_panel(fig, "fig5_A")


def fig5_B():
    """Per-edit-budget breakdown: phase calibration helps most at large B."""
    df = smc_df.copy()
    df["best_base_reward"] = pd.to_numeric(df["best_base_reward"], errors="coerce")
    df["best_hamming"] = pd.to_numeric(df["best_hamming"], errors="coerce")
    grp = df.groupby(["edit_budget", "phase_beta"]).agg(
        mean_reward=("best_base_reward", "mean"),
        mean_ham=("best_hamming", "mean"),
        n=("best_base_reward", "count"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(W_S * 0.65, W_S * 0.5))
    budgets = sorted(grp["edit_budget"].unique())
    cmap = mpl.cm.plasma(np.linspace(0.15, 0.8, len(budgets)))
    for i, B in enumerate(budgets):
        sub = grp[grp["edit_budget"] == B].sort_values("phase_beta")
        ax.plot(sub["phase_beta"], sub["mean_reward"], "o-",
                color=cmap[i], linewidth=1.2, markersize=4,
                markeredgewidth=0.3, markeredgecolor="black",
                label=f"B = {B}")
    ax.set_xlabel(r"phase weight $\beta_{\mathrm{phase}}$")
    ax.set_ylabel("mean base log φ reward")
    ax.legend(loc="best", fontsize=5, frameon=False, title="edit budget",
              title_fontsize=5)
    save_panel(fig, "fig5_B")


def fig5_C():
    """Hamming distance vs β: mode-collapse warning at small budgets."""
    df = smc_df.copy()
    df["best_hamming"] = pd.to_numeric(df["best_hamming"], errors="coerce")
    grp = df.groupby(["edit_budget", "phase_beta"]).agg(
        mean_ham=("best_hamming", "mean"),
        sem_ham=("best_hamming", "sem"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(W_S * 0.65, W_S * 0.5))
    budgets = sorted(grp["edit_budget"].unique())
    cmap = mpl.cm.plasma(np.linspace(0.15, 0.8, len(budgets)))
    for i, B in enumerate(budgets):
        sub = grp[grp["edit_budget"] == B].sort_values("phase_beta")
        ax.errorbar(sub["phase_beta"], sub["mean_ham"], yerr=sub["sem_ham"],
                    fmt="o-", color=cmap[i], linewidth=1.2, markersize=4,
                    markeredgewidth=0.3, markeredgecolor="black",
                    capsize=2, label=f"B = {B}")
        ax.axhline(B, color=cmap[i], linestyle=":", linewidth=0.4, alpha=0.5)
    ax.set_xlabel(r"phase weight $\beta_{\mathrm{phase}}$")
    ax.set_ylabel("mean Hamming distance")
    ax.legend(loc="best", fontsize=5, frameon=False, title="edit budget",
              title_fontsize=5)
    save_panel(fig, "fig5_C")


# ============================================================================
# Figure 6 — Epistasis residual analysis
# ============================================================================

def fig6_A():
    """Distribution of V(2)_obs - V(2)_pred (epistasis signal)."""
    df = ms_df.dropna(subset=["V2_obs", "V2_pred_MC"]).copy()
    df["epistasis"] = df["V2_obs"] - df["V2_pred_MC"]

    fig, ax = plt.subplots(figsize=(W_S * 0.55, W_S * 0.5))
    ax.hist(df["epistasis"], bins=30, color=OKABE["purple"],
            edgecolor="black", linewidth=0.3)
    ax.axvline(0, color=OKABE["grey"], linestyle="--", linewidth=0.6)
    ax.axvline(df["epistasis"].median(), color=OKABE["vermilion"],
               linestyle="-", linewidth=0.8,
               label=f"median = {df['epistasis'].median():+.2f}")
    ax.set_xlabel(r"$V_{\mathrm{obs}}(2) - V_{\mathrm{null}}(2)$")
    ax.set_ylabel("# proteins")
    ax.legend(loc="upper right", fontsize=5, frameon=False)
    annotate_n(ax, len(df))
    save_panel(fig, "fig6_A")


def fig6_B():
    """Hardness vs epistasis (with regression line)."""
    df = ms_df.dropna(subset=["hardness_H", "V2_obs", "V2_pred_MC"]).copy()
    df["epistasis"] = df["V2_obs"] - df["V2_pred_MC"]

    fig, ax = plt.subplots(figsize=(W_S * 0.55, W_S * 0.55))
    ax.scatter(df["hardness_H"], df["epistasis"],
               s=10, color=OKABE["vermilion"], alpha=0.7,
               edgecolor="black", linewidth=0.25)
    # regression line
    z = np.polyfit(df["hardness_H"], df["epistasis"], 1)
    x_fit = np.linspace(df["hardness_H"].min(), df["hardness_H"].max(), 50)
    ax.plot(x_fit, z[0] * x_fit + z[1], color=OKABE["blue"], linewidth=1.0)
    r = float(np.corrcoef(df["hardness_H"], df["epistasis"])[0, 1])
    ax.axhline(0, color=OKABE["grey"], linestyle="--", linewidth=0.5)
    ax.text(0.04, 0.04, f"r = {r:.2f}", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=6, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
    annotate_n(ax, len(df), loc="upper right")
    ax.set_xlabel(r"hardness $H_g = \sigma_g/m_g$")
    ax.set_ylabel(r"$V_{\mathrm{obs}}(2) - V_{\mathrm{null}}(2)$")
    save_panel(fig, "fig6_B")


def fig6_C():
    """Per-protein scatter: phase-boundary parameters tied to epistasis."""
    df = ms_df.dropna(subset=["dc_pred", "V2_obs", "V2_pred_MC", "hardness_H"]).copy()
    df["epistasis"] = df["V2_obs"] - df["V2_pred_MC"]

    fig, ax = plt.subplots(figsize=(W_S * 0.6, W_S * 0.55))
    sc = ax.scatter(df["dc_pred"], df["epistasis"],
                    c=df["hardness_H"], cmap="viridis",
                    s=12, alpha=0.85, edgecolor="black", linewidth=0.2)
    cb = plt.colorbar(sc, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label(r"$H_g$", fontsize=6)
    cb.ax.tick_params(labelsize=5)
    ax.axhline(0, color=OKABE["grey"], linestyle="--", linewidth=0.5)
    annotate_n(ax, len(df))
    ax.set_xlabel(r"$d_c^{\mathrm{pred}}$")
    ax.set_ylabel(r"$V_{\mathrm{obs}}(2) - V_{\mathrm{null}}(2)$")
    save_panel(fig, "fig6_C")


# ============================================================================
# Figure 7 — Conformal validity & OOD
# ============================================================================

def fig7_A():
    """Conformal validity: target alpha vs realised FDP, on rescue events."""
    # The conformal table from §6 of the paper.
    alpha = np.array([0.05, 0.10, 0.20, 0.30])
    realised = np.array([0.037, 0.071, 0.201, 0.296])
    set_size = np.array([325, 803, 7019, 13666])

    fig, ax = plt.subplots(figsize=(W_S * 0.55, W_S * 0.55))
    ax.plot([0, 0.35], [0, 0.35], color=OKABE["grey"], linestyle="--",
            linewidth=0.5, label="target")
    ax.plot(alpha, realised, "o-", color=OKABE["vermilion"],
            markersize=5.5, markeredgecolor="black", markeredgewidth=0.4,
            linewidth=1.2, label="realised")
    for a, r, n in zip(alpha, realised, set_size):
        ax.text(a, r + 0.012, f"n={n:,}", fontsize=4.5,
                ha="center", va="bottom")
    ax.set_xlabel(r"target false-edit rate $\alpha$")
    ax.set_ylabel(r"realised FDP")
    ax.set_xlim(0, 0.35)
    ax.set_ylim(0, 0.35)
    ax.legend(loc="lower right", fontsize=5, frameon=False)
    ax.set_aspect("equal")
    save_panel(fig, "fig7_A")


def fig7_B():
    """Conformal acceptance set size vs alpha (log scale)."""
    alpha = np.array([0.05, 0.10, 0.20, 0.30])
    set_size = np.array([325, 803, 7019, 13666])
    n_total = 15033

    fig, ax = plt.subplots(figsize=(W_S * 0.55, W_S * 0.5))
    ax.semilogy(alpha, set_size, "o-", color=OKABE["blue"],
                markersize=5, markeredgecolor="black", markeredgewidth=0.4,
                linewidth=1.2)
    ax.axhline(n_total, color=OKABE["grey"], linestyle="--", linewidth=0.5,
               label=f"calib. size = {n_total:,}")
    ax.set_xlabel(r"target false-edit rate $\alpha$")
    ax.set_ylabel(r"$|\hat{S}_\alpha|$ accepted")
    ax.legend(loc="lower right", fontsize=5, frameon=False)
    save_panel(fig, "fig7_B")


# ============================================================================
# Master driver
# ============================================================================

def main():
    print("=" * 60)
    print("Building Nature-quality figure suite -> outputs/phase/figures_nature/")
    print("=" * 60)

    fig_funcs = [
        ("Figure 1: Theorem validation", [fig1_A, fig1_B, fig1_C, fig1_D]),
        ("Figure 2: Spectrum statistics", [fig2_A, fig2_B, fig2_C, fig2_D]),
        ("Figure 3: BE/Edgeworth/Cramer theory", [fig3_A, fig3_B, fig3_C, fig3_D]),
        ("Figure 4: Phase-calibrated reward", [fig4_A, fig4_B, fig4_C]),
        ("Figure 5: Phase-cal SMC results", [fig5_A, fig5_B, fig5_C]),
        ("Figure 6: Epistasis residual", [fig6_A, fig6_B, fig6_C]),
        ("Figure 7: Conformal validity", [fig7_A, fig7_B]),
    ]
    total = 0
    for name, funcs in fig_funcs:
        print(f"\n{name}")
        print("-" * len(name))
        for fn in funcs:
            try:
                fn()
                total += 1
            except Exception as exc:
                print(f"  [error] {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{'=' * 60}")
    print(f"Done. {total} panels written to {OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
