"""Matplotlib figure generators."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import shell_fitness_ruggedness_proxy


def _save(fig, out_path):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_phase_curves(example_results: dict[str, pd.DataFrame], out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for ds_id, v in example_results.items():
        ax.plot(v["mutation_distance"], v["viability_density"], marker="o", label=ds_id)
    ax.set_xlabel("Mutation distance d")
    ax.set_ylabel("Viability density V(d)")
    ax.set_title("Functional viability vs mutation distance")
    ax.legend(fontsize=8, frameon=False)
    _save(fig, out_path)


def plot_phase_atlas(atlas_table: pd.DataFrame, out_path):
    pivot = atlas_table.pivot_table(
        index="dataset_id",
        columns="mutation_distance",
        values="viability_density",
    )
    if "dc" in atlas_table.columns:
        order = atlas_table.groupby("dataset_id")["dc"].first().sort_values().index
        pivot = pivot.reindex(order)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(pivot))))
    im = ax.imshow(
        pivot.values, aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="lower"
    )
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_xlabel("Mutation distance")
    ax.set_ylabel("Dataset (sorted by dc)")
    fig.colorbar(im, ax=ax, label="V(d)")
    _save(fig, out_path)


def plot_dc_alpha_distributions(boundary_table: pd.DataFrame, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].hist(boundary_table["dc"].dropna(), bins=20, color="steelblue")
    axes[0].set_xlabel("Critical radius dc")
    axes[0].set_ylabel("count")
    axes[1].hist(boundary_table["alpha"].dropna(), bins=20, color="coral")
    axes[1].set_xlabel("Sharpness alpha")
    fig.tight_layout()
    _save(fig, out_path)


def plot_boundary_error(sim_results: pd.DataFrame, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for policy, sub in sim_results.groupby("policy"):
        agg = sub.groupby("n_queries")["dc_error"].agg(["mean", "sem"]).reset_index()
        ax.plot(agg["n_queries"], agg["mean"], label=policy, marker="o")
        ax.fill_between(
            agg["n_queries"],
            agg["mean"] - agg["sem"],
            agg["mean"] + agg["sem"],
            alpha=0.2,
        )
    ax.set_xlabel("Oracle queries")
    ax.set_ylabel("|dc_hat - dc_true|")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out_path)


def plot_query_efficiency(sim_results: pd.DataFrame, out_path):
    plot_boundary_error(sim_results, out_path)


def plot_phase_aware_frontier(search_results: pd.DataFrame, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for policy, sub in search_results.groupby("policy"):
        ax.scatter(
            sub["mean_mutation_distance"],
            sub["functional_hit_rate"],
            label=policy,
            alpha=0.6,
        )
    ax.set_xlabel("Mean mutation distance (novelty)")
    ax.set_ylabel("Functional hit rate")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out_path)


def plot_epistasis_vs_sharpness(
    boundary_table: pd.DataFrame,
    epistasis_table: pd.DataFrame,
    out_path,
):
    fig, ax = plt.subplots(figsize=(5, 4))
    if "ruggedness" in epistasis_table.columns:
        prox = epistasis_table.groupby("dataset_id")["ruggedness"].mean().reset_index()
        col = "ruggedness"
    else:
        prox = (
            epistasis_table.groupby("dataset_id")["mean_abs_epistasis"].mean().reset_index()
        )
        col = "mean_abs_epistasis"
    merged = prox.merge(boundary_table[["dataset_id", "alpha"]], on="dataset_id")
    ax.scatter(merged[col], merged["alpha"], color="darkgreen")
    ax.set_xlabel(f"Epistasis / ruggedness proxy ({col})")
    ax.set_ylabel("Transition sharpness alpha")
    _save(fig, out_path)


def plot_failure_examples(df: pd.DataFrame, dc: float, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["mutation_distance"], df["fitness_norm"], alpha=0.3, s=8)
    ax.axvline(dc, color="red", linestyle="--", label=f"dc={dc:.2f}")
    ax.set_xlabel("Mutation distance")
    ax.set_ylabel("Normalized fitness")
    ax.legend(frameon=False)
    _save(fig, out_path)


def plot_boundary_taxonomy_examples(
    per_dataset_v: dict[str, pd.DataFrame],
    boundaries: pd.DataFrame,
    out_path,
):
    examples = {}
    for regime, sub in boundaries.groupby("regime"):
        ds = sub.iloc[0]["dataset_id"]
        examples[regime] = (ds, per_dataset_v.get(ds))
    if not examples:
        return
    fig, axes = plt.subplots(
        1, max(1, len(examples)), figsize=(4 * len(examples), 3.2), squeeze=False
    )
    for ax, (regime, (ds, v)) in zip(axes.flat, examples.items()):
        if v is None:
            ax.set_visible(False)
            continue
        ax.plot(v["mutation_distance"], v["viability_density"], "-o")
        ax.set_title(f"{regime}\n{ds}", fontsize=9)
        ax.set_xlabel("d")
        ax.set_ylabel("V(d)")
    fig.tight_layout()
    _save(fig, out_path)


def plot_plm_vs_fitness(plm_df: pd.DataFrame, out_path):
    if plm_df.empty or "plm_score" not in plm_df.columns:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ds_ids = plm_df["dataset_id"].unique()
    for ds in ds_ids[:8]:
        sub = plm_df[plm_df["dataset_id"] == ds]
        ax.scatter(sub["plm_score"], sub["fitness_norm"], s=4, alpha=0.4, label=ds)
    ax.set_xlabel("ESM-2 mean log P(token)")
    ax.set_ylabel("Normalized fitness")
    ax.legend(fontsize=7, frameon=False)
    _save(fig, out_path)


def make_all_figures(tables_dir: Path, figures_dir: Path, data_path: Optional[Path] = None) -> int:
    tables_dir = Path(tables_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    boundaries = pd.read_csv(tables_dir / "phase_boundaries.csv")
    v_by_d = pd.read_csv(tables_dir / "viability_by_distance.csv")
    n_made = 0

    examples = {ds: g for ds, g in v_by_d.groupby("dataset_id")}
    plot_phase_curves(dict(list(examples.items())[:6]), figures_dir / "fig1_phase_transition_curves")
    n_made += 1

    atlas = v_by_d.merge(
        boundaries[["dataset_id", "dc"]].rename(columns={"dc": "dc_b"}),
        on="dataset_id",
        how="left",
    )
    atlas["dc"] = atlas["dc_b"]
    plot_phase_atlas(atlas, figures_dir / "fig2_phase_atlas_heatmap")
    n_made += 1

    sim_path = tables_dir / "phaseagent_simulation.csv"
    if sim_path.exists():
        sim = pd.read_csv(sim_path)
        plot_boundary_error(sim, figures_dir / "fig3_boundary_error_vs_queries")
        plot_query_efficiency(sim, figures_dir / "fig6_query_efficiency")
        n_made += 2

    if data_path is not None and Path(data_path).exists():
        df = pd.read_parquet(data_path)
        rugged = []
        for ds, sub in df.groupby("dataset_id"):
            r = shell_fitness_ruggedness_proxy(sub)
            r["dataset_id"] = ds
            rugged.append(r)
        rugged = pd.concat(rugged, ignore_index=True)
        plot_epistasis_vs_sharpness(boundaries, rugged, figures_dir / "fig4_epistasis_vs_sharpness")
        n_made += 1

        first_ds = df["dataset_id"].iloc[0]
        sub = df[df["dataset_id"] == first_ds]
        bdc = boundaries[boundaries["dataset_id"] == first_ds]
        if len(bdc):
            plot_failure_examples(sub, bdc.iloc[0]["dc"], figures_dir / "fig8_failure_examples")
            n_made += 1

    search_path = tables_dir / "search_benchmark.csv"
    if search_path.exists():
        sr = pd.read_csv(search_path)
        plot_phase_aware_frontier(sr, figures_dir / "fig5_phase_aware_frontier")
        n_made += 1

    plot_dc_alpha_distributions(boundaries, figures_dir / "fig6_dc_alpha_distribution")
    n_made += 1

    per_ds_v = {ds: g for ds, g in v_by_d.groupby("dataset_id")}
    plot_boundary_taxonomy_examples(per_ds_v, boundaries, figures_dir / "fig7_boundary_taxonomy_examples")
    n_made += 1

    plm_path = tables_dir / "plm_scores.csv"
    if plm_path.exists():
        plm_df = pd.read_csv(plm_path)
        plot_plm_vs_fitness(plm_df, figures_dir / "fig9_plm_vs_fitness")
        n_made += 1

    return n_made
