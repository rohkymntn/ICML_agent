"""Modal entrypoints for the v2 EditGuard pipeline (few-shot, OOD, Megascale).

This module is *additive* to ``modal_app.py``. The original ``modal_app.py``
remains the source of truth for the v1 pipeline (full-DMS prior, ProteinGym
test split, etc.); this file holds the new entrypoints we added for the
ICML extensions:

- ``train_few_shot_prior``: subsample the train DMS to N variants and fit the
  RF prior + per-protein isotonic calibration.
- ``run_few_shot_sweep``: run the full {n_train × strategy × seed} grid.
- ``run_ood_family_eval``: hold out one protein family entirely and evaluate.
- ``download_megascale``: pull the HF Megascale Parquet shards to the volume.
- ``train_prior_on_megascale``: fit RF prior on Megascale single-mutant train.
- ``run_megascale_t1``: T1 single-mutant evaluation on the dataset3_single test.
- ``run_megascale_t2``: T2 double-mutant evaluation with mandatory additive
  baseline comparison + epistasis-slice metrics.
- ``run_thermompnn_baseline``, ``run_venusrem_baseline``: external SOTA hooks.
- ``build_v2_leaderboard``: aggregate everything into the headline table.
- ``make_nature_figures``: render the publication figures.

Use ``modal run modal_app_v2.py::run_v2_pipeline`` to run end-to-end.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import modal


APP_NAME = "phaseagent-v2"
VOLUME_NAME = "phaseagent-data"  # share the v1 volume so we reuse downloaded data
VOLUME_PATH = "/data"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


CPU_PIP = [
    "numpy>=1.24,<2",  # match v1 pinning for inter-op
    "pandas>=2.0",
    "scipy>=1.10",
    "scikit-learn>=1.2",
    "matplotlib>=3.7",
    "lightgbm>=4.0",
    "pyarrow>=14.0",
    "pyyaml>=6.0",
    "tqdm>=4.65",
    "requests>=2.31",
    "biopython>=1.81",
    "datasets>=2.18",
    "huggingface_hub>=0.20",
]

GPU_PIP = [
    "torch==2.4.0",
    "fair-esm==2.0.0",
    "biotite==0.40.0",
    "torch-geometric==2.4.0",
    "transformers==4.39.2",
]

# torch_scatter ships per-CUDA wheels via the PyG index; pip can't resolve it
# from PyPI alone for torch 2.4.0 + CUDA 12.x.
PYG_WHEEL_INDEX = "https://data.pyg.org/whl/torch-2.4.0+cu121.html"

cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*CPU_PIP)
    .add_local_dir("src/phaseagent", remote_path="/root/phaseagent", copy=True)
)

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*CPU_PIP, *GPU_PIP)
    .add_local_dir("src/phaseagent", remote_path="/root/phaseagent", copy=True)
)

gpu_lora_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*CPU_PIP, *GPU_PIP, "peft==0.10.0", "accelerate==0.29.3")
    .add_local_dir("src/phaseagent", remote_path="/root/phaseagent", copy=True)
)


# ---------------------------------------------------------------------------
# Few-shot
# ---------------------------------------------------------------------------

@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600 * 4)
def train_few_shot_prior(
    n_train: int = 200,
    strategy: str = "random",
    seed: int = 0,
    data: str = "processed/all_dms.parquet",
    splits_csv: str = "outputs/editguard/edit_splits.csv",
    model_out: str | None = None,
    metrics_out: str = "outputs/editguard/few_shot_metrics.csv",
    per_protein_calibration: bool = True,
) -> dict:
    """Subsample the train split to ``n_train`` and fit + calibrate the prior."""
    import pandas as pd

    from phaseagent.edit_splits import add_splits, filter_by_split
    from phaseagent.editguard_prior import DMSFunctionPrior, evaluate_prior
    from phaseagent.few_shot import FewShotConfig, subsample

    df = pd.read_parquet(Path(VOLUME_PATH) / data)
    splits = pd.read_csv(Path(VOLUME_PATH) / splits_csv)
    train = filter_by_split(df, splits, "train")
    val = filter_by_split(df, splits, "val")
    test = filter_by_split(df, splits, "test")

    cfg = FewShotConfig(n_train=int(n_train), strategy=str(strategy), seed=int(seed))
    train_sub = subsample(train, cfg)
    prior = DMSFunctionPrior(per_protein_calibration=per_protein_calibration).fit(
        train_sub, calibrate_df=val, dms_context=train_sub
    )

    if model_out is None:
        model_out = f"outputs/editguard/few_shot/prior_n{n_train}_{strategy}_seed{seed}.pkl"
    model_path = Path(VOLUME_PATH) / model_out
    model_path.parent.mkdir(parents=True, exist_ok=True)
    prior.save(model_path)

    rows = []
    for split_name, sub in [("train_subsample", train_sub), ("val", val), ("test", test)]:
        m = evaluate_prior(prior, sub)
        rows.append(
            {
                "n_train": int(n_train),
                "strategy": str(strategy),
                "seed": int(seed),
                "split": split_name,
                **m,
            }
        )
    metrics_path = Path(VOLUME_PATH) / metrics_out
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    new_rows = pd.DataFrame(rows)
    if metrics_path.exists():
        prev = pd.read_csv(metrics_path)
        out = pd.concat([prev, new_rows], ignore_index=True)
    else:
        out = new_rows
    out.to_csv(metrics_path, index=False)
    volume.commit()
    return {"model": str(model_path), "metrics": str(metrics_path), "n_rows": int(len(new_rows))}


@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600 * 6)
def run_few_shot_sweep(
    sizes: str = "50,100,200,500",
    strategies: str = "random,stratified,active",
    seeds: str = "0,1,2,3,4,5,6,7,8,9",
    data: str = "processed/all_dms.parquet",
    splits_csv: str = "outputs/editguard/edit_splits.csv",
    out: str = "outputs/editguard/few_shot_sweep.csv",
) -> dict:
    """Run the full few-shot grid in one container.

    For the headline table: 4 sizes × 3 strategies × 10 seeds = 120 fits.
    Each fit is fast (RF on ≤500 samples), so this still fits in one CPU
    container. Returns one row per (n_train, strategy, seed, split) cell.
    """
    import pandas as pd

    from phaseagent.edit_splits import filter_by_split
    from phaseagent.editguard_prior import DMSFunctionPrior, evaluate_prior
    from phaseagent.few_shot import FewShotConfig, subsample

    df = pd.read_parquet(Path(VOLUME_PATH) / data)
    splits = pd.read_csv(Path(VOLUME_PATH) / splits_csv)
    train = filter_by_split(df, splits, "train")
    val = filter_by_split(df, splits, "val")
    test = filter_by_split(df, splits, "test")

    sizes_list = [int(x) for x in str(sizes).split(",") if x.strip()]
    strategies_list = [s.strip() for s in str(strategies).split(",") if s.strip()]
    seeds_list = [int(x) for x in str(seeds).split(",") if x.strip()]
    rows = []
    for n in sizes_list:
        for strat in strategies_list:
            for seed in seeds_list:
                cfg = FewShotConfig(n_train=int(n), strategy=str(strat), seed=int(seed))
                sub = subsample(train, cfg)
                if len(sub) < 10 or sub["viable"].nunique() < 2:
                    continue
                prior = DMSFunctionPrior(per_protein_calibration=True).fit(
                    sub, calibrate_df=val, dms_context=sub
                )
                for split_name, frame in [("test", test), ("val", val)]:
                    m = evaluate_prior(prior, frame)
                    rows.append(
                        {
                            "n_train": int(n),
                            "strategy": str(strat),
                            "seed": int(seed),
                            "split": split_name,
                            **m,
                        }
                    )
    out_path = Path(VOLUME_PATH) / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    volume.commit()
    return {"out": str(out_path), "rows": len(rows)}


# ---------------------------------------------------------------------------
# OOD family split
# ---------------------------------------------------------------------------

@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600 * 2)
def run_ood_family_eval(
    held_out_family: str = "fluorescent_protein",
    data: str = "processed/all_dms.parquet",
    out: str | None = None,
    splits_out: str | None = None,
    seed: int = 0,
) -> dict:
    """Hold out a protein family entirely; train on the rest; eval on it."""
    import pandas as pd

    from phaseagent.editguard_prior import DMSFunctionPrior, evaluate_prior
    from phaseagent.family_splits import (
        FamilyOODConfig,
        assert_no_family_leakage,
        make_family_ood_splits,
    )

    if out is None:
        out = f"outputs/editguard/ood_family/{held_out_family}_metrics.csv"
    if splits_out is None:
        splits_out = f"outputs/editguard/ood_family/{held_out_family}_splits.csv"
    df = pd.read_parquet(Path(VOLUME_PATH) / data)
    cfg = FamilyOODConfig(held_out_family=held_out_family, seed=seed)
    splits = make_family_ood_splits(df["dataset_id"].unique(), cfg)
    assert_no_family_leakage(splits, held_out_family)
    splits_path = Path(VOLUME_PATH) / splits_out
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(splits_path, index=False)

    train_ids = set(splits.loc[splits["split"] == "train", "dataset_id"])
    val_ids = set(splits.loc[splits["split"] == "val", "dataset_id"])
    test_ids = set(splits.loc[splits["split"] == "ood_test", "dataset_id"])
    train = df[df["dataset_id"].astype(str).isin(train_ids)]
    val = df[df["dataset_id"].astype(str).isin(val_ids)]
    test = df[df["dataset_id"].astype(str).isin(test_ids)]

    if len(train) == 0:
        raise RuntimeError(f"OOD train set empty for family {held_out_family!r}")
    prior = DMSFunctionPrior(per_protein_calibration=True).fit(
        train, calibrate_df=val if len(val) else None, dms_context=train
    )
    rows = []
    for name, sub in [("train", train), ("val", val), ("ood_test", test)]:
        if len(sub) == 0:
            continue
        m = evaluate_prior(prior, sub)
        rows.append({"held_out_family": held_out_family, "split": name, **m})
    out_path = Path(VOLUME_PATH) / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    volume.commit()
    return {"out": str(out_path), "rows": len(rows)}


# ---------------------------------------------------------------------------
# Megascale
# ---------------------------------------------------------------------------

@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600 * 4)
def download_megascale(
    cache_dir: str = "data/megascale",
    configs: str = "dataset3_single,dataset2,AlphaFold_model_PDBs",
) -> dict:
    """Cache Megascale Parquet shards on the volume.

    Pulls each config from the HF dataset hub. The AlphaFold_model_PDBs
    config is bundled structures we'll feed to ESM-IF / ThermoMPNN.
    """
    from datasets import load_dataset

    cache_path = Path(VOLUME_PATH) / cache_dir
    cache_path.mkdir(parents=True, exist_ok=True)
    configs_list = [c.strip() for c in str(configs).split(",") if c.strip()]
    sizes = {}
    for cfg in configs_list:
        ds = load_dataset("RosettaCommons/MegaScale", cfg, cache_dir=str(cache_path))
        sizes[cfg] = {split: int(len(ds[split])) for split in ds.keys()}
    volume.commit()
    return {"cache_dir": str(cache_path), "sizes": sizes}


@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600 * 4)
def train_prior_on_megascale(
    cache_dir: str = "data/megascale",
    model_out: str = "outputs/megascale/prior_singles.pkl",
    metrics_out: str = "outputs/megascale/prior_singles_metrics.csv",
    per_protein_calibration: bool = True,
) -> dict:
    """Train the RF prior on Megascale single-mutant ΔΔG."""
    import pandas as pd

    from phaseagent.editguard_prior import DMSFunctionPrior, evaluate_prior
    from phaseagent.megascale import MegascaleConfig, load_megascale

    train = load_megascale(MegascaleConfig(config_name="dataset3_single", split="train", cache_dir=cache_dir))
    val = load_megascale(MegascaleConfig(config_name="dataset3_single", split="val", cache_dir=cache_dir))
    test = load_megascale(MegascaleConfig(config_name="dataset3_single", split="test", cache_dir=cache_dir))
    prior = DMSFunctionPrior(per_protein_calibration=per_protein_calibration).fit(
        train, calibrate_df=val, dms_context=train
    )
    model_path = Path(VOLUME_PATH) / model_out
    model_path.parent.mkdir(parents=True, exist_ok=True)
    prior.save(model_path)
    rows = [
        {"split": name, **evaluate_prior(prior, sub)}
        for name, sub in [("train", train), ("val", val), ("test", test)]
    ]
    metrics_path = Path(VOLUME_PATH) / metrics_out
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    volume.commit()
    return {"model": str(model_path), "metrics": str(metrics_path)}


@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600 * 4)
def run_megascale_t2_with_additive(
    cache_dir: str = "data/megascale",
    model_path: str = "outputs/megascale/prior_singles.pkl",
    out: str = "outputs/megascale/t2_doubles_metrics.csv",
    epistasis_threshold: float = 0.1,
    stabilizing_threshold: float = 0.75,
    k: int = 50,
) -> dict:
    """T2 — double-mutant evaluation with mandatory additive baseline.

    The additive baseline (sum-of-singles predictions) is the field-required
    null model post ThermoMPNN-D / MULTI-evolve critique. We report:
    - overall Spearman of model vs additive
    - epistasis-slice Spearman (where |obs - additive| > threshold)
    - stabilizing-pair recall@k
    """
    import pandas as pd

    from phaseagent.additive_baseline import (
        AdditivePredictor,
        epistasis_slice_metrics,
        stabilizing_pair_recall,
    )
    from phaseagent.editguard_prior import DMSFunctionPrior
    from phaseagent.megascale import (
        MegascaleConfig,
        load_megascale,
        load_megascale_double_mutant_test,
    )

    singles_train = load_megascale(MegascaleConfig(config_name="dataset3_single", split="train", cache_dir=cache_dir))
    doubles = load_megascale_double_mutant_test(cache_dir=cache_dir)
    prior = DMSFunctionPrior.load(Path(VOLUME_PATH) / model_path)

    # Fit the additive baseline on the singles train set (= same data the prior saw).
    additive = AdditivePredictor(score_col="fitness_norm").fit(singles_train)

    add_pred = additive.predict(doubles)
    model_pred = prior.predict_fitness(doubles)
    rows = []
    for ds_id, sub in doubles.groupby("dataset_id"):
        add_sub = additive.predict(sub)
        model_sub = prior.predict_fitness(sub)
        m = epistasis_slice_metrics(sub, model_sub, add_sub, epistasis_threshold=epistasis_threshold)
        m["dataset_id"] = str(ds_id)
        m["stabilizing_recall_at_k_model"] = stabilizing_pair_recall(
            sub, model_sub, stabilizing_threshold=stabilizing_threshold, k=k
        )
        m["stabilizing_recall_at_k_additive"] = stabilizing_pair_recall(
            sub, add_sub, stabilizing_threshold=stabilizing_threshold, k=k
        )
        m["additive_coverage"] = additive.coverage(sub)
        rows.append(m)
    out_path = Path(VOLUME_PATH) / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    volume.commit()
    return {"out": str(out_path), "n_proteins": len(rows)}


# ---------------------------------------------------------------------------
# Conformal abstention
# ---------------------------------------------------------------------------

@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600 * 2)
def run_conformal_abstention(
    data: str = "processed/all_dms.parquet",
    splits_csv: str = "outputs/editguard/edit_splits.csv",
    model_path: str = "outputs/editguard/editguard_prior.pkl",
    out: str = "outputs/editguard/conformal_abstention_curve.csv",
    alpha: float = 0.1,
    abstain_rates: str = "0.0,0.1,0.2,0.3,0.4,0.5",
) -> dict:
    """Sweep abstention rate and report per-protein hit rate on what's kept."""
    import numpy as np
    import pandas as pd

    from phaseagent.conformal import ConformalRegressor
    from phaseagent.edit_splits import filter_by_split
    from phaseagent.editguard_prior import DMSFunctionPrior

    df = pd.read_parquet(Path(VOLUME_PATH) / data)
    splits = pd.read_csv(Path(VOLUME_PATH) / splits_csv)
    val = filter_by_split(df, splits, "val")
    test = filter_by_split(df, splits, "test")

    prior = DMSFunctionPrior.load(Path(VOLUME_PATH) / model_path)

    abstain_list = [float(x) for x in str(abstain_rates).split(",") if x.strip()]
    rows = []
    for ar in abstain_list:
        cr = ConformalRegressor(prior, alpha=alpha, group_col="dataset_id", max_abstain_rate=ar)
        cr.calibrate(val)
        coverage = cr.coverage(test)
        abstain = cr.abstain_mask(test)
        kept = (~abstain).sum()
        # On the kept variants, what fraction are truly viable?
        if kept > 0:
            kept_df = test[~abstain]
            hit_rate = float((kept_df["viable"] == 1).mean())
        else:
            hit_rate = float("nan")
        rows.append(
            {
                "abstain_rate": ar,
                "n_kept": int(kept),
                "n_total": int(len(test)),
                "kept_fraction": float(kept / max(1, len(test))),
                "hit_rate": hit_rate,
                "coverage_global": coverage.get("global", float("nan")),
            }
        )
    out_path = Path(VOLUME_PATH) / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    volume.commit()
    return {"out": str(out_path), "rows": len(rows)}


# ---------------------------------------------------------------------------
# Nature figures
# ---------------------------------------------------------------------------

@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=1800)
def make_nature_figures(
    leaderboard: str = "outputs/editguard/headline_leaderboard.csv",
    pairwise: str = "outputs/editguard/headline_pairwise_wilcoxon.csv",
    few_shot: str = "outputs/editguard/few_shot_sweep.csv",
    abstention: str = "outputs/editguard/conformal_abstention_curve.csv",
    reliability: str = "outputs/editguard/editguard_prior_calibration_bins.csv",
    out_dir: str = "outputs/editguard/figures_v2",
) -> dict:
    """Render the publication figures from the saved CSVs."""
    import pandas as pd

    from phaseagent.nature_figures import (
        figure_calibration_and_abstention,
        figure_few_shot_scaling,
        figure_headline_leaderboard,
        save_figure,
        setup_nature_style,
    )

    setup_nature_style()
    out = Path(VOLUME_PATH) / out_dir
    out.mkdir(parents=True, exist_ok=True)
    written = []

    lb_path = Path(VOLUME_PATH) / leaderboard
    if lb_path.exists():
        lb = pd.read_csv(lb_path)
        pw_path = Path(VOLUME_PATH) / pairwise
        pw = pd.read_csv(pw_path) if pw_path.exists() else None
        fig = figure_headline_leaderboard(lb, pairwise_pvals=pw)
        written += [str(p) for p in save_figure(fig, out / "fig1_headline")]

    fs_path = Path(VOLUME_PATH) / few_shot
    if fs_path.exists():
        fs = pd.read_csv(fs_path)
        # Aggregate across seeds.
        agg = fs.groupby(["n_train", "strategy", "split"]).agg(
            mean=("auroc", "mean"),
            ci_lo=("auroc", lambda v: float(v.quantile(0.025))),
            ci_hi=("auroc", lambda v: float(v.quantile(0.975))),
        ).reset_index()
        agg = agg[agg["split"] == "test"]
        agg["method"] = agg["strategy"]
        fig = figure_few_shot_scaling(agg, metric="functional_hit_rate")
        written += [str(p) for p in save_figure(fig, out / "fig2_few_shot")]

    rel_path = Path(VOLUME_PATH) / reliability
    if rel_path.exists():
        rel = pd.read_csv(rel_path)
        # Translate the existing reliability-bins schema into the figure schema.
        if "predicted_mean" in rel.columns:
            rel = rel.rename(columns={"predicted_mean": "bin_center", "empirical_mean": "empirical"})
        ab_path = Path(VOLUME_PATH) / abstention
        ab = pd.read_csv(ab_path) if ab_path.exists() else None
        fig = figure_calibration_and_abstention(rel, ab)
        written += [str(p) for p in save_figure(fig, out / "fig3_calibration")]

    volume.commit()
    return {"figures_written": written, "out_dir": str(out)}


# ---------------------------------------------------------------------------
# Real ESM-2 mean-pool RF baseline (EVOLVEpro recipe)
# ---------------------------------------------------------------------------

@app.function(image=gpu_image, volumes={VOLUME_PATH: volume}, gpu="A10G", timeout=3600 * 6)
def precompute_esm2_embeddings(
    data: str = "processed/all_dms.parquet",
    out: str = "outputs/editguard/esm2_embeddings.parquet",
    model_name: str = "esm2_t33_650M_UR50D",
    pool: str = "mean",
    batch_size: int = 8,
) -> dict:
    """Pre-compute ESM-2 650M mean-pool embeddings for every variant.

    Runs once on GPU; downstream RF/LightGBM baselines read this Parquet
    and never touch the GPU again. Caches under
    ``outputs/editguard/esm2_embeddings.parquet`` keyed by
    (dataset_id, mutation_notation).
    """
    import numpy as np
    import pandas as pd

    from phaseagent.embeddings import EmbeddingConfig, ESM2Embedder, embed_dataframe

    df = pd.read_parquet(Path(VOLUME_PATH) / data)
    if "mutated_sequence" not in df.columns:
        raise ValueError("processed all_dms.parquet must include 'mutated_sequence'")
    cfg = EmbeddingConfig(model_name=model_name, pool=pool, batch_size=batch_size)
    embedder = ESM2Embedder(cfg)
    print(f"[esm2-emb] computing {len(df)} embeddings with {model_name} pool={pool}")
    X = embed_dataframe(df, embedder, seq_col="mutated_sequence")
    embed_cols = [f"emb_{i}" for i in range(X.shape[1])]
    out_df = pd.DataFrame(X, columns=embed_cols)
    out_df["dataset_id"] = df["dataset_id"].astype(str).to_numpy()
    out_df["mutation_notation"] = df["mutation_notation"].astype(str).to_numpy()
    out_path = Path(VOLUME_PATH) / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    volume.commit()
    return {"out": str(out_path), "n": int(len(out_df)), "dim": int(X.shape[1])}


@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600 * 4)
def run_esm2_rf_baseline(
    data: str = "processed/all_dms.parquet",
    embeddings: str = "outputs/editguard/esm2_embeddings.parquet",
    splits_csv: str = "outputs/editguard/edit_splits.csv",
    out: str = "outputs/editguard/esm2_rf_metrics.csv",
) -> dict:
    """Real ESM-2-mean-pool + RF (EVOLVEpro recipe) on the test split."""
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr
    from sklearn.metrics import average_precision_score, roc_auc_score

    from phaseagent.edit_splits import filter_by_split
    from phaseagent.embedding_baselines import RFEmbeddingBaseline

    df = pd.read_parquet(Path(VOLUME_PATH) / data)
    emb = pd.read_parquet(Path(VOLUME_PATH) / embeddings)
    splits = pd.read_csv(Path(VOLUME_PATH) / splits_csv)

    train = filter_by_split(df, splits, "train")
    test = filter_by_split(df, splits, "test")
    embed_cols = [c for c in emb.columns if c.startswith("emb_")]
    train_X = train.merge(emb, on=["dataset_id", "mutation_notation"], how="left")[embed_cols].to_numpy(dtype=np.float32)
    test_X = test.merge(emb, on=["dataset_id", "mutation_notation"], how="left")[embed_cols].to_numpy(dtype=np.float32)
    print(f"[esm2-rf] train n={len(train)} (X {train_X.shape}), test n={len(test)} (X {test_X.shape})")

    model = RFEmbeddingBaseline().fit(train, precomputed_embeddings=train_X)
    preds = model.predict_fitness(test, precomputed_embeddings=test_X)
    rows = []
    for ds_id, sub in test.groupby("dataset_id"):
        idx = test["dataset_id"] == ds_id
        sub_preds = preds[idx.to_numpy()]
        sub_y = sub["viable"].to_numpy(dtype=int)
        sub_fit = sub["fitness_norm"].to_numpy(dtype=float)
        try:
            auroc = float(roc_auc_score(sub_y, sub_preds)) if sub_y.std() else float("nan")
            auprc = float(average_precision_score(sub_y, sub_preds)) if sub_y.std() else float("nan")
        except Exception:
            auroc = float("nan")
            auprc = float("nan")
        rows.append(
            {
                "method": "esm2_rf_eVOLVEpro",
                "dataset_id": str(ds_id),
                "n": int(len(sub)),
                "auroc": auroc,
                "auprc": auprc,
                "spearman": float(spearmanr(sub_fit, sub_preds, nan_policy="omit").statistic),
            }
        )
    out_path = Path(VOLUME_PATH) / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    volume.commit()
    return {"out": str(out_path), "n_proteins": len(rows)}


# ---------------------------------------------------------------------------
# Real Twisted SMC sampler on DPLM-650M
# ---------------------------------------------------------------------------

@app.function(image=gpu_image, volumes={VOLUME_PATH: volume}, gpu="A10G", timeout=3600 * 8)
def run_twisted_smc_dplm(
    data: str = "processed/all_dms.parquet",
    tasks_csv: str = "outputs/editguard/edit_tasks.csv",
    splits_csv: str = "outputs/editguard/edit_splits.csv",
    prior_path: str = "outputs/editguard/editguard_prior.pkl",
    out: str = "outputs/editguard/twisted_smc_metrics.csv",
    n_particles: int = 8,
    n_denoise_steps: int = 64,
    n_outer_iterations: int = 2,
    beta: float = 1.0,
    seeds: int = 3,
    max_tasks: int = 0,
) -> dict:
    """Run real Twisted SMC on DPLM-650M for held-out test editing tasks."""
    import numpy as np
    import pandas as pd

    from phaseagent.edit_generators import infer_wildtype_sequence
    from phaseagent.edit_splits import assert_tasks_in_split
    from phaseagent.editguard_prior import DMSFunctionPrior
    from phaseagent.editing_tasks import task_from_row
    from phaseagent.twisted_smc import TwistedSMCConfig, run_twisted_smc, best_particle
    from phaseagent.twisted_smc_dplm import DMSPriorRewardModel, RealDPLMBackbone

    df = pd.read_parquet(Path(VOLUME_PATH) / data)
    tasks = pd.read_csv(Path(VOLUME_PATH) / tasks_csv)
    splits = pd.read_csv(Path(VOLUME_PATH) / splits_csv)
    test_ds = set(splits.loc[splits["split"] == "test", "dataset_id"].astype(str))
    tasks = tasks[tasks["dataset_id"].astype(str).isin(test_ds)].reset_index(drop=True)
    assert_tasks_in_split(tasks, splits, "test")
    if max_tasks > 0:
        tasks = tasks.head(max_tasks)
    prior = DMSFunctionPrior.load(Path(VOLUME_PATH) / prior_path)

    backbone = RealDPLMBackbone()
    rows = []
    for task_idx, row in tasks.iterrows():
        task = task_from_row(row)
        ds_sub = df[df["dataset_id"] == task.dataset_id]
        if len(ds_sub) == 0:
            continue
        wt_seq = infer_wildtype_sequence(ds_sub, task.dataset_id)
        if not wt_seq:
            continue
        forbidden = frozenset(task.forbidden_residues)
        backbone.forbidden_aas = forbidden
        reward = DMSPriorRewardModel(prior=prior, dataset_id=task.dataset_id, wt_sequence=wt_seq)
        for seed in range(seeds):
            cfg = TwistedSMCConfig(
                n_particles=n_particles,
                n_denoise_steps=n_denoise_steps,
                n_outer_iterations=n_outer_iterations,
                beta=beta,
                seed=int(seed),
            )
            print(f"[smc] task={task_idx} ds={task.dataset_id} obj={task.objective} k={task.edit_budget} seed={seed} ...")
            result = run_twisted_smc(
                wt_seq=wt_seq,
                edit_budget=int(task.edit_budget),
                backbone=backbone,
                reward=reward,
                config=cfg,
            )
            best = best_particle(result)
            rows.append(
                {
                    "method": f"twisted_smc_dplm_beta{beta}_K{n_particles}",
                    "task_idx": int(task_idx),
                    "dataset_id": task.dataset_id,
                    "objective": task.objective,
                    "edit_budget": int(task.edit_budget),
                    "seed": int(seed),
                    "best_reward": float(np.max(result["rewards"])),
                    "mean_reward": float(np.mean(result["rewards"])),
                    "best_sequence": best.sequence,
                }
            )
    out_path = Path(VOLUME_PATH) / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    volume.commit()
    return {"out": str(out_path), "rows": len(rows)}


# ---------------------------------------------------------------------------
# ThermoMPNN-D (real, Kuhlman-Lab) — Megascale double-mutant baseline
# ---------------------------------------------------------------------------

thermompnn_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(*CPU_PIP, *GPU_PIP)
    .add_local_dir("src/phaseagent", remote_path="/root/phaseagent", copy=True)
)


@app.function(image=thermompnn_image, volumes={VOLUME_PATH: volume}, gpu="A10G", timeout=3600 * 6)
def run_thermompnn_d_megascale_t2(
    cache_dir: str = "data/megascale",
    out: str = "outputs/megascale/thermompnn_d_t2_metrics.csv",
    repo_url: str = "https://github.com/Kuhlman-Lab/ThermoMPNN-D.git",
    repo_branch: str = "main",
) -> dict:
    """Run ThermoMPNN-D on the Megascale double-mutant test split.

    Clones the upstream Kuhlman-Lab/ThermoMPNN-D repo at runtime and adds
    its source root to sys.path (the repo is a research codebase, not a
    pip-installable package, so installing inside the image fails at
    build time). If the repo's import surface changes the function will
    fail loud — never silently fall back to a stub.
    """
    import os
    import subprocess
    import sys
    import pandas as pd

    repo_dir = Path("/tmp/thermompnn_d")
    if not repo_dir.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", repo_branch, repo_url, str(repo_dir)],
            check=True,
        )
    sys.path.insert(0, str(repo_dir))
    try:
        from thermompnn.inference import ThermoMPNNDInference  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            f"ThermoMPNN-D import failed after cloning {repo_url}@{repo_branch}: {exc}\n"
            f"Repo layout (top-level): {sorted(p.name for p in repo_dir.iterdir())}"
        ) from exc

    from phaseagent.megascale import load_megascale_double_mutant_test

    doubles = load_megascale_double_mutant_test(cache_dir=cache_dir)
    runner = ThermoMPNNDInference()
    rows = []
    for ds_id, sub in doubles.groupby("dataset_id"):
        try:
            preds = runner.predict_for_dataset(ds_id, sub)
        except Exception as exc:
            print(f"[thermompnn_d] {ds_id} skipped: {exc}")
            continue
        sub2 = sub.copy()
        sub2["thermompnn_d_pred"] = preds
        rows.append(sub2)
    if not rows:
        out_path = Path(VOLUME_PATH) / out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(out_path, index=False)
        return {"out": str(out_path), "rows": 0, "warning": "ThermoMPNN-D returned 0 predictions"}
    full = pd.concat(rows, ignore_index=True)
    out_path = Path(VOLUME_PATH) / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out_path, index=False)
    volume.commit()
    return {"out": str(out_path), "rows": int(len(full))}


# ---------------------------------------------------------------------------
# EditGuard-Clin: rescue-event extraction + variant-conditioned predictor
# ---------------------------------------------------------------------------

@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600 * 4)
def extract_megascale_rescue_events(
    cache_dir: str = "data/megascale",
    out: str = "outputs/clin/rescue_events_megascale.parquet",
    summary_out: str = "outputs/clin/rescue_events_summary.csv",
    only_damaging_first: bool = True,
) -> dict:
    """Extract rescue training pairs from Megascale singles + doubles."""
    import pandas as pd

    from phaseagent.megascale import (
        MegascaleConfig,
        load_megascale,
        load_megascale_double_mutant_test,
    )
    from phaseagent.rescue_events import extract_rescue_events, summarize_rescue_events

    print("[clin] loading Megascale singles + doubles ...")
    singles = load_megascale(MegascaleConfig(config_name="dataset3_single", split="train", cache_dir=cache_dir))
    doubles = load_megascale_double_mutant_test(cache_dir=cache_dir)
    print(f"[clin] {len(singles)} singles, {len(doubles)} doubles")
    events = extract_rescue_events(doubles, singles, only_damaging_first=only_damaging_first)
    print(f"[clin] extracted {len(events)} rescue-event rows")
    summary = summarize_rescue_events(events)

    out_path = Path(VOLUME_PATH) / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(out_path, index=False)
    summary.to_csv(Path(VOLUME_PATH) / summary_out, index=False)
    volume.commit()
    return {
        "events_out": str(out_path),
        "summary_out": str(Path(VOLUME_PATH) / summary_out),
        "n_events": int(len(events)),
        "n_proteins": int(summary["dataset_id"].nunique()) if len(summary) else 0,
        "global_rescue_rate": float(events["rescue_class"].isin(("partial", "full", "super")).mean()) if len(events) else float("nan"),
    }


@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600 * 4)
def train_rescue_predictor(
    events_path: str = "outputs/clin/rescue_events_megascale.parquet",
    model_out: str = "outputs/clin/rescue_predictor.pkl",
    metrics_out: str = "outputs/clin/rescue_predictor_metrics.csv",
    train_frac: float = 0.7,
    n_estimators: int = 200,
    seed: int = 0,
    held_out_proteins_csv: str | None = None,
) -> dict:
    """Train the rescue predictor on extracted rescue events.

    Splits at the *protein* level (not row level) so test proteins are unseen.
    Reports per-class precision/recall/AUROC and per-protein recovery of
    full-rescue events.
    """
    import pickle
    import numpy as np
    import pandas as pd
    from sklearn.metrics import average_precision_score, roc_auc_score

    from phaseagent.rescue_predictor import RescuePredictor

    events = pd.read_parquet(Path(VOLUME_PATH) / events_path)
    if len(events) < 100:
        raise RuntimeError(f"need >=100 events to train; got {len(events)}")
    rng = np.random.default_rng(int(seed))
    proteins = sorted(events["dataset_id"].unique())
    rng.shuffle(proteins)
    n_train = int(round(len(proteins) * train_frac))
    train_proteins = set(proteins[:n_train])
    test_proteins = set(proteins[n_train:])
    train_df = events[events["dataset_id"].isin(train_proteins)]
    test_df = events[events["dataset_id"].isin(test_proteins)]
    print(f"[clin] train: {len(train_df)} rows / {len(train_proteins)} proteins; "
          f"test: {len(test_df)} rows / {len(test_proteins)} proteins")

    model = RescuePredictor(n_estimators=int(n_estimators), random_state=int(seed)).fit(train_df)
    p_train = model.predict_rescue_proba(train_df)
    p_test = model.predict_rescue_proba(test_df)

    rows = []
    for label, df, scores in (("train", train_df, p_train), ("test", test_df, p_test)):
        y = df["rescue_class"].isin(("partial", "full", "super")).astype(int).to_numpy()
        if y.sum() in (0, len(y)):
            rows.append({"split": label, "n": len(df), "auroc": float("nan"), "auprc": float("nan"), "rescue_rate": float(y.mean())})
            continue
        rows.append({
            "split": label,
            "n": int(len(df)),
            "auroc": float(roc_auc_score(y, scores)),
            "auprc": float(average_precision_score(y, scores)),
            "rescue_rate": float(y.mean()),
        })

    out_model = Path(VOLUME_PATH) / model_out
    out_model.parent.mkdir(parents=True, exist_ok=True)
    with open(out_model, "wb") as f:
        pickle.dump({
            "model": model,
            "train_proteins": sorted(train_proteins),
            "test_proteins": sorted(test_proteins),
            "n_estimators": int(n_estimators),
            "seed": int(seed),
        }, f)
    pd.DataFrame(rows).to_csv(Path(VOLUME_PATH) / metrics_out, index=False)
    volume.commit()
    return {
        "model": str(out_model),
        "metrics": str(Path(VOLUME_PATH) / metrics_out),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "test_auroc": float(rows[1]["auroc"]) if len(rows) > 1 else float("nan"),
    }


@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600 * 4)
def calibrate_rescue_conformal(
    events_path: str = "outputs/clin/rescue_events_megascale.parquet",
    model_path: str = "outputs/clin/rescue_predictor.pkl",
    out: str = "outputs/clin/rescue_conformal.pkl",
    eval_out: str = "outputs/clin/rescue_conformal_eval.csv",
    alphas: str = "0.05,0.1,0.2,0.3",
    calib_frac_of_test: float = 0.5,
) -> dict:
    """Calibrate the conformal rescue-set guarantee + measure realised coverage.

    Splits the held-out test set into calibration + held-out evaluation. For
    each tolerated false-rescue rate alpha, calibrates the threshold and
    reports realised precision on the eval split.
    """
    import pickle
    import numpy as np
    import pandas as pd

    from phaseagent.rescue_predictor import ConformalRescueSet

    events = pd.read_parquet(Path(VOLUME_PATH) / events_path)
    with open(Path(VOLUME_PATH) / model_path, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    test_proteins = set(bundle["test_proteins"])
    test_df = events[events["dataset_id"].isin(test_proteins)].reset_index(drop=True)
    rng = np.random.default_rng(int(bundle.get("seed", 0)))
    idx = np.arange(len(test_df))
    rng.shuffle(idx)
    n_calib = int(len(idx) * float(calib_frac_of_test))
    calib_df = test_df.iloc[idx[:n_calib]].reset_index(drop=True)
    eval_df = test_df.iloc[idx[n_calib:]].reset_index(drop=True)

    alphas_list = [float(a) for a in str(alphas).split(",") if a.strip()]
    rows = []
    csets = {}
    for a in alphas_list:
        cset = ConformalRescueSet(predictor=model, alpha=a)
        cset.calibrate(calib_df)
        csets[a] = cset
        scores = model.predict_rescue_proba(eval_df)
        labels = eval_df["rescue_class"].isin(("partial", "full", "super")).astype(int).to_numpy()
        above = scores >= cset.calibrated_threshold
        n_above = int(above.sum())
        precision = float(labels[above].mean()) if n_above > 0 else float("nan")
        recall = float((labels[above].sum() / max(labels.sum(), 1)))
        rows.append({
            "alpha": a,
            "calibrated_threshold": cset.calibrated_threshold,
            "n_calib": int(len(calib_df)),
            "n_eval": int(len(eval_df)),
            "n_above_threshold": n_above,
            "realised_precision": precision,
            "realised_recall": recall,
            "false_rescue_rate": 1.0 - precision if n_above > 0 else float("nan"),
        })

    with open(Path(VOLUME_PATH) / out, "wb") as f:
        pickle.dump({"calibrators": csets, "alphas": alphas_list, "test_proteins": sorted(test_proteins)}, f)
    pd.DataFrame(rows).to_csv(Path(VOLUME_PATH) / eval_out, index=False)
    volume.commit()
    return {"conformal_out": str(Path(VOLUME_PATH) / out), "eval_out": str(Path(VOLUME_PATH) / eval_out)}


# ---------------------------------------------------------------------------
# EditGuard-Clin: ProteinGym cross-domain rescue validation
# ---------------------------------------------------------------------------

@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600 * 4)
def scan_proteingym_multimutant_assays(
    raw_dir: str = "raw/proteingym_v1_3/DMS_ProteinGym_substitutions",
    out: str = "outputs/clin/proteingym_multimutant_index.csv",
    min_doubles: int = 50,
) -> dict:
    """Scan all ProteinGym v1.3 CSVs and report which have ≥min_doubles multi-mutants."""
    import pandas as pd

    raw_path = Path(VOLUME_PATH) / raw_dir
    rows = []
    for csv_path in sorted(raw_path.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path, usecols=["mutant"])
        except Exception as e:
            print(f"[scan-skip] {csv_path.name}: {e}")
            continue
        n = len(df)
        d = df["mutant"].astype(str).apply(
            lambda s: len([t for t in s.replace(",", ":").split(":") if t.strip()])
        )
        n_doubles = int((d == 2).sum())
        n_higher = int((d >= 2).sum())
        if n_doubles >= min_doubles or n_higher >= min_doubles:
            rows.append({
                "assay": csv_path.stem,
                "n_total": n,
                "n_singles": int((d == 1).sum()),
                "n_doubles": n_doubles,
                "n_higher_order": int((d >= 3).sum()),
            })
    df_out = pd.DataFrame(rows).sort_values("n_doubles", ascending=False).reset_index(drop=True)
    df_out.to_csv(Path(VOLUME_PATH) / out, index=False)
    volume.commit()
    return {"out": str(Path(VOLUME_PATH) / out), "n_assays_with_multimutants": len(df_out)}


@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600 * 4)
def validate_rescue_on_proteingym_clinical(
    raw_dir: str = "raw/proteingym_v1_3/DMS_ProteinGym_substitutions",
    model_path: str = "outputs/clin/rescue_predictor.pkl",
    out: str = "outputs/clin/proteingym_clinical_rescue_eval.csv",
    pretty_out: str = "outputs/clin/proteingym_clinical_rescue_summary.csv",
    target_assays: str = (
        "BRCA1_HUMAN_Findlay_2018,"
        "PTEN_HUMAN_Mighell_2018,"
        "PTEN_HUMAN_Matreyek_2021,"
        "TPMT_HUMAN_Matreyek_2018,"
        "P53_HUMAN_Kotler_2018,"
        "BLAT_ECOLX_Stiffler_2015,"
        "BLAT_ECOLX_Firnberg_2014,"
        "BLAT_ECOLX_Jacquier_2013"
    ),
) -> dict:
    """Extract rescue events from clinical ProteinGym assays + score with predictor.

    Each assay's CSV has per-variant DMS scores including multi-mutants in
    ``mutant`` notation. We rank-norm fitness per assay (matching our
    canonical schema), extract rescue events (m1 damaging + m2 candidate),
    score with the Megascale-trained predictor, and report cross-domain AUROC.
    """
    import pickle
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr
    from sklearn.metrics import average_precision_score, roc_auc_score

    from phaseagent.rescue_events import extract_rescue_events
    from phaseagent.rescue_predictor import RescuePredictor

    with open(Path(VOLUME_PATH) / model_path, "rb") as f:
        bundle = pickle.load(f)
    model: RescuePredictor = bundle["model"]

    raw_path = Path(VOLUME_PATH) / raw_dir
    assays = [a.strip() for a in str(target_assays).split(",") if a.strip()]
    rows = []
    summary_rows = []
    for assay in assays:
        csv_path = raw_path / f"{assay}.csv"
        if not csv_path.exists():
            print(f"[skip] {assay}: csv not found")
            continue
        df = pd.read_csv(csv_path)
        # Standard ProteinGym v1.3 columns: mutant, mutated_sequence, DMS_score, DMS_score_bin.
        if "mutant" not in df.columns or "DMS_score" not in df.columns:
            print(f"[skip] {assay}: unexpected schema")
            continue
        df = df.rename(columns={"mutant": "mutation_notation", "DMS_score": "_raw_fit"})
        df["dataset_id"] = assay
        # Per-assay rank-normalize the DMS_score into [0,1].
        ranks = df["_raw_fit"].rank(method="average")
        df["fitness_norm"] = (ranks - 1) / max(len(df) - 1, 1)
        df["mutation_distance"] = df["mutation_notation"].astype(str).apply(
            lambda s: len([t for t in s.replace(",", ":").split(":") if t.strip()])
        )
        singles = df[df["mutation_distance"] == 1]
        doubles = df[df["mutation_distance"] == 2]
        if len(singles) < 30 or len(doubles) < 30:
            print(f"[skip] {assay}: too few singles ({len(singles)}) or doubles ({len(doubles)})")
            continue
        events = extract_rescue_events(doubles, singles, only_damaging_first=True)
        if len(events) < 30:
            print(f"[skip] {assay}: only {len(events)} rescue events extracted")
            continue
        scores = model.predict_rescue_proba(events)
        labels = events["rescue_class"].isin(("partial", "full", "super")).astype(int).to_numpy()
        if labels.sum() in (0, len(labels)):
            print(f"[skip] {assay}: all-positive or all-negative labels")
            continue
        auroc = float(roc_auc_score(labels, scores))
        auprc = float(average_precision_score(labels, scores))
        spearman_obs_score = float(
            spearmanr(events["combined_score"], scores, nan_policy="omit").statistic
        )
        rate = float(labels.mean())
        summary_rows.append(
            {
                "assay": assay,
                "n_events": int(len(events)),
                "n_singles": int(len(singles)),
                "n_doubles": int(len(doubles)),
                "rescue_rate": rate,
                "auroc": auroc,
                "auprc": auprc,
                "auprc_lift_over_base_rate": auprc - rate,
                "spearman_score_vs_combined_fitness": spearman_obs_score,
            }
        )
        # Per-row predictions for the figure / case-studies layer.
        events = events.copy()
        events["assay"] = assay
        events["rescue_score"] = scores
        events["positive_label"] = labels
        rows.append(events)
        print(f"[ok ] {assay:35s} n={len(events):>5}  AUROC={auroc:.3f}  AUPRC={auprc:.3f}  rate={rate:.3f}")

    out_path = Path(VOLUME_PATH) / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(out_path, index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(Path(VOLUME_PATH) / pretty_out, index=False)
    volume.commit()
    return {
        "summary_out": str(Path(VOLUME_PATH) / pretty_out),
        "events_out": str(out_path),
        "n_assays_evaluated": len(summary_rows),
        "median_auroc": float(summary_df["auroc"].median()) if len(summary_df) else float("nan"),
    }


# ---------------------------------------------------------------------------
# EditGuard-Clin: ClinVar atlas builder (the deployable artifact)
# ---------------------------------------------------------------------------

clin_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*CPU_PIP, "biopython>=1.81", "requests>=2.31")
    .add_local_dir("src/phaseagent", remote_path="/root/phaseagent", copy=True)
)


@app.function(image=clin_image, volumes={VOLUME_PATH: volume}, timeout=3600 * 2)
def build_curated_pathogenic_atlas_inputs(
    out_variants: str = "outputs/clin/curated_pathogenic.csv",
    out_seqs: str = "outputs/clin/curated_wt_sequences.csv",
) -> dict:
    """Resolve curated pathogenic variants + their WT UniProt sequences.

    Pulls the canonical sequence for each unique UniProt accession from the
    UniProt REST API (free, no rate-limit issues at our scale of <50 calls).
    """
    import time
    import pandas as pd
    import requests

    from phaseagent.curated_pathogenic import to_dataframe

    df = to_dataframe()
    out_path_v = Path(VOLUME_PATH) / out_variants
    out_path_v.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path_v, index=False)

    seqs = []
    for uniprot in sorted(df["uniprot"].unique()):
        try:
            r = requests.get(f"https://rest.uniprot.org/uniprotkb/{uniprot}.fasta", timeout=15)
            r.raise_for_status()
            fasta = r.text
            seq = "".join(line for line in fasta.splitlines() if not line.startswith(">"))
            seqs.append({"uniprot": uniprot, "sequence": seq, "length": len(seq)})
            print(f"[uniprot] {uniprot}: {len(seq)} residues")
        except Exception as exc:
            print(f"[uniprot] {uniprot} failed: {exc}")
        time.sleep(0.4)
    seq_df = pd.DataFrame(seqs)
    out_path_s = Path(VOLUME_PATH) / out_seqs
    seq_df.to_csv(out_path_s, index=False)
    # Also write a gene→sequence convenience CSV for the atlas builder.
    gene_seqs = (
        df[["gene", "uniprot"]]
        .drop_duplicates()
        .merge(seq_df, on="uniprot", how="left")[["gene", "sequence"]]
    )
    gene_seqs.to_csv(Path(VOLUME_PATH) / "outputs/clin/curated_gene_sequences.csv", index=False)
    volume.commit()
    return {
        "variants_csv": str(out_path_v),
        "sequences_csv": str(out_path_s),
        "n_variants": int(len(df)),
        "n_uniprot": int(len(seq_df)),
    }


@app.function(image=clin_image, volumes={VOLUME_PATH: volume}, timeout=3600 * 4)
def fetch_clinvar_pathogenic(
    genes: str = "TP53,CFTR,BRCA1,PTEN,TPMT,SERPINA1,LYZ,DCTN1",
    out: str = "outputs/clin/clinvar_pathogenic.csv",
    max_per_gene: int = 500,
) -> dict:
    """Fetch pathogenic missense ClinVar records for the disease-gene panel."""
    import pandas as pd

    from phaseagent.clinvar import ClinVarConfig, fetch_for_gene_panel

    gene_list = [g.strip() for g in str(genes).split(",") if g.strip()]
    cfg = ClinVarConfig(max_per_gene=int(max_per_gene))
    df = fetch_for_gene_panel(gene_list, cfg)
    out_path = Path(VOLUME_PATH) / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    volume.commit()
    summary = df.groupby("gene").size().to_dict() if len(df) else {}
    return {"out": str(out_path), "n_variants": int(len(df)), "per_gene": summary}


@app.function(image=clin_image, volumes={VOLUME_PATH: volume}, timeout=3600 * 4)
def build_rescue_atlas(
    clinvar_csv: str = "outputs/clin/clinvar_pathogenic.csv",
    model_path: str = "outputs/clin/rescue_predictor.pkl",
    conformal_path: str = "outputs/clin/rescue_conformal.pkl",
    out_dir: str = "outputs/clin/atlas",
    max_top_k: int = 50,
    alphas: str = "0.05,0.10,0.20,0.30",
    wt_seqs_csv: str | None = None,
) -> dict:
    """Build the deployable atlas: per (gene, pathogenic variant) → top-k rescues
    at multiple α-controlled false-rescue rates.

    For each pathogenic variant whose UniProt sequence we have, we enumerate
    every candidate single-AA edit at every other position, score with the
    rescue predictor, and emit a JSON record:
      {
        gene, variant, top_k: [
          { suppressor, score, position, mutant_aa,
            in_alpha_5pct: bool, in_alpha_10pct: bool, ...
          }, ...
        ]
      }
    """
    import json
    import pickle
    import pandas as pd

    from phaseagent.rescue_predictor import (
        ConformalRescueSet,
        RescuePredictor,
        enumerate_candidate_pairs_for_target,
    )

    clin_df = pd.read_csv(Path(VOLUME_PATH) / clinvar_csv)
    if len(clin_df) == 0:
        return {"warning": "no clinvar variants supplied", "n_records": 0}

    with open(Path(VOLUME_PATH) / model_path, "rb") as f:
        bundle = pickle.load(f)
    model: RescuePredictor = bundle["model"]
    with open(Path(VOLUME_PATH) / conformal_path, "rb") as f:
        cal_bundle = pickle.load(f)
    csets: dict[float, ConformalRescueSet] = cal_bundle["calibrators"]

    # WT sequences come from a CSV (gene → sequence) provided by the caller, or
    # fall back to the curated lookup we use for the literature validation.
    wt_lookup: dict[str, str] = {}
    if wt_seqs_csv is not None:
        wt_df = pd.read_csv(Path(VOLUME_PATH) / wt_seqs_csv)
        wt_lookup = dict(zip(wt_df["gene"].astype(str), wt_df["sequence"].astype(str)))

    out_root = Path(VOLUME_PATH) / out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    atlas_index = []
    for (gene, variant), sub in clin_df.groupby(["gene", "mutation_notation"]):
        seq = wt_lookup.get(str(gene))
        if seq is None:
            print(f"[atlas-skip] {gene}/{variant}: no WT sequence")
            continue
        candidates = enumerate_candidate_pairs_for_target(
            m1_notation=str(variant),
            m1_score=0.05,
            seq=seq,
            dataset_id=str(gene),
            forbidden_aas=("C", "M", "W", "P"),
        )
        if len(candidates) == 0:
            continue
        scores = model.predict_rescue_proba(candidates)
        candidates = candidates.copy()
        candidates["rescue_score"] = scores
        # Mark which alphas a candidate clears (use underscore-safe column suffixes).
        for a, cset in csets.items():
            suffix = f"a{int(round(a * 100)):02d}"
            candidates[f"in_alpha_{suffix}"] = scores >= cset.calibrated_threshold
        candidates = candidates.sort_values("rescue_score", ascending=False).head(int(max_top_k))
        # Emit per-variant JSON.
        record = {
            "gene": str(gene),
            "variant": str(variant),
            "n_candidates_total": int(len(scores)),
            "alphas": sorted(csets.keys()),
            "top_k": [
                {
                    "suppressor": f"{r.m2_notation}",
                    "position": int(r.m2_pos),
                    "mutant_aa": r.m2_aa,
                    "rescue_score": float(r.rescue_score),
                    **{
                        f"in_alpha_{int(round(a * 100)):02d}":
                            bool(getattr(r, f"in_alpha_a{int(round(a * 100)):02d}"))
                        for a in csets.keys()
                    },
                }
                for r in candidates.itertuples(index=False)
            ],
        }
        per_variant_path = out_root / f"{gene}_{variant}.json"
        per_variant_path.write_text(json.dumps(record, indent=2))
        atlas_index.append(
            {
                "gene": gene,
                "variant": variant,
                "json_path": str(per_variant_path.relative_to(Path(VOLUME_PATH))),
                "n_candidates_total": int(len(scores)),
                **{
                    f"n_in_alpha_{int(round(a * 100)):02d}": int(
                        candidates[f"in_alpha_a{int(round(a * 100)):02d}"].sum()
                    )
                    for a in csets.keys()
                },
            }
        )
    pd.DataFrame(atlas_index).to_csv(out_root / "atlas_index.csv", index=False)
    volume.commit()
    return {"out_dir": str(out_root), "n_atlas_records": len(atlas_index)}


# ---------------------------------------------------------------------------
# EditGuard-Clin: extract Megascale AF2 PDBs from HF Arrow cache to .pdb files
# ---------------------------------------------------------------------------

@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600 * 2)
def extract_megascale_pdbs(
    cache_dir: str = "data/megascale",
    out_dir: str = "data/megascale/pdbs",
) -> dict:
    """The HF Megascale dataset stores PDB strings inside an Arrow file.
    Extract them as actual .pdb files keyed by protein name (e.g. ``2MXD.pdb``).
    """
    from datasets import load_dataset

    ds = load_dataset(
        "RosettaCommons/MegaScale",
        "AlphaFold_model_PDBs",
        split="train",
        cache_dir=str(Path(VOLUME_PATH) / cache_dir),
    )
    out_root = Path(VOLUME_PATH) / out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    written = 0
    name_keys = [k for k in ds.column_names if k.lower() in ("name", "id", "wt_name", "pdb_id")]
    pdb_keys = [k for k in ds.column_names if k.lower() in ("pdb", "pdb_text", "structure", "pdb_string", "atomic_pdb")]
    print(f"[pdbs] columns={ds.column_names}; name_key={name_keys}; pdb_key={pdb_keys}")
    if not name_keys or not pdb_keys:
        return {"error": "could not infer PDB columns", "columns": ds.column_names}
    name_key, pdb_key = name_keys[0], pdb_keys[0]
    for row in ds:
        name = str(row[name_key])
        if not name.endswith(".pdb"):
            name = f"{name}.pdb"
        pdb_text = row[pdb_key]
        if pdb_text is None:
            continue
        (out_root / name).write_text(pdb_text)
        written += 1
    volume.commit()
    return {"out_dir": str(out_root), "n_written": written, "n_total_rows": int(len(ds))}


# ---------------------------------------------------------------------------
# EditGuard-Clin: Structural rescue GNN (the ICML-novelty add)
# ---------------------------------------------------------------------------

structural_gnn_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*CPU_PIP, *GPU_PIP)
    .pip_install("torch-scatter", extra_options=f"-f {PYG_WHEEL_INDEX}")
    .add_local_dir("src/phaseagent", remote_path="/root/phaseagent", copy=True)
)


@app.function(image=structural_gnn_image, volumes={VOLUME_PATH: volume}, gpu="A10G", timeout=3600 * 8)
def train_structural_rescue_gnn(
    events_path: str = "outputs/clin/rescue_events_megascale.parquet",
    structures_dir: str = "data/megascale/pdbs",
    chemistry_predictor_path: str = "outputs/clin/rescue_predictor.pkl",
    out_model: str = "outputs/clin/structural_rescue_gnn.pt",
    out_metrics: str = "outputs/clin/structural_rescue_gnn_metrics.csv",
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    train_frac: float = 0.7,
    seed: int = 0,
    max_train_events: int = 30000,
    max_test_events: int = 10000,
) -> dict:
    """Train the SE(3)-style structural rescue GNN on Megascale rescue events.

    For each rescue (m1, m2) pair we:
      1. Load the protein's AF2 PDB (cached on the volume from Megascale).
      2. Extract Cα coordinates and the local neighborhood of m1 + m2.
      3. Featurize: residue identity per node, distance/contact per edge,
         chemistry features for the (m1, m2) pair, geometric pair features.
      4. Predict P(rescue).

    Compares to the chemistry-only RF predictor on the same held-out proteins.
    """
    import pickle
    import time
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import average_precision_score, roc_auc_score

    from phaseagent.rescue_predictor import RescuePredictor, featurize_event_table
    from phaseagent.structural_rescue_gnn import (
        StructuralRescueConfig,
        build_protein_context,
        build_torch_gnn,
        local_neighborhood,
        pair_geometric_features,
        cb_distance_matrix,
    )
    from phaseagent.spectrum import AA_TO_ID

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gnn] device: {device}")

    # Load events.
    events = pd.read_parquet(Path(VOLUME_PATH) / events_path)
    rng = np.random.default_rng(int(seed))
    proteins = sorted(events["dataset_id"].unique())
    rng.shuffle(proteins)
    n_train = int(round(len(proteins) * train_frac))
    train_proteins = set(proteins[:n_train])
    test_proteins = set(proteins[n_train:])
    train_df = events[events["dataset_id"].isin(train_proteins)]
    test_df = events[events["dataset_id"].isin(test_proteins)]
    if max_train_events and len(train_df) > max_train_events:
        train_df = train_df.sample(max_train_events, random_state=int(seed)).reset_index(drop=True)
    if max_test_events and len(test_df) > max_test_events:
        test_df = test_df.sample(max_test_events, random_state=int(seed)).reset_index(drop=True)
    print(f"[gnn] train: {len(train_df)} rows / {len(train_proteins)} proteins; "
          f"test: {len(test_df)} rows / {len(test_proteins)} proteins")

    # Load structures lazily (one PDB per protein needed).
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    proteins_needed = train_df["dataset_id"].unique().tolist() + test_df["dataset_id"].unique().tolist()
    contexts: dict[str, object] = {}
    structures_path = Path(VOLUME_PATH) / structures_dir
    print(f"[gnn] loading structures from {structures_path}")
    for protein in set(proteins_needed):
        pdb_path = structures_path / f"{protein}"
        if not pdb_path.suffix:
            # Megascale stores files like "1A0N.pdb" — try with that suffix.
            pdb_path = structures_path / f"{protein}"
        if not pdb_path.exists():
            # Fallback: try matching the bare PDB-id as a stem
            cand = list(structures_path.glob(f"{protein.split('.')[0]}*"))
            if cand:
                pdb_path = cand[0]
        if not pdb_path.exists():
            continue
        try:
            structure = parser.get_structure(protein, str(pdb_path))
            atoms = []
            seq = []
            three_to_one = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q",
                            "GLU":"E","GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K",
                            "MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W",
                            "TYR":"Y","VAL":"V"}
            for residue in structure.get_residues():
                resname = residue.get_resname()
                if resname not in three_to_one:
                    continue
                if "CA" not in residue:
                    continue
                atoms.append(residue["CA"].coord)
                seq.append(three_to_one[resname])
            if len(atoms) < 5:
                continue
            coords = np.array(atoms, dtype=np.float32)
            contexts[protein] = build_protein_context(protein, coords, "".join(seq))
        except Exception as exc:
            print(f"[gnn] {protein} failed: {exc}")
            continue
    print(f"[gnn] loaded {len(contexts)} protein structural contexts")

    # Filter events to those with structures available.
    train_df = train_df[train_df["dataset_id"].isin(contexts.keys())].reset_index(drop=True)
    test_df = test_df[test_df["dataset_id"].isin(contexts.keys())].reset_index(drop=True)
    print(f"[gnn] after structure-filter: train {len(train_df)}, test {len(test_df)}")
    if len(train_df) < 100 or len(test_df) < 50:
        return {"error": "too few events with structures available", "train": len(train_df), "test": len(test_df)}

    # Pre-compute chemistry + geometric features per row.
    def _row_features(row, ctx):
        # The Megascale m1_pos / m2_pos are 1-indexed in the canonical sequence;
        # but the AF2 PDB is trimmed to measured boundaries — fall back to clipping.
        L = len(ctx.coords)
        m1 = max(0, min(int(row.m1_pos) - 1, L - 1))
        m2 = max(0, min(int(row.m2_pos) - 1, L - 1))
        geom = pair_geometric_features(ctx.coords, m1, m2)
        chem = featurize_event_table(pd.DataFrame([{
            "dataset_id": row.dataset_id,
            "m1_notation": row.m1_notation,
            "m2_notation": row.m2_notation,
            "m1_pos": int(row.m1_pos),
            "m2_pos": int(row.m2_pos),
            "m1_aa": row.m1_aa,
            "m2_aa": row.m2_aa,
            "m1_score": float(row.m1_score),
            "m2_score": float(getattr(row, "m2_score", float("nan"))),
        }]))[0]
        return chem.astype(np.float32), geom.astype(np.float32), m1, m2, ctx.sequence, ctx.coords, L

    cfg = StructuralRescueConfig(epochs=int(epochs), batch_size=int(batch_size), learning_rate=float(lr), seed=int(seed))
    model = build_torch_gnn(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    def _make_batch(df_chunk):
        """Assemble PyG-style batch from a chunk of rescue rows."""
        m1_node_ids_list, m2_node_ids_list = [], []
        m1_edge_index_list, m2_edge_index_list = [], []
        m1_edge_attr_list, m2_edge_attr_list = [], []
        m1_batch_assign, m2_batch_assign = [], []
        chem_list, geom_list, label_list = [], [], []
        offset_m1, offset_m2 = 0, 0
        for b_idx, (_, row) in enumerate(df_chunk.iterrows()):
            ctx = contexts[row["dataset_id"]]
            try:
                chem, geom, m1, m2, seq, coords, L = _row_features(row, ctx)
            except Exception:
                continue
            n1 = local_neighborhood(coords, m1, cfg.pair_neighborhood_radius_A, cfg.max_neighbors_per_residue)
            n2 = local_neighborhood(coords, m2, cfg.pair_neighborhood_radius_A, cfg.max_neighbors_per_residue)
            for nbrs, edge_index_list, edge_attr_list, node_ids_list, batch_assign, offset in (
                (n1, m1_edge_index_list, m1_edge_attr_list, m1_node_ids_list, m1_batch_assign, offset_m1),
                (n2, m2_edge_index_list, m2_edge_attr_list, m2_node_ids_list, m2_batch_assign, offset_m2),
            ):
                node_ids = np.array(
                    [AA_TO_ID.get(seq[i], 20) for i in nbrs], dtype=np.int64
                )
                node_ids_list.append(node_ids)
                # Build complete edges within neighborhood (small graph).
                ii, jj = np.meshgrid(np.arange(len(nbrs)), np.arange(len(nbrs)), indexing="ij")
                mask = ii != jj
                src = ii[mask].flatten() + offset
                dst = jj[mask].flatten() + offset
                edge_index_list.append(np.stack([src, dst], axis=0))
                # Edge features: distance, log-dist, contact-8, contact-4, sep, sep<=4, dummy, dummy
                d = np.linalg.norm(
                    coords[nbrs[ii[mask].flatten()]] - coords[nbrs[jj[mask].flatten()]], axis=-1
                )
                seq_sep = np.abs(nbrs[ii[mask].flatten()] - nbrs[jj[mask].flatten()])
                ef = np.stack([
                    d, np.log1p(d), (d <= 8.0).astype(np.float32), (d <= 4.0).astype(np.float32),
                    seq_sep, (seq_sep <= 4).astype(np.float32),
                    np.zeros_like(d), np.zeros_like(d),
                ], axis=-1).astype(np.float32)
                edge_attr_list.append(ef)
                batch_assign.append(np.full(len(nbrs), b_idx, dtype=np.int64))
            offset_m1 += len(n1)
            offset_m2 += len(n2)
            chem_list.append(chem)
            geom_list.append(geom)
            label_list.append(int(row["rescue_class"] in ("partial", "full", "super")))
        if not chem_list:
            return None
        out = {
            "m1_node_ids": torch.from_numpy(np.concatenate(m1_node_ids_list)).to(device),
            "m2_node_ids": torch.from_numpy(np.concatenate(m2_node_ids_list)).to(device),
            "m1_edge_index": torch.from_numpy(np.concatenate(m1_edge_index_list, axis=1)).to(device),
            "m2_edge_index": torch.from_numpy(np.concatenate(m2_edge_index_list, axis=1)).to(device),
            "m1_edge_attr": torch.from_numpy(np.concatenate(m1_edge_attr_list, axis=0)).to(device),
            "m2_edge_attr": torch.from_numpy(np.concatenate(m2_edge_attr_list, axis=0)).to(device),
            "m1_batch_assign": torch.from_numpy(np.concatenate(m1_batch_assign)).to(device),
            "m2_batch_assign": torch.from_numpy(np.concatenate(m2_batch_assign)).to(device),
            "chem": torch.from_numpy(np.stack(chem_list)).to(device),
            "geom": torch.from_numpy(np.stack(geom_list)).to(device),
            "labels": torch.tensor(label_list, dtype=torch.float32, device=device),
        }
        return out

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    metrics_rows = []
    for epoch in range(int(epochs)):
        model.train()
        perm = np.random.permutation(len(train_df))
        train_loss = 0.0
        n_seen = 0
        t0 = time.time()
        for start in range(0, len(perm), int(batch_size)):
            chunk = train_df.iloc[perm[start : start + int(batch_size)]]
            batch = _make_batch(chunk)
            if batch is None:
                continue
            optim.zero_grad()
            logits = model(batch)
            loss = F.binary_cross_entropy_with_logits(logits, batch["labels"])
            loss.backward()
            optim.step()
            train_loss += float(loss.item()) * len(chunk)
            n_seen += len(chunk)
        # Eval on test.
        model.eval()
        all_logits, all_labels = [], []
        with torch.inference_mode():
            for start in range(0, len(test_df), int(batch_size)):
                chunk = test_df.iloc[start : start + int(batch_size)]
                batch = _make_batch(chunk)
                if batch is None:
                    continue
                logits = model(batch)
                all_logits.append(logits.cpu().numpy())
                all_labels.append(batch["labels"].cpu().numpy())
        if all_logits:
            scores = 1.0 / (1.0 + np.exp(-np.concatenate(all_logits)))
            labels = np.concatenate(all_labels)
            try:
                auroc = float(roc_auc_score(labels, scores))
                auprc = float(average_precision_score(labels, scores))
            except Exception:
                auroc, auprc = float("nan"), float("nan")
        else:
            auroc, auprc = float("nan"), float("nan")
        elapsed = time.time() - t0
        avg_loss = train_loss / max(n_seen, 1)
        metrics_rows.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "test_auroc": auroc,
            "test_auprc": auprc,
            "n_test": int(sum(len(a) for a in all_labels)),
            "elapsed_sec": float(elapsed),
        })
        print(f"[gnn] epoch {epoch:>3d} loss={avg_loss:.4f}  test AUROC={auroc:.3f}  AUPRC={auprc:.3f}  ({elapsed:.1f}s)")

    # Compare to chemistry-RF baseline on the same test set.
    with open(Path(VOLUME_PATH) / chemistry_predictor_path, "rb") as f:
        bundle = pickle.load(f)
    chem_model: RescuePredictor = bundle["model"]
    chem_scores = chem_model.predict_rescue_proba(test_df)
    chem_labels = test_df["rescue_class"].isin(("partial", "full", "super")).astype(int).to_numpy()
    chem_auroc = float(roc_auc_score(chem_labels, chem_scores))
    chem_auprc = float(average_precision_score(chem_labels, chem_scores))

    out_metrics_path = Path(VOLUME_PATH) / out_metrics
    out_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_metrics_path, index=False)
    # Save model weights.
    out_model_path = Path(VOLUME_PATH) / out_model
    torch.save({
        "state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "train_proteins": sorted(train_proteins),
        "test_proteins": sorted(test_proteins),
        "chemistry_baseline_auroc": chem_auroc,
        "chemistry_baseline_auprc": chem_auprc,
    }, out_model_path)
    volume.commit()
    final_auroc = float(metrics_df["test_auroc"].iloc[-1]) if len(metrics_df) else float("nan")
    return {
        "model": str(out_model_path),
        "metrics": str(out_metrics_path),
        "final_test_auroc": final_auroc,
        "chemistry_baseline_auroc": chem_auroc,
        "structural_lift": final_auroc - chem_auroc,
    }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Theorem 1 validation: spectrum -> phase boundary
# ---------------------------------------------------------------------------

@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600 * 4)
def validate_phase_theorem_megascale(
    cache_dir: str = "data/megascale",
    out: str = "outputs/phase/megascale_theorem_validation.csv",
    delta_g_kcal: float = 1.0,
    n_mc_samples: int = 200_000,
) -> dict:
    """Per-protein Theorem 1 validation on Megascale singles + doubles.

    For every protein with at least one single and one double mutant
    measurement in dataset3_single + dataset2 respectively, compute the
    spectrum stats (m_g, sigma_g) from raw ddG_ML singles, predict V(2)
    via Berry-Esseen and Monte-Carlo, and compare to observed V(2) on
    the doubles. Output a row per protein.
    """
    import pandas as pd
    from phaseagent.megascale import (
        MegascaleConfig, load_megascale, load_megascale_double_mutant_test,
    )
    from phaseagent.phase_validation import validate_megascale_protein

    # Singles: combine train+val+test for max sample size per protein.
    print("[phase] loading Megascale singles...")
    parts = []
    for split in ("train", "val", "test"):
        cfg = MegascaleConfig(config_name="dataset3_single", split=split, cache_dir=cache_dir)
        sub = load_megascale(cfg)
        sub["split"] = split
        parts.append(sub)
    singles = pd.concat(parts, axis=0, ignore_index=True)
    print(f"[phase] singles n={len(singles)} proteins={singles['dataset_id'].nunique()}")

    # Doubles
    print("[phase] loading Megascale doubles...")
    doubles = load_megascale_double_mutant_test(cache_dir=cache_dir)
    print(f"[phase] doubles n={len(doubles)} proteins={doubles['dataset_id'].nunique()}")

    # Need raw ddG_ML on both. Re-add it from the underlying datasets if we
    # dropped it in the schema mapping.
    if "ddG_ML" not in singles.columns:
        # Reload with raw label preserved.
        from datasets import load_dataset
        ds = load_dataset(
            "RosettaCommons/MegaScale", "dataset3_single",
            cache_dir=cache_dir,
        )
        raw = pd.concat([ds[s].to_pandas() for s in ds.keys()], ignore_index=True)
        # Re-key by WT_name + mutant
        if "WT_name" in raw.columns and "mutant" in raw.columns:
            singles = singles.merge(
                raw[["WT_name", "mutant", "ddG_ML"]].rename(
                    columns={"WT_name": "dataset_id", "mutant": "mutation_notation"}
                ),
                on=["dataset_id", "mutation_notation"], how="left",
            )

    if "ddG_ML" not in doubles.columns:
        from datasets import load_dataset
        ds2 = load_dataset(
            "RosettaCommons/MegaScale", "dataset2",
            split="train",
            cache_dir=cache_dir,
        )
        raw2 = ds2.to_pandas()
        # The double-mutant column might be 'mut_type'.
        for cand in ("mut_type", "mutant", "mutation"):
            if cand in raw2.columns:
                raw2 = raw2.rename(columns={cand: "mutation_notation"})
                break
        if "WT_name" in raw2.columns:
            raw2 = raw2.rename(columns={"WT_name": "dataset_id"})
        doubles = doubles.merge(
            raw2[["dataset_id", "mutation_notation", "ddG_ML"]],
            on=["dataset_id", "mutation_notation"], how="left",
        )

    proteins = sorted(set(singles["dataset_id"].astype(str)) & set(doubles["dataset_id"].astype(str)))
    print(f"[phase] {len(proteins)} proteins with both singles + doubles")

    rows = []
    for i, protein in enumerate(proteins):
        s = singles[singles["dataset_id"].astype(str) == protein]
        d = doubles[doubles["dataset_id"].astype(str) == protein]
        if len(s) < 30 or len(d) < 5:
            continue
        result = validate_megascale_protein(
            singles_df=s,
            doubles_df=d,
            raw_label_col="ddG_ML",
            delta_g_kcal=float(delta_g_kcal),
            n_mc_samples=int(n_mc_samples),
            seed=int(i),
        )
        result["protein"] = protein
        rows.append(result)
        if (i + 1) % 20 == 0:
            print(f"[phase] {i+1}/{len(proteins)} proteins processed")

    out_df = pd.DataFrame(rows)
    out_path = Path(VOLUME_PATH) / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    volume.commit()
    print(f"[phase] wrote {len(out_df)} rows to {out_path}")
    return {"out": str(out_path), "n_proteins": len(out_df)}


@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600 * 4)
def validate_phase_theorem_proteingym(
    data: str = "processed/all_dms.parquet",
    splits_csv: str = "outputs/editguard/edit_splits.csv",
    out: str = "outputs/phase/proteingym_theorem_validation.csv",
    out_curves: str = "outputs/phase/proteingym_V_curves.csv",
    quantile_threshold: float = 0.50,
    n_mc_samples: int = 200_000,
    min_singles: int = 100,
    min_max_distance: int = 3,
) -> dict:
    """Per-assay Theorem 1 validation on ProteinGym multi-mutant assays.

    Strategy: for every assay with at least ``min_singles`` single mutants
    and a max edit distance >= ``min_max_distance``, fit V(d) sigmoid on
    observed multi-distance data, then predict (d_c, alpha) from spectrum
    of d=1 only. Compare.
    """
    import pandas as pd
    from phaseagent.phase_validation import validate_proteingym_assay

    print("[phase] loading ProteinGym all_dms parquet...")
    df = pd.read_parquet(Path(VOLUME_PATH) / data)
    print(f"[phase] all_dms n={len(df)} assays={df['dataset_id'].nunique()}")
    print(f"[phase] columns: {list(df.columns)}")

    # Theorem 1 requires additive structure on raw fitness. Use raw column
    # (DMS_score / fitness_raw) NOT rank-normalized.
    if "DMS_score" in df.columns:
        fitness_col = "DMS_score"
    elif "fitness_raw" in df.columns:
        fitness_col = "fitness_raw"
    else:
        fitness_col = "fitness_norm"
    print(f"[phase] using fitness column: {fitness_col}")

    rows = []
    curve_rows = []
    for assay, sub in df.groupby("dataset_id"):
        if "mutation_distance" not in sub.columns:
            continue
        max_d = int(pd.to_numeric(sub["mutation_distance"], errors="coerce").max() or 0)
        if max_d < int(min_max_distance):
            continue
        n_singles = int((sub["mutation_distance"] == 1).sum())
        if n_singles < int(min_singles):
            continue
        result = validate_proteingym_assay(
            sub,
            fitness_col=fitness_col,
            quantile_threshold=float(quantile_threshold),
            n_mc_samples=int(n_mc_samples),
            seed=hash(str(assay)) & 0xFFFFFFFF,
        )
        if "error" in result:
            continue
        curve = result.pop("_curve")
        curve["dataset_id"] = str(assay)
        curve_rows.append(curve)
        result["dataset_id"] = str(assay)
        rows.append(result)

    out_df = pd.DataFrame(rows)
    out_path = Path(VOLUME_PATH) / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    if curve_rows:
        curve_all = pd.concat(curve_rows, axis=0, ignore_index=True)
        curve_path = Path(VOLUME_PATH) / out_curves
        curve_all.to_csv(curve_path, index=False)
    volume.commit()
    print(f"[phase] wrote {len(out_df)} assays to {out_path}")
    return {"out": str(out_path), "n_assays": len(out_df), "out_curves": out_curves}


# ---------------------------------------------------------------------------
# Phase-calibrated SMC: real run with phase prior in the reward
# ---------------------------------------------------------------------------

@app.function(image=gpu_image, volumes={VOLUME_PATH: volume}, gpu="A10G", timeout=3600 * 8)
def run_phase_calibrated_smc(
    data: str = "processed/all_dms.parquet",
    tasks_csv: str = "outputs/editguard/edit_tasks.csv",
    splits_csv: str = "outputs/editguard/edit_splits.csv",
    prior_path: str = "outputs/editguard/editguard_prior.pkl",
    out: str = "outputs/phase/phase_smc_metrics.csv",
    n_particles: int = 8,
    n_denoise_steps: int = 64,
    n_outer_iterations: int = 2,
    base_beta: float = 1.0,
    phase_betas: str = "0.0,0.5,1.0,2.0",  # 0.0 = ablation = no phase term
    seeds: int = 3,
    max_tasks: int = 0,
    quantile_threshold: float = 0.50,
) -> dict:
    """Phase-calibrated Twisted SMC over DPLM-650M.

    Sweeps ``phase_beta`` over the supplied list. ``phase_beta=0.0``
    reproduces the standard fixed-temperature Twisted SMC baseline (B2 in
    the paper). Larger values increasingly weight the LDP phase prior.
    """
    import numpy as np
    import pandas as pd

    from phaseagent.edit_generators import infer_wildtype_sequence
    from phaseagent.edit_splits import assert_tasks_in_split
    from phaseagent.editguard_prior import DMSFunctionPrior
    from phaseagent.editing_tasks import task_from_row
    from phaseagent.twisted_smc import TwistedSMCConfig, run_twisted_smc, best_particle
    from phaseagent.twisted_smc_dplm import (
        DMSPriorRewardModel, RealDPLMBackbone, build_phase_reward_from_singles,
    )

    df = pd.read_parquet(Path(VOLUME_PATH) / data)
    tasks = pd.read_csv(Path(VOLUME_PATH) / tasks_csv)
    splits = pd.read_csv(Path(VOLUME_PATH) / splits_csv)
    test_ds = set(splits.loc[splits["split"] == "test", "dataset_id"].astype(str))
    tasks = tasks[tasks["dataset_id"].astype(str).isin(test_ds)].reset_index(drop=True)
    assert_tasks_in_split(tasks, splits, "test")
    if max_tasks > 0:
        tasks = tasks.head(max_tasks)
    prior = DMSFunctionPrior.load(Path(VOLUME_PATH) / prior_path)
    fitness_col = "DMS_score" if "DMS_score" in df.columns else "fitness_norm"

    backbone = RealDPLMBackbone()
    beta_list = [float(x) for x in str(phase_betas).split(",") if x.strip()]
    rows = []
    for task_idx, row in tasks.iterrows():
        task = task_from_row(row)
        ds_sub = df[df["dataset_id"] == task.dataset_id]
        if len(ds_sub) == 0:
            continue
        wt_seq = infer_wildtype_sequence(ds_sub, task.dataset_id)
        if not wt_seq:
            continue
        # Compute the per-protein phase prior from observed singles only.
        singles = ds_sub[ds_sub["mutation_distance"] == 1]
        if len(singles) < 30:
            print(f"[phase-smc] skipping {task.dataset_id}: only {len(singles)} singles")
            continue
        wt_fitness = float(singles[fitness_col].quantile(0.95))
        threshold = float(singles[fitness_col].quantile(quantile_threshold))
        delta_g = float(wt_fitness - threshold)
        if delta_g <= 0:
            print(f"[phase-smc] skipping {task.dataset_id}: delta_g={delta_g:.3f} <= 0")
            continue
        damage = (wt_fitness - singles[fitness_col].astype(float)).to_numpy()
        damage = damage[np.isfinite(damage)]

        forbidden = frozenset(task.forbidden_residues)
        backbone.forbidden_aas = forbidden
        base_reward = DMSPriorRewardModel(
            prior=prior, dataset_id=task.dataset_id, wt_sequence=wt_seq,
        )
        for phase_beta in beta_list:
            phase_reward = build_phase_reward_from_singles(
                base_reward=base_reward,
                wt_sequence=wt_seq,
                single_effects=damage,
                delta_g=delta_g,
                beta=float(phase_beta),
                use_mc=True,
            )
            for seed in range(seeds):
                cfg = TwistedSMCConfig(
                    n_particles=int(n_particles),
                    n_denoise_steps=int(n_denoise_steps),
                    n_outer_iterations=int(n_outer_iterations),
                    beta=float(base_beta),
                    seed=int(seed),
                )
                tag = f"phase_smc_dplm_phaseB{phase_beta}_K{n_particles}"
                print(f"[phase-smc] task={task_idx} ds={task.dataset_id} "
                      f"obj={task.objective} k={task.edit_budget} "
                      f"phaseB={phase_beta} seed={seed} ...")
                result = run_twisted_smc(
                    wt_seq=wt_seq,
                    edit_budget=int(task.edit_budget),
                    backbone=backbone,
                    reward=phase_reward,
                    config=cfg,
                )
                best = best_particle(result)
                # Compute distance to WT and base reward separately for analysis
                best_seq = best.sequence
                best_d = int(sum(1 for x, y in zip(best_seq, wt_seq) if x != y))
                base_r = float(base_reward.reward([best_seq])[0])
                rows.append(
                    {
                        "method": tag,
                        "phase_beta": float(phase_beta),
                        "task_idx": int(task_idx),
                        "dataset_id": task.dataset_id,
                        "objective": task.objective,
                        "edit_budget": int(task.edit_budget),
                        "seed": int(seed),
                        "best_total_reward": float(np.max(result["rewards"])),
                        "mean_total_reward": float(np.mean(result["rewards"])),
                        "best_base_reward": base_r,
                        "best_hamming": best_d,
                        "delta_g": float(delta_g),
                        "m_g": float(np.mean(damage)),
                        "sigma_g": float(np.std(damage, ddof=1) if len(damage) > 1 else 0.0),
                        "best_sequence": best_seq,
                    }
                )
    out_path = Path(VOLUME_PATH) / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    volume.commit()
    return {"out": str(out_path), "rows": len(rows)}


# ---------------------------------------------------------------------------
# DMS-conditioned protein diffusion: trained context adapter + same-task eval
# ---------------------------------------------------------------------------

@app.function(image=gpu_image, volumes={VOLUME_PATH: volume}, gpu="A10G", timeout=3600 * 12)
def train_dms_context_adapter(
    data: str = "processed/all_dms.parquet",
    splits_csv: str = "outputs/editguard/edit_splits.csv",
    out_model: str = "outputs/context_generation/dms_context_adapter.pt",
    out_trace: str = "outputs/context_generation/dms_context_adapter_train_trace.csv",
    fitness_col: str = "fitness_norm",
    min_distance: int = 2,
    max_distance: int = 8,
    max_train_variants: int = 50000,
    epochs: int = 2,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    pairwise_weight: float = 0.10,
    seed: int = 0,
) -> dict:
    """Train a frozen-DPLM logit adapter conditioned on single-mutant DMS maps."""
    import numpy as np
    import pandas as pd
    import torch

    from phaseagent.dms_context_adapter import (
        DMSContextAdapter,
        DMSContextAdapterConfig,
        adapter_loss,
        make_training_frame,
        parse_variant_target,
        save_adapter,
    )
    from phaseagent.dplm_backbone import _allowed_aa_token_ids, _build_masked_sequence, _load_dplm
    from phaseagent.edit_generators import infer_wildtype_sequence
    from phaseagent.mutation_map_context import build_mutation_map_context
    from phaseagent.mutations import parse_mutation_notation

    rng = np.random.default_rng(seed)
    root = Path(VOLUME_PATH)
    df = pd.read_parquet(root / data)
    if fitness_col not in df.columns:
        fitness_col = "DMS_score" if "DMS_score" in df.columns else fitness_col
    splits_path = root / splits_csv
    if splits_path.exists() and "dataset_id" in df.columns:
        splits = pd.read_csv(splits_path)
        train_ids = set(splits.loc[splits["split"] == "train", "dataset_id"].astype(str))
        df = df[df["dataset_id"].astype(str).isin(train_ids)].reset_index(drop=True)

    rows = make_training_frame(
        df,
        min_distance=int(min_distance),
        max_distance=int(max_distance),
        fitness_col=fitness_col,
    )
    if len(rows) == 0:
        raise RuntimeError("no multi-mutant rows available for DMS-context adapter training")
    if max_train_variants > 0 and len(rows) > max_train_variants:
        rows = rows.sample(int(max_train_variants), random_state=seed).reset_index(drop=True)

    contexts = {}
    wildtypes = {}
    usable_ids = []
    for ds, sub in df.groupby("dataset_id"):
        ds = str(ds)
        wt = infer_wildtype_sequence(sub, ds)
        if not wt:
            continue
        singles = sub[sub["mutation_notation"].apply(lambda x: len(parse_mutation_notation(x)) == 1)].copy()
        if "mutation_distance" in singles.columns:
            singles = singles[singles["mutation_distance"] == 1].copy()
        if len(singles) < 20:
            continue
        try:
            contexts[ds] = build_mutation_map_context(
                singles,
                dataset_id=ds,
                wildtype_sequence=wt,
                fitness_col=fitness_col,
                strict_singles=True,
            )
        except Exception as exc:
            print(f"[context-adapter] skip context {ds}: {exc}")
            continue
        wildtypes[ds] = wt
        usable_ids.append(ds)
    rows = rows[rows["dataset_id"].astype(str).isin(usable_ids)].reset_index(drop=True)
    if len(rows) == 0:
        raise RuntimeError("no training rows remain after building single-mutant contexts")

    model, tokenizer = _load_dplm()
    for param in model.parameters():
        param.requires_grad_(False)
    allowed_token_ids, allowed_aas = _allowed_aa_token_ids(tokenizer)
    if tuple(allowed_aas) != tuple("ACDEFGHIKLMNPQRSTVWY"):
        raise RuntimeError("unexpected tokenizer amino-acid order")
    device = next(model.parameters()).device
    adapter = DMSContextAdapter(
        DMSContextAdapterConfig(hidden_dim=int(model.config.hidden_size))
    ).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=float(learning_rate), weight_decay=1e-4)

    trace = []
    row_indices = np.arange(len(rows))
    for epoch in range(int(epochs)):
        rng.shuffle(row_indices)
        running = []
        for batch_start in range(0, len(row_indices), int(batch_size)):
            batch_idx = row_indices[batch_start : batch_start + int(batch_size)]
            batch = rows.iloc[batch_idx].reset_index(drop=True)
            masked_sequences = []
            parsed_targets = []
            variant_fitness = []
            for _, row in batch.iterrows():
                ds = str(row["dataset_id"])
                wt = wildtypes.get(ds)
                if wt is None or ds not in contexts:
                    continue
                target = parse_variant_target(str(row["mutation_notation"]), wt)
                if target is None:
                    continue
                masked_sequences.append(_build_masked_sequence(wt, target.positions))
                parsed_targets.append((ds, target))
                variant_fitness.append(float(row[fitness_col]))
            if not masked_sequences:
                continue
            inputs = tokenizer(masked_sequences, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
                logits = out.logits.detach()
                hidden = out.hidden_states[-1].detach()

            h_list, base_list, ctx_list, target_list, budget_list, weight_list = [], [], [], [], [], []
            offsets = []
            token_cursor = 0
            for b, ((ds, target), fit) in enumerate(zip(parsed_targets, variant_fitness)):
                ctx = contexts[ds]
                n_tok = len(target.positions)
                offsets.append((token_cursor, token_cursor + n_tok))
                token_cursor += n_tok
                for pos, aa_id in zip(target.positions, target.aa_ids):
                    h_list.append(hidden[b, int(pos), :])
                    base_list.append(logits[b, int(pos), allowed_token_ids])
                    ctx_list.append(ctx.features_for_positions([int(pos)])[0])
                    target_list.append(int(aa_id))
                    budget_list.append(float(len(target.positions)))
                    weight_list.append(0.25 + max(float(fit), 0.0))
            if not h_list:
                continue

            hidden_t = torch.stack(h_list).to(device)
            base_t = torch.stack(base_list).to(device)
            ctx_t = torch.tensor(np.stack(ctx_list), dtype=torch.float32, device=device)
            target_t = torch.tensor(target_list, dtype=torch.long, device=device)
            budget_t = torch.tensor(budget_list, dtype=torch.float32, device=device)
            weight_t = torch.tensor(weight_list, dtype=torch.float32, device=device)

            adapter.train()
            opt.zero_grad(set_to_none=True)
            losses = adapter_loss(
                adapter,
                base_aa_logits=base_t,
                hidden_states=hidden_t,
                context_features=ctx_t,
                target_aa_ids=target_t,
                edit_budgets=budget_t,
                token_weights=weight_t,
                variant_offsets=offsets,
                variant_fitness=np.asarray(variant_fitness, dtype=np.float32),
                pairwise_weight=float(pairwise_weight),
            )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            opt.step()
            loss_val = float(losses["loss"].detach().cpu())
            running.append(loss_val)
            trace.append(
                {
                    "epoch": int(epoch),
                    "batch_start": int(batch_start),
                    "loss": loss_val,
                    "nll_loss": float(losses["nll_loss"].detach().cpu()),
                    "rank_loss": float(losses["rank_loss"].detach().cpu()),
                    "n_variants": int(len(parsed_targets)),
                    "n_tokens": int(len(target_list)),
                }
            )
        print(f"[context-adapter] epoch={epoch} mean_loss={np.mean(running) if running else np.nan:.4f}")

    out_model_path = root / out_model
    out_trace_path = root / out_trace
    out_trace_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trace).to_csv(out_trace_path, index=False)
    save_adapter(
        adapter,
        out_model_path,
        metadata={
            "data": data,
            "fitness_col": fitness_col,
            "n_train_variants": int(len(rows)),
            "n_contexts": int(len(contexts)),
            "min_distance": int(min_distance),
            "max_distance": int(max_distance),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "pairwise_weight": float(pairwise_weight),
            "seed": int(seed),
        },
    )
    volume.commit()
    return {
        "out_model": str(out_model_path),
        "out_trace": str(out_trace_path),
        "n_train_variants": int(len(rows)),
        "n_contexts": int(len(contexts)),
        "last_loss": float(trace[-1]["loss"]) if trace else None,
    }


@app.function(image=gpu_image, volumes={VOLUME_PATH: volume}, gpu="A10G", timeout=3600 * 12)
def run_dms_context_generation(
    data: str = "processed/all_dms.parquet",
    tasks_csv: str = "outputs/editguard/edit_tasks.csv",
    splits_csv: str = "outputs/editguard/edit_splits.csv",
    prior_path: str = "outputs/editguard/editguard_prior.pkl",
    adapter_path: str = "outputs/context_generation/dms_context_adapter.pt",
    out_metrics: str = "outputs/context_generation/context_generation_metrics.csv",
    out_selections: str = "outputs/context_generation/context_generation_selections.parquet",
    fitness_col: str = "fitness_norm",
    n_mask_patterns: int = 16,
    samples_per_pattern: int = 16,
    top_k: int = 50,
    seeds: int = 3,
    max_tasks: int = 0,
) -> dict:
    """Evaluate DMS-context generation and same-task baselines on held-out tasks."""
    import pandas as pd
    import torch

    from phaseagent.context_generation import ContextGenerationConfig, benchmark_context_generation_task
    from phaseagent.dms_context_adapter import load_adapter
    from phaseagent.dplm_backbone import _load_dplm
    from phaseagent.edit_generators import infer_wildtype_sequence
    from phaseagent.edit_splits import assert_tasks_in_split
    from phaseagent.editing_tasks import task_from_row
    from phaseagent.editguard_prior import DMSFunctionPrior
    from phaseagent.mutation_map_context import build_mutation_map_context
    from phaseagent.mutations import parse_mutation_notation

    root = Path(VOLUME_PATH)
    df = pd.read_parquet(root / data)
    if fitness_col not in df.columns:
        fitness_col = "DMS_score" if "DMS_score" in df.columns else fitness_col
    tasks = pd.read_csv(root / tasks_csv)
    splits_path = root / splits_csv
    if splits_path.exists():
        splits = pd.read_csv(splits_path)
        test_ids = set(splits.loc[splits["split"] == "test", "dataset_id"].astype(str))
        tasks = tasks[tasks["dataset_id"].astype(str).isin(test_ids)].reset_index(drop=True)
        assert_tasks_in_split(tasks, splits, "test")
    if max_tasks > 0:
        tasks = tasks.head(int(max_tasks)).reset_index(drop=True)

    prior = DMSFunctionPrior.load(root / prior_path) if (root / prior_path).exists() else None
    model, tokenizer = _load_dplm()
    adapter = load_adapter(root / adapter_path, map_location="cpu")
    adapter = adapter.to(next(model.parameters()).device).eval()

    metric_frames = []
    selection_frames = []
    for task_idx, task_row in tasks.iterrows():
        task = task_from_row(task_row)
        sub = df[df["dataset_id"].astype(str) == task.dataset_id].copy()
        if len(sub) == 0:
            continue
        wt = infer_wildtype_sequence(sub, task.dataset_id)
        if not wt:
            continue
        singles = sub[sub["mutation_notation"].apply(lambda x: len(parse_mutation_notation(x)) == 1)].copy()
        if "mutation_distance" in singles.columns:
            singles = singles[singles["mutation_distance"] == 1].copy()
        if len(singles) < 20:
            print(f"[context-generation] skip {task.dataset_id}: only {len(singles)} singles")
            continue
        try:
            ctx = build_mutation_map_context(
                singles,
                dataset_id=task.dataset_id,
                wildtype_sequence=wt,
                fitness_col=fitness_col,
                strict_singles=True,
            )
        except Exception as exc:
            print(f"[context-generation] skip {task.dataset_id}: {exc}")
            continue
        for seed in range(int(seeds)):
            cfg = ContextGenerationConfig(
                n_mask_patterns=int(n_mask_patterns),
                samples_per_pattern=int(samples_per_pattern),
                top_k=int(top_k),
                seed=int(seed),
            )
            print(
                f"[context-generation] task={task_idx} ds={task.dataset_id} "
                f"obj={task.objective} B={task.edit_budget} seed={seed}"
            )
            with torch.inference_mode():
                metrics, selections = benchmark_context_generation_task(
                    dms_df=df,
                    task=task,
                    wt_sequence=wt,
                    model=model,
                    tokenizer=tokenizer,
                    adapter=adapter,
                    context=ctx,
                    prior=prior,
                    config=cfg,
                )
            metrics["task_idx"] = int(task_idx)
            selections["task_idx"] = int(task_idx)
            metric_frames.append(metrics)
            selection_frames.append(selections)

    metrics_out = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    selections_out = pd.concat(selection_frames, ignore_index=True) if selection_frames else pd.DataFrame()
    out_metrics_path = root / out_metrics
    out_sel_path = root / out_selections
    out_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.to_csv(out_metrics_path, index=False)
    if len(selections_out):
        selections_out.to_parquet(out_sel_path, index=False)
    volume.commit()
    return {
        "metrics": str(out_metrics_path),
        "selections": str(out_sel_path),
        "n_metric_rows": int(len(metrics_out)),
        "n_selection_rows": int(len(selections_out)),
    }


@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600 * 2)
def build_context_generation_figures(
    metrics_csv: str = "outputs/context_generation/context_generation_metrics.csv",
    out_dir: str = "outputs/context_generation/figures",
) -> dict:
    """Build first-pass figures for the experimental-context generation paper."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    root = Path(VOLUME_PATH)
    metrics_path = root / metrics_csv
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    df = pd.read_csv(metrics_path)
    out_root = root / out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    if len(df) == 0:
        return {"figures": [], "reason": "empty metrics"}
    metric_col = "functional_hit_rate_labeled" if "functional_hit_rate_labeled" in df.columns else "functional_hit_rate"
    summary = (
        df.groupby("method")[metric_col]
        .agg(["mean", "sem", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    summary.to_csv(out_root / "context_generation_leaderboard.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    x = np.arange(len(summary))
    ax.bar(x, summary["mean"], yerr=summary["sem"].fillna(0), color="#4C78A8", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["method"], rotation=35, ha="right")
    ax.set_ylabel("held-out labeled hit rate")
    ax.set_title("DMS-conditioned generation benchmark")
    fig.tight_layout()
    fig.savefig(out_root / "fig3_context_generation_leaderboard.pdf")
    fig.savefig(out_root / "fig3_context_generation_leaderboard.png", dpi=300)
    plt.close(fig)

    if "edit_budget" in df.columns:
        pivot = df.groupby(["method", "edit_budget"])[metric_col].mean().reset_index()
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        for method, sub in pivot.groupby("method"):
            ax.plot(sub["edit_budget"], sub[metric_col], marker="o", label=method)
        ax.set_xlabel("edit budget")
        ax.set_ylabel("held-out labeled hit rate")
        ax.legend(fontsize=6, frameon=False)
        fig.tight_layout()
        fig.savefig(out_root / "fig5_context_by_budget.pdf")
        fig.savefig(out_root / "fig5_context_by_budget.png", dpi=300)
        plt.close(fig)
    volume.commit()
    return {"figures": [str(out_root / "fig3_context_generation_leaderboard.pdf")]}


@app.function(image=gpu_image, volumes={VOLUME_PATH: volume}, gpu="A10G", timeout=3600 * 14)
def run_context_generation_benchmark(
    train_first: bool = True,
    max_train_variants: int = 50000,
    epochs: int = 2,
    max_tasks: int = 0,
    seeds: int = 3,
) -> dict:
    """Convenience marker for the two-step benchmark workflow."""
    return {
        "status": "use two-step execution for separately logged jobs",
        "commands": [
            "modal run modal_app_v2.py::train_dms_context_adapter",
            "modal run modal_app_v2.py::run_dms_context_generation",
            "modal run modal_app_v2.py::build_context_generation_figures",
        ],
        "train_first": bool(train_first),
        "max_train_variants": int(max_train_variants),
        "epochs": int(epochs),
        "max_tasks": int(max_tasks),
        "seeds": int(seeds),
    }


# ---------------------------------------------------------------------------
# Epistasis program: additive / specific-epistasis decomposition + atlas (gate)
# ---------------------------------------------------------------------------

@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600 * 2)
def run_epistasis_atlas(
    cache_dir: str = "data/megascale",
    config_name: str = "dataset2",
    split: str = "train",
    out_atlas: str = "outputs/epistasis/atlas.parquet",
    out_decomposed: str = "outputs/epistasis/decomposed.parquet",
    out_summary: str = "outputs/epistasis/gate_summary.json",
    min_covered: int = 10,
) -> dict:
    """Decompose Megascale multi-mutants into additive + specific epistasis and
    write the per-protein atlas + go/no-go gate summary.

    Foundation of the cross-protein epistasis program: measures, on real data,
    how much specific epistasis (``eps = ddG_obs - sum singles``) exists and
    where -- the signal the operator (Model 1) must learn to beat additive.
    """
    import json

    import numpy as np
    import pandas as pd

    from phaseagent.epistasis_decomposition import (
        decompose_multimutants,
        epistasis_gate_summary,
        protein_epistasis_atlas,
    )
    from phaseagent.megascale import MegascaleConfig, load_megascale

    df = load_megascale(
        MegascaleConfig(config_name=config_name, split=split, cache_dir=cache_dir)
    )
    dist = pd.to_numeric(df["mutation_distance"], errors="coerce")
    n_singles = int((dist == 1).sum())
    n_multi = int((dist >= 2).sum())

    decomposed = decompose_multimutants(df)
    atlas = protein_epistasis_atlas(df, decomposed, min_covered=int(min_covered))
    summary = epistasis_gate_summary(atlas)
    summary["loaded_rows"] = int(len(df))
    summary["loaded_singles"] = n_singles
    summary["loaded_multimutants"] = n_multi
    summary["covered_multimutants"] = (
        int(np.isfinite(decomposed["epsilon"].to_numpy(dtype=float)).sum())
        if len(decomposed)
        else 0
    )

    out_a = Path(VOLUME_PATH) / out_atlas
    out_d = Path(VOLUME_PATH) / out_decomposed
    out_s = Path(VOLUME_PATH) / out_summary
    out_a.parent.mkdir(parents=True, exist_ok=True)
    atlas.to_parquet(out_a, index=False)
    keep = [
        c
        for c in (
            "dataset_id",
            "mutation_notation",
            "mutation_distance",
            "ddG_ML",
            "ddG_additive",
            "epsilon",
            "n_edits",
        )
        if c in decomposed.columns
    ]
    if len(decomposed):
        decomposed[keep].to_parquet(out_d, index=False)
    out_s.write_text(json.dumps(summary, indent=2))
    volume.commit()

    print("[epistasis-atlas] gate summary:")
    print(json.dumps(summary, indent=2))
    return summary


@app.function(image=gpu_image, volumes={VOLUME_PATH: volume}, gpu="A10G", timeout=3600 * 4)
def run_headroom_gate(
    cache_dir: str = "data/megascale",
    max_proteins: int = 30,
    max_doubles_per_protein: int = 500,
    batch_size: int = 16,
    out_results: str = "outputs/epistasis/headroom_dplm.parquet",
    out_summary: str = "outputs/epistasis/headroom_summary.json",
    seed: int = 0,
) -> dict:
    """Headroom gate: does DPLM's own zero-shot predicted epistasis correlate
    with the MEASURED specific-epistasis residual?

    DPLM epistasis = LL(double) - LL(single_i) - LL(single_j) + LL(wt), a
    sequence-level pseudo-log-likelihood non-additivity (summed over residues).
    If this label-free signal tracks ``epsilon_specific``, a learned operator on
    DPLM features can beat additive+global -- the precondition for Model 1.
    Builds on the Sept-2025 "PLMs capture epistasis zero-shot" result.
    """
    import json

    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr, spearmanr

    from phaseagent.dplm_backbone import _load_dplm, score_sequence_logprob
    from phaseagent.epistasis_decomposition import (
        add_global_specific_layers,
        decompose_multimutants,
    )
    from phaseagent.megascale import MegascaleConfig, load_megascale
    from phaseagent.mutations import parse_mutation_notation

    df = load_megascale(
        MegascaleConfig(config_name="dataset2", split="train", cache_dir=cache_dir)
    )
    dec = add_global_specific_layers(decompose_multimutants(df))
    dbl = dec[
        (pd.to_numeric(dec["mutation_distance"], errors="coerce") == 2)
        & np.isfinite(dec["epsilon_specific"])
    ].copy()
    # In Megascale dataset2 each row's ``aa_seq`` IS that variant's own sequence
    # (for a distance-2 row, the double-mutant sequence). Reconstruct WT and the
    # two single-mutant sequences by reverting / re-applying the known mutations,
    # verifying the double carries the mutant residues at the notation positions
    # (a built-in numbering check).
    seq_col = next((c for c in ("aa_seq", "mutated_sequence") if c in dbl.columns), None)
    print(f"[headroom-gate] variant-sequence column: {seq_col}")
    if seq_col is None:
        return {"error": "no per-variant sequence column", "columns": list(dbl.columns)}
    dbl = dbl[dbl[seq_col].notna()]

    def _build(dd_seq: str, toks: list[str]):
        chars = list(dd_seq)
        parsed = []
        for t in toks:
            try:
                wa, pos, ma = t[0], int(t[1:-1]), t[-1]
            except (ValueError, IndexError):
                return None
            if not (1 <= pos <= len(chars)) or chars[pos - 1] != ma:
                return None  # double should carry the mutant residue here
            parsed.append((wa, pos, ma))
        wt = list(dd_seq)
        for wa, pos, ma in parsed:
            wt[pos - 1] = wa
        wt = "".join(wt)

        def mk(muts):
            c = list(wt)
            for wa, pos, ma in muts:
                c[pos - 1] = ma
            return "".join(c)

        return wt, mk([parsed[0]]), mk([parsed[1]]), dd_seq

    counts = dbl.groupby("dataset_id").size().sort_values(ascending=False)
    keep_ds = list(counts.index[: int(max_proteins)])
    parts = []
    for ds in keep_ds:
        g = dbl[dbl["dataset_id"] == ds]
        if len(g) > int(max_doubles_per_protein):
            g = g.sample(int(max_doubles_per_protein), random_state=int(seed))
        parts.append(g)
    sub = pd.concat(parts).reset_index(drop=True)

    recs, seqset, n_bad = [], set(), 0
    for _, r in sub.iterrows():
        toks = parse_mutation_notation(str(r["mutation_notation"]))
        if len(toks) != 2:
            continue
        built = _build(str(r[seq_col]), toks)
        if built is None:
            n_bad += 1
            continue
        wt, s1, s2, dd = built
        recs.append(
            {
                "dataset_id": str(r["dataset_id"]),
                "notation": str(r["mutation_notation"]),
                "eps_specific": float(r["epsilon_specific"]),
                "wt": wt,
                "s1": s1,
                "s2": s2,
                "dd": dd,
            }
        )
        seqset.update([wt, s1, s2, dd])
    print(f"[headroom-gate] built {len(recs)} doubles; skipped {n_bad} (numbering mismatch)")
    if not recs:
        return {"error": "no usable doubles after reconstruction", "n_bad": int(n_bad)}

    seqs = sorted(seqset)
    print(f"[headroom-gate] scoring {len(seqs)} unique sequences for {len(recs)} doubles ...")
    model, tok = _load_dplm()
    mean_ll = score_sequence_logprob(model, tok, seqs, batch_size=int(batch_size))
    ll = {s: float(mean_ll[i]) * len(s) for i, s in enumerate(seqs)}  # mean -> summed LL

    out = []
    for rec in recs:
        me = ll[rec["dd"]] - ll[rec["s1"]] - ll[rec["s2"]] + ll[rec["wt"]]
        out.append(
            {
                "dataset_id": rec["dataset_id"],
                "notation": rec["notation"],
                "eps_specific": rec["eps_specific"],
                "dplm_epistasis": me,
            }
        )
    res = pd.DataFrame(out)

    m = np.isfinite(res["eps_specific"]) & np.isfinite(res["dplm_epistasis"])
    per_prot = []
    for ds, g in res.groupby("dataset_id"):
        mm = np.isfinite(g["eps_specific"]) & np.isfinite(g["dplm_epistasis"])
        if int(mm.sum()) >= 20:
            per_prot.append(float(spearmanr(g["eps_specific"][mm], g["dplm_epistasis"][mm]).statistic))
    summary = {
        "n_doubles_scored": int(len(res)),
        "n_proteins": int(res["dataset_id"].nunique()),
        "n_unique_sequences": int(len(seqs)),
        "overall_spearman": float(spearmanr(res["eps_specific"][m], res["dplm_epistasis"][m]).statistic)
        if int(m.sum()) > 10
        else float("nan"),
        "overall_pearson": float(pearsonr(res["eps_specific"][m], res["dplm_epistasis"][m])[0])
        if int(m.sum()) > 10
        else float("nan"),
        "median_per_protein_spearman": float(np.median(per_prot)) if per_prot else float("nan"),
        "frac_proteins_pp_spearman_gt_0p2": float(np.mean(np.array(per_prot) > 0.2)) if per_prot else float("nan"),
        "n_proteins_with_pp_corr": int(len(per_prot)),
    }
    out_r = Path(VOLUME_PATH) / out_results
    out_r.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(out_r, index=False)
    (Path(VOLUME_PATH) / out_summary).write_text(json.dumps(summary, indent=2))
    volume.commit()
    print("[headroom-gate] DPLM zero-shot epistasis vs measured specific residual:")
    print(json.dumps(summary, indent=2))
    return summary


@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600 * 2)
def enrich_pair_features(
    cache_dir: str = "data/megascale",
    headroom: str = "outputs/epistasis/headroom_dplm.parquet",
    out: str = "outputs/epistasis/pair_features.parquet",
) -> dict:
    """Add structural + single-effect features to the scored double-mutant table.

    Reuses the DPLM-scored doubles (``run_headroom_gate`` output) and attaches,
    per double: Calpha-Calpha contact distance between the two mutated positions
    (from the bundled AlphaFold PDB), sequence separation, and the two
    single-mutant ddGs. This is the feature table for the contact-distance
    stratification and for Model 1.
    """
    import json

    import numpy as np
    import pandas as pd

    from phaseagent.megascale import MegascaleConfig, load_megascale
    from phaseagent.mutations import parse_mutation_notation

    def _ca_coords(pdb_path):
        from Bio.PDB import PDBParser

        s = PDBParser(QUIET=True).get_structure("p", str(pdb_path))
        xyz = []
        for model in s:
            for chain in model:
                for res in chain:
                    if "CA" in res:
                        a = res["CA"].coord
                        xyz.append((float(a[0]), float(a[1]), float(a[2])))
                break
            break
        return np.asarray(xyz, dtype=float) if xyz else None

    hr = pd.read_parquet(Path(VOLUME_PATH) / headroom)
    df = load_megascale(MegascaleConfig(config_name="dataset2", split="train", cache_dir=cache_dir))
    singles = df[pd.to_numeric(df["mutation_distance"], errors="coerce") == 1]
    sg = singles[["dataset_id", "mutation_notation", "ddG_ML"]].copy()
    sg["ddG_ML"] = pd.to_numeric(sg["ddG_ML"], errors="coerce")
    sg = sg.dropna()
    s_ddg = {(str(a), str(b)): float(c) for a, b, c in zip(sg["dataset_id"], sg["mutation_notation"], sg["ddG_ML"])}

    pdb_dir = Path(VOLUME_PATH) / cache_dir / "pdbs"
    def _norm(s):
        return str(s).replace("v2_", "").replace(".pdb", "")
    pdb_map = {}
    for f in pdb_dir.iterdir():
        if f.suffix == ".pdb":
            pdb_map.setdefault(_norm(f.name), f)

    coord_cache: dict = {}
    def get_coords(ds_id):
        key = _norm(ds_id)
        if key in coord_cache:
            return coord_cache[key]
        fp = pdb_map.get(key)
        coords = None
        if fp is not None:
            try:
                coords = _ca_coords(fp)
            except Exception:
                coords = None
        coord_cache[key] = coords
        return coords

    rows = []
    for _, r in hr.iterrows():
        ds = str(r["dataset_id"])
        toks = parse_mutation_notation(str(r["notation"]))
        if len(toks) != 2:
            continue
        try:
            p1, p2 = int(toks[0][1:-1]), int(toks[1][1:-1])
        except (ValueError, IndexError):
            continue
        coords = get_coords(ds)
        dist = np.nan
        if coords is not None and 1 <= p1 <= len(coords) and 1 <= p2 <= len(coords):
            dist = float(np.sqrt(((coords[p1 - 1] - coords[p2 - 1]) ** 2).sum()))
        s1, s2 = s_ddg.get((ds, toks[0]), np.nan), s_ddg.get((ds, toks[1]), np.nan)
        rows.append(
            {
                "dataset_id": ds,
                "notation": str(r["notation"]),
                "eps_specific": float(r["eps_specific"]),
                "dplm_epistasis": float(r["dplm_epistasis"]),
                "contact_distance": dist,
                "seq_sep": int(abs(p1 - p2)),
                "s1_ddg": s1,
                "s2_ddg": s2,
                "abs_s1": abs(s1) if np.isfinite(s1) else np.nan,
                "abs_s2": abs(s2) if np.isfinite(s2) else np.nan,
            }
        )
    res = pd.DataFrame(rows)
    summary = {
        "n": int(len(res)),
        "n_with_structure": int(np.isfinite(res["contact_distance"]).sum()) if len(res) else 0,
        "n_with_singles": int(np.isfinite(res["s1_ddg"]).sum()) if len(res) else 0,
        "n_proteins": int(res["dataset_id"].nunique()) if len(res) else 0,
    }
    out_p = Path(VOLUME_PATH) / out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(out_p, index=False)
    volume.commit()
    print("[enrich-pair-features]", json.dumps(summary))
    return summary


@app.function(image=gpu_image, volumes={VOLUME_PATH: volume}, gpu="A10G", timeout=3600 * 4)
def extract_function_features(
    assay_csv: str,
    assay_name: str,
    max_doubles: int = 12000,
    batch_size: int = 16,
    out: str = "",
    seed: int = 0,
) -> dict:
    """Extract rich DPLM features per double-mutant for a FUNCTION assay, so a
    learned operator (Model 1) can be trained to beat the zero-shot baseline.

    Per double we save: DPLM last-layer hidden states at the two mutated
    positions (h_i, h_j), the zero-shot pseudo-LL epistasis scalar, mutation +
    wildtype identities, positions, and the measured specific-epistasis target.
    """
    import json

    import numpy as np
    import pandas as pd
    import torch

    from phaseagent.dplm_backbone import _load_dplm, score_sequence_logprob
    from phaseagent.epistasis_decomposition import add_global_specific_layers, decompose_multimutants
    from phaseagent.mutations import parse_mutation_notation
    from phaseagent.spectrum import AA_TO_ID

    d = pd.read_csv(Path(VOLUME_PATH) / assay_csv).rename(columns={"mutant": "mutation_notation"})
    d["dataset_id"] = assay_name
    d["mutation_distance"] = d["mutation_notation"].astype(str).apply(lambda s: len(parse_mutation_notation(s)))
    d["DMS_score"] = pd.to_numeric(d["DMS_score"], errors="coerce")
    dec = add_global_specific_layers(
        decompose_multimutants(d, label_col="DMS_score", max_distance=2), label_col="DMS_score"
    )
    dbl = dec[np.isfinite(dec["epsilon_specific"])].copy()
    if len(dbl) > int(max_doubles):
        dbl = dbl.sample(int(max_doubles), random_state=int(seed))

    sg = d[d["mutation_distance"] == 1]
    slk = {str(a): str(b) for a, b in zip(sg["mutation_notation"], sg["mutated_sequence"])}
    one = sg.iloc[0]
    t0 = parse_mutation_notation(str(one["mutation_notation"]))[0]
    seq0 = str(one["mutated_sequence"])
    wt = list(seq0)
    wt[int(t0[1:-1]) - 1] = t0[0]
    wt = "".join(wt)

    recs = []
    for _, r in dbl.iterrows():
        toks = parse_mutation_notation(str(r["mutation_notation"]))
        if len(toks) != 2:
            continue
        s1, s2 = slk.get(toks[0]), slk.get(toks[1])
        if s1 is None or s2 is None:
            continue
        try:
            p1, p2 = int(toks[0][1:-1]), int(toks[1][1:-1])
        except (ValueError, IndexError):
            continue
        if not (1 <= p1 < len(wt) and 1 <= p2 < len(wt)):
            continue
        recs.append({"eps": float(r["epsilon_specific"]), "p1": p1, "p2": p2,
                     "ma1": toks[0][-1], "ma2": toks[1][-1], "wa1": toks[0][0], "wa2": toks[1][0],
                     "dd": str(r["mutated_sequence"]), "s1": s1, "s2": s2})

    seqs = {wt}
    for rec in recs:
        seqs.update([rec["dd"], rec["s1"], rec["s2"]])
    seqs = sorted(seqs)
    model, tok = _load_dplm()
    mean_ll = score_sequence_logprob(model, tok, seqs, batch_size=int(batch_size))
    ll = {s: float(mean_ll[i]) * len(s) for i, s in enumerate(seqs)}

    with torch.inference_mode():
        inp = tok([wt], return_tensors="pt")
        inp = {k: v.to(next(model.parameters()).device) for k, v in inp.items()}
        H = model(**inp, output_hidden_states=True).hidden_states[-1][0].float().cpu().numpy()

    N, Dh = len(recs), H.shape[1]
    Hi = np.zeros((N, Dh), np.float16)
    Hj = np.zeros((N, Dh), np.float16)
    meta = np.zeros((N, 8), np.float32)
    for k, rec in enumerate(recs):
        Hi[k] = H[rec["p1"]]
        Hj[k] = H[rec["p2"]]
        pll = ll[rec["dd"]] - ll[rec["s1"]] - ll[rec["s2"]] + ll[wt]
        meta[k] = [rec["eps"], pll,
                   AA_TO_ID.get(rec["ma1"], 20), AA_TO_ID.get(rec["ma2"], 20),
                   AA_TO_ID.get(rec["wa1"], 20), AA_TO_ID.get(rec["wa2"], 20),
                   rec["p1"], rec["p2"]]
    outp = out or f"outputs/epistasis/feat_{assay_name}.npz"
    op = Path(VOLUME_PATH) / outp
    op.parent.mkdir(parents=True, exist_ok=True)
    np.savez(op, Hi=Hi, Hj=Hj, meta=meta)
    volume.commit()
    from scipy.stats import spearmanr
    zs = float(spearmanr(meta[:, 0], meta[:, 1]).statistic) if N > 10 else float("nan")
    summary = {"assay": assay_name, "n_doubles": int(N), "hidden_dim": int(Dh),
               "zero_shot_pll_spearman": zs, "out": outp}
    print("[extract-function-features]", json.dumps(summary))
    return summary


@app.function(image=gpu_image, volumes={VOLUME_PATH: volume}, gpu="A10G", timeout=3600 * 5)
def extract_function_shifts(
    assay_csv: str,
    assay_name: str,
    max_doubles: int = 12000,
    batch_size: int = 16,
    out: str = "",
    seed: int = 0,
) -> dict:
    """Representation-shift features for cross-position epistasis transfer.

    For each double (i->a, j->b) we add the DPLM representation shifts:
       shift_j = H^(i->a)[j] - H^wt[j]   (how mutating i moves j's embedding)
       shift_i = H^(j->b)[i] - H^wt[i]
    These measure coupling directly and should transfer to unseen positions
    better than position-specific identity. Singles are streamed (forward once
    each, extract only the needed partner positions) to bound memory.
    """
    import json

    import numpy as np
    import pandas as pd
    import torch

    from phaseagent.dplm_backbone import _load_dplm
    from phaseagent.epistasis_decomposition import add_global_specific_layers, decompose_multimutants
    from phaseagent.mutations import parse_mutation_notation

    d = pd.read_csv(Path(VOLUME_PATH) / assay_csv).rename(columns={"mutant": "mutation_notation"})
    d["dataset_id"] = assay_name
    d["mutation_distance"] = d["mutation_notation"].astype(str).apply(lambda s: len(parse_mutation_notation(s)))
    d["DMS_score"] = pd.to_numeric(d["DMS_score"], errors="coerce")
    dec = add_global_specific_layers(
        decompose_multimutants(d, label_col="DMS_score", max_distance=2), label_col="DMS_score"
    )
    dbl = dec[np.isfinite(dec["epsilon_specific"])].copy()
    if len(dbl) > int(max_doubles):
        dbl = dbl.sample(int(max_doubles), random_state=int(seed))

    sg = d[d["mutation_distance"] == 1]
    slk = {str(a): str(b) for a, b in zip(sg["mutation_notation"], sg["mutated_sequence"])}
    one = sg.iloc[0]
    t0 = parse_mutation_notation(str(one["mutation_notation"]))[0]
    seq0 = str(one["mutated_sequence"])
    wt = list(seq0)
    wt[int(t0[1:-1]) - 1] = t0[0]
    wt = "".join(wt)

    recs = []
    for _, r in dbl.iterrows():
        toks = parse_mutation_notation(str(r["mutation_notation"]))
        if len(toks) != 2:
            continue
        s1, s2 = slk.get(toks[0]), slk.get(toks[1])
        if s1 is None or s2 is None:
            continue
        try:
            p1, p2 = int(toks[0][1:-1]), int(toks[1][1:-1])
        except (ValueError, IndexError):
            continue
        if not (1 <= p1 < len(wt) and 1 <= p2 < len(wt)):
            continue
        recs.append({"eps": float(r["epsilon_specific"]), "p1": p1, "p2": p2, "s1": s1, "s2": s2})

    model, tok = _load_dplm()
    dev = next(model.parameters()).device

    def hidden(seqs):
        inp = tok(list(seqs), return_tensors="pt", padding=True)
        inp = {k: v.to(dev) for k, v in inp.items()}
        with torch.inference_mode():
            return model(**inp, output_hidden_states=True).hidden_states[-1].float().cpu().numpy()

    H_wt = hidden([wt])[0]  # (Ltok, D); residue p at token index p
    N, Dh = len(recs), H_wt.shape[1]
    shift_i = np.zeros((N, Dh), np.float16)
    shift_j = np.zeros((N, Dh), np.float16)

    # map each unique single seq -> list of (rec_idx, partner_pos, slot)
    needs: dict = {}
    for k, rec in enumerate(recs):
        needs.setdefault(rec["s1"], []).append((k, rec["p2"], "j"))  # mutate i (s1) -> read j
        needs.setdefault(rec["s2"], []).append((k, rec["p1"], "i"))  # mutate j (s2) -> read i
    singles = list(needs.keys())
    for s in range(0, len(singles), int(batch_size)):
        chunk = singles[s : s + int(batch_size)]
        Hs = hidden(chunk)
        for bi, seq in enumerate(chunk):
            for (k, partner, slot) in needs[seq]:
                sh = Hs[bi, partner] - H_wt[partner]
                if slot == "j":
                    shift_j[k] = sh
                else:
                    shift_i[k] = sh

    meta = np.zeros((N, 3), np.float32)
    for k, rec in enumerate(recs):
        meta[k] = [rec["eps"], rec["p1"], rec["p2"]]
    outp = out or f"outputs/epistasis/shift_{assay_name}.npz"
    op = Path(VOLUME_PATH) / outp
    op.parent.mkdir(parents=True, exist_ok=True)
    np.savez(op, shift_i=shift_i, shift_j=shift_j, meta=meta)
    volume.commit()
    print("[extract-function-shifts]", json.dumps({"assay": assay_name, "n": int(N), "n_singles": len(singles), "out": outp}))
    return {"assay": assay_name, "n": int(N), "out": outp}


@app.function(image=gpu_lora_image, volumes={VOLUME_PATH: volume}, gpu="H100", timeout=3600 * 12)
def train_epistasis_e2e(
    n_holdout_proteins: int = 25,
    epochs: int = 4,
    lr: float = 1e-4,
    lora_r: int = 16,
    batch_size: int = 48,
    max_doubles: int = 120000,
    min_doubles_per_protein: int = 30,
    out: str = "outputs/epistasis/e2e_results.json",
    seed: int = 0,
) -> dict:
    """End-to-end LoRA fine-tune of DPLM-650M + a pairwise coupling head to
    predict specific-epistasis, trained across many proteins and evaluated on
    HELD-OUT proteins (the cross-protein generalization test). The whole
    backbone adapts via LoRA -- not a frozen feature extractor.
    """
    import json

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from scipy.stats import spearmanr
    from transformers import AutoTokenizer, EsmForMaskedLM

    from phaseagent.epistasis_decomposition import add_global_specific_layers, decompose_multimutants
    from phaseagent.megascale import MegascaleConfig, load_megascale
    from phaseagent.mutations import parse_mutation_notation

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    df = load_megascale(MegascaleConfig(config_name="dataset2", split="train", cache_dir="data/megascale"))
    dec = add_global_specific_layers(decompose_multimutants(df))
    dbl = dec[(pd.to_numeric(dec["mutation_distance"], errors="coerce") == 2) & np.isfinite(dec["epsilon_specific"])].copy()
    dbl = dbl[dbl["aa_seq"].notna()]

    rows = []
    for _, r in dbl.iterrows():
        toks = parse_mutation_notation(str(r["mutation_notation"]))
        if len(toks) != 2:
            continue
        try:
            p1, p2 = int(toks[0][1:-1]), int(toks[1][1:-1])
        except (ValueError, IndexError):
            continue
        seq = str(r["aa_seq"])
        if not (1 <= p1 <= len(seq) and 1 <= p2 <= len(seq) and seq[p1 - 1] == toks[0][-1] and seq[p2 - 1] == toks[1][-1]):
            continue
        rows.append((str(r["dataset_id"]), seq, p1, p2, float(r["epsilon_specific"])))
    feat = pd.DataFrame(rows, columns=["prot", "seq", "p1", "p2", "eps"])
    counts = feat.groupby("prot").size()
    keep = counts[counts >= min_doubles_per_protein].index
    feat = feat[feat["prot"].isin(keep)].reset_index(drop=True)
    # per-protein standardize target on full per-protein data (ddof=0 avoids NaN)
    feat["eps_z"] = feat.groupby("prot")["eps"].transform(lambda v: (v - v.mean()) / (v.std(ddof=0) + 1e-6))
    if len(feat) > max_doubles:
        feat = feat.sample(max_doubles, random_state=seed).reset_index(drop=True)

    prots = sorted(feat["prot"].unique())
    rng.shuffle(prots)
    holdout = set(prots[: int(n_holdout_proteins)])
    tr = feat[~feat["prot"].isin(holdout)].reset_index(drop=True)
    te = feat[feat["prot"].isin(holdout)].reset_index(drop=True)
    print(f"[e2e] {len(feat)} doubles, {len(prots)} proteins; train {len(tr)} / {len(prots)-len(holdout)} prot, "
          f"holdout {len(te)} / {len(holdout)} prot")

    tok = AutoTokenizer.from_pretrained("airkingbd/dplm_650m")
    base = EsmForMaskedLM.from_pretrained("airkingbd/dplm_650m")
    lconf = LoraConfig(r=int(lora_r), lora_alpha=int(2 * lora_r), lora_dropout=0.05,
                       target_modules=["query", "value"], bias="none")
    model = get_peft_model(base, lconf).cuda()
    Dh = base.config.hidden_size
    head = nn.Sequential(nn.LayerNorm(3 * Dh), nn.Linear(3 * Dh, 512), nn.GELU(), nn.Dropout(0.2),
                         nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1)).cuda()
    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    n_train_params = sum(p.numel() for p in params)
    print(f"[e2e] trainable params (LoRA + head): {n_train_params/1e6:.1f}M")
    opt = torch.optim.AdamW(params, lr=float(lr), weight_decay=1e-4)

    def forward_batch(sub):
        enc = tok(list(sub["seq"]), return_tensors="pt", padding=True)
        enc = {k: v.cuda() for k, v in enc.items()}
        H = model(**enc, output_hidden_states=True).hidden_states[-1]
        idx = torch.arange(len(sub), device="cuda")
        p1 = torch.tensor(sub["p1"].to_numpy(), device="cuda")
        p2 = torch.tensor(sub["p2"].to_numpy(), device="cuda")
        hi, hj = H[idx, p1], H[idx, p2]
        return head(torch.cat([hi, hj, hi * hj], -1)).squeeze(-1)

    for ep in range(int(epochs)):
        model.train(); head.train()
        order = rng.permutation(len(tr))
        tot, nb = 0.0, 0
        for s in range(0, len(tr), int(batch_size)):
            sub = tr.iloc[order[s : s + int(batch_size)]]
            y = torch.tensor(sub["eps_z"].to_numpy(), dtype=torch.float32, device="cuda")
            opt.zero_grad()
            loss = F.smooth_l1_loss(forward_batch(sub), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += float(loss); nb += 1
        print(f"[e2e] epoch {ep+1}/{epochs} train Huber {tot/max(nb,1):.4f}")

    # eval held-out proteins
    model.eval(); head.eval()
    preds = np.zeros(len(te))
    with torch.no_grad():
        for s in range(0, len(te), int(batch_size)):
            sub = te.iloc[s : s + int(batch_size)]
            preds[s : s + len(sub)] = forward_batch(sub).cpu().numpy()
    te = te.assign(pred=preds)
    per_prot = []
    for p, g in te.groupby("prot"):
        if len(g) >= 20 and g["pred"].std() > 0:
            per_prot.append(float(spearmanr(g["eps"], g["pred"]).statistic))
    summary = {
        "n_doubles": int(len(feat)), "n_proteins": int(len(prots)),
        "n_holdout_proteins": int(len(holdout)), "epochs": int(epochs), "lora_r": int(lora_r),
        "trainable_params_M": round(n_train_params / 1e6, 2),
        "heldout_protein_median_spearman": float(np.median(per_prot)) if per_prot else float("nan"),
        "heldout_protein_mean_spearman": float(np.mean(per_prot)) if per_prot else float("nan"),
        "frac_heldout_prot_gt_0p2": float(np.mean(np.array(per_prot) > 0.2)) if per_prot else float("nan"),
        "n_heldout_prot_scored": len(per_prot),
        "zero_shot_reference": 0.25,
    }
    (Path(VOLUME_PATH) / out).parent.mkdir(parents=True, exist_ok=True)
    (Path(VOLUME_PATH) / out).write_text(json.dumps(summary, indent=2))
    volume.commit()
    print("[e2e]", json.dumps(summary, indent=2))
    return summary


@app.function(image=gpu_lora_image, volumes={VOLUME_PATH: volume}, gpu="A10G", timeout=3600 * 8)
def train_generative_design(
    assay_csv: str = "raw/proteingym_v1_3/DMS_ProteinGym_substitutions/SPG1_STRSG_Wu_2016.csv",
    assay_name: str = "GB1_Wu",
    epochs: int = 5,
    lr: float = 1e-4,
    lora_r: int = 16,
    batch_size: int = 32,
    n_samples: int = 4000,
    out: str = "outputs/epistasis/gen_design.json",
    seed: int = 0,
) -> dict:
    """Generative Model 2: function-weighted LoRA fine-tune of DPLM that GENERATES
    high-function multi-mutants. Trained on measured combos with loss weighted by
    measured function (so high-function combos dominate); generation is iterative
    masked sampling over the editable positions (captures coupling). Evaluated by
    the measured function of the GENERATED library vs unconditioned DPLM, random,
    and the additive-top library, on a complete combinatorial assay so every
    sampled combo is measurable.
    """
    import json

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, EsmForMaskedLM

    from phaseagent.mutations import parse_mutation_notation

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d = pd.read_csv(Path(VOLUME_PATH) / assay_csv).rename(columns={"mutant": "mutation_notation"})
    d["DMS_score"] = pd.to_numeric(d["DMS_score"], errors="coerce")
    d = d[d["DMS_score"].notna() & d["mutated_sequence"].notna()].copy()
    posset = set()
    for s in d["mutation_notation"].astype(str):
        for t in parse_mutation_notation(s):
            try:
                posset.add(int(t[1:-1]))
            except (ValueError, IndexError):
                pass
    positions = sorted(posset)
    one = d[d["mutation_notation"].astype(str).apply(lambda s: len(parse_mutation_notation(s)) == 1)].iloc[0]
    t0 = parse_mutation_notation(str(one["mutation_notation"]))[0]
    seq0 = str(one["mutated_sequence"])
    wt = list(seq0)
    wt[int(t0[1:-1]) - 1] = t0[0]
    wt = "".join(wt)

    def combo_of(seq):
        return tuple(seq[p - 1] for p in positions)
    d["combo"] = d["mutated_sequence"].astype(str).apply(combo_of)
    combo_score = dict(zip(d["combo"], d["DMS_score"]))
    print(f"[gen] {assay_name}: {len(positions)} editable positions {positions}, {len(combo_score)} measured combos")

    idx = rng.permutation(len(d))
    cut = int(0.85 * len(d))
    tr = d.iloc[idx[:cut]].reset_index(drop=True)
    train_combos = set(tr["combo"])
    tr_seqs = tr["mutated_sequence"].astype(str).tolist()
    tr_w = tr["DMS_score"].rank(pct=True).to_numpy()  # function weight in [0,1]

    tok = AutoTokenizer.from_pretrained("airkingbd/dplm_650m")
    base = EsmForMaskedLM.from_pretrained("airkingbd/dplm_650m")
    lconf = LoraConfig(r=int(lora_r), lora_alpha=int(2 * lora_r), lora_dropout=0.05,
                       target_modules=["query", "value"], bias="none")
    model = get_peft_model(base, lconf).cuda()
    AAs = "ACDEFGHIKLMNPQRSTVWY"
    aa_ids = torch.tensor([tok.convert_tokens_to_ids(a) for a in AAs], device="cuda")
    mask_tok = tok.mask_token_id
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(lr))

    def masked_ids(seqs, mask_positions_list):
        enc = tok(list(seqs), return_tensors="pt", padding=True)
        ids = enc["input_ids"]
        for b, mps in enumerate(mask_positions_list):
            for p in mps:
                ids[b, p] = mask_tok  # token index = residue position (CLS at 0)
        return {"input_ids": ids.cuda(), "attention_mask": enc["attention_mask"].cuda()}

    # ---- train: random-subset masking of editable positions, function-weighted ----
    for ep in range(int(epochs)):
        model.train()
        order = rng.permutation(len(tr_seqs))
        tot, nb = 0.0, 0
        for s in range(0, len(order), int(batch_size)):
            bidx = order[s : s + int(batch_size)]
            seqs = [tr_seqs[i] for i in bidx]
            mps = [list(rng.choice(positions, size=rng.integers(1, len(positions) + 1), replace=False)) for _ in bidx]
            enc = masked_ids(seqs, mps)
            tgt_ids = tok(seqs, return_tensors="pt", padding=True)["input_ids"].cuda()
            logits = model(**enc).logits
            loss = 0.0
            for b, mp in enumerate(mps):
                for p in mp:
                    lp = F.log_softmax(logits[b, p], -1)
                    loss = loss - tr_w[bidx[b]] * lp[tgt_ids[b, p]]
            loss = loss / max(1, sum(len(mp) for mp in mps))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            tot += float(loss); nb += 1
        print(f"[gen] epoch {ep+1}/{epochs} weighted-NLL {tot/max(nb,1):.4f}")

    # ---- iterative masked generation over editable positions ----
    def generate(net, n):
        net.eval()
        combos = []
        with torch.no_grad():
            for s in range(0, n, 64):
                bs = min(64, n - s)
                seqs = [wt] * bs
                cur = [list(wt) for _ in range(bs)]
                fill_order = [list(rng.permutation(positions)) for _ in range(bs)]
                for step in range(len(positions)):
                    mps = [[fo[step]] for fo in fill_order]
                    enc = masked_ids(["".join(c) for c in cur], mps)
                    logits = net(**enc).logits
                    for b in range(bs):
                        p = fill_order[b][step]
                        probs = F.softmax(logits[b, p, aa_ids], -1)
                        a = AAs[int(torch.multinomial(probs, 1))]
                        cur[b][p - 1] = a
                combos += [tuple(c[p - 1] for p in positions) for c in cur]
        return combos

    def lib_quality(combos):
        sc = [combo_score[c] for c in combos if c in combo_score]
        novel = [combo_score[c] for c in combos if c in combo_score and c not in train_combos]
        cov = len(sc) / max(1, len(combos))
        return (float(np.mean(sc)) if sc else float("nan"),
                float(np.mean(novel)) if novel else float("nan"),
                float(cov), len(novel))

    gen_mean, gen_novel, gen_cov, n_novel = lib_quality(generate(model, int(n_samples)))
    with model.disable_adapter():  # unconditioned DPLM = adapters off
        base_combos = generate(model, int(n_samples // 2))
    base_mean, _, _, _ = lib_quality(base_combos)
    rand_combos = [tuple(AAs[int(i)] for i in rng.integers(0, 20, len(positions))) for _ in range(int(n_samples))]
    rand_mean, _, _, _ = lib_quality(rand_combos)
    all_scores = np.array(list(combo_score.values()))
    add_top = float(np.mean(np.sort(all_scores)[-int(n_samples):]))  # additive/oracle-ish top library

    summary = {
        "assay": assay_name, "positions": positions, "epochs": int(epochs),
        "generated_lib_mean_function": gen_mean,
        "generated_NOVEL_lib_mean_function": gen_novel,
        "n_novel_generated": int(n_novel), "coverage": round(gen_cov, 3),
        "unconditioned_DPLM_mean": base_mean,
        "random_lib_mean": rand_mean,
        "top_library_ceiling": add_top,
        "library_mean_overall": float(np.mean(all_scores)),
    }
    (Path(VOLUME_PATH) / out).parent.mkdir(parents=True, exist_ok=True)
    (Path(VOLUME_PATH) / out).write_text(json.dumps(summary, indent=2))
    volume.commit()
    print("[gen]", json.dumps(summary, indent=2))
    return summary


@app.function(image=gpu_lora_image, gpu="H100", timeout=300)
def h100_probe() -> dict:
    import torch
    return {"cuda": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


# ---------------------------------------------------------------------------
# Convenience: full v2 pipeline driver
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def run_v2_pipeline():
    """Smoke-run the v2 pipeline. Each step writes to the shared volume."""
    print("[v2] download_megascale ...")
    print(download_megascale.remote())
    print("[v2] train_prior_on_megascale ...")
    print(train_prior_on_megascale.remote())
    print("[v2] run_megascale_t2_with_additive ...")
    print(run_megascale_t2_with_additive.remote())
    print("[v2] run_few_shot_sweep ...")
    print(run_few_shot_sweep.remote())
    print("[v2] run_conformal_abstention ...")
    print(run_conformal_abstention.remote())
    print("[v2] precompute_esm2_embeddings ...")
    print(precompute_esm2_embeddings.remote())
    print("[v2] run_esm2_rf_baseline ...")
    print(run_esm2_rf_baseline.remote())
    print("[v2] run_twisted_smc_dplm ...")
    print(run_twisted_smc_dplm.remote(max_tasks=4))
    print("[v2] make_nature_figures ...")
    print(make_nature_figures.remote())
