"""Modal app for PhaseAgent-lite.

Volume:    phaseagent-data (mounted at /data inside containers)
Layout on volume:
    /data/raw/proteingym_v1_3/   raw per-assay CSVs from the substitution archive
    /data/processed/all_dms.parquet
    /data/outputs/tables/*.csv
    /data/outputs/figures/*.png|.pdf

Quickstart:
    modal run modal_app.py::download_proteingym
    modal run modal_app.py::build_dataset --n-datasets 10
    modal run modal_app.py::run_phase_atlas
    modal run modal_app.py::run_phaseagent
    modal run modal_app.py::run_phase_aware_search
    modal run modal_app.py::run_plm_scoring  # GPU
    modal run modal_app.py::make_figures
    modal run modal_app.py::pull_outputs --local-dir outputs

Or run the whole pipeline:
    modal run modal_app.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import modal

APP_NAME = "phaseagent"
VOLUME_NAME = "phaseagent-data"
VOLUME_PATH = "/data"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

CPU_PIP = [
    "numpy>=1.24",
    "pandas>=2.0",
    "scipy>=1.10",
    "scikit-learn>=1.2",
    "matplotlib>=3.7",
    "pyyaml>=6.0",
    "tqdm>=4.65",
    "requests>=2.31",
    "biopython>=1.81",
    "pyarrow>=14.0",
]

GPU_PIP = [
    "torch==2.4.0",
    "fair-esm==2.0.0",
    # ESM-IF1 inverse folding deps. fair-esm doesn't pin these in install_requires.
    # biotite 0.40 is built against numpy 1.x — we must pin numpy<2 for ABI.
    "numpy<2",
    "biotite==0.40.0",
    "torch-geometric==2.4.0",
    # Phase 2.2 — DPLM 650M loads via standard transformers (it's ESM-2 arch).
    "transformers==4.39.2",
    "huggingface_hub>=0.20",
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
    .pip_install("torch-scatter", extra_options=f"-f {PYG_WHEEL_INDEX}")
    .add_local_dir("src/phaseagent", remote_path="/root/phaseagent", copy=True)
)


# ---------- 1. Download ----------

@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600 * 2)
def download_proteingym(
    url: str = "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip",
    out_dir: str = "raw/proteingym_v1_3",
    overwrite: bool = False,
) -> dict:
    import io
    import zipfile

    import requests
    from tqdm import tqdm

    target = Path(VOLUME_PATH) / out_dir
    if target.exists() and any(target.rglob("*.csv")) and not overwrite:
        n = sum(1 for _ in target.rglob("*.csv"))
        print(f"[download] {target} already populated with {n} CSVs; skip (overwrite=False)")
        return {"path": str(target), "n_csvs": n, "skipped": True}
    target.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url}")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        buf = io.BytesIO()
        chunk = 8 * 1024 * 1024
        with tqdm(total=total, unit="B", unit_scale=True) as pbar:
            for c in r.iter_content(chunk_size=chunk):
                if c:
                    buf.write(c)
                    pbar.update(len(c))
        buf.seek(0)
    print("[download] extracting…")
    with zipfile.ZipFile(buf) as z:
        z.extractall(target)
    volume.commit()
    csvs = list(target.rglob("*.csv"))
    print(f"[download] wrote {len(csvs)} CSVs to {target}")
    return {"path": str(target), "n_csvs": len(csvs), "skipped": False}


DATASET_TO_UNIPROT: dict[str, str] = {
    # Hand-curated for the 3 test-split datasets used in Phase 2.3 / 2.2.
    # Extend as more datasets enter the test split. The canonical ProteinGym
    # reference table also has these mappings; we keep a small inline copy
    # to avoid a third archive download for the few datasets we evaluate.
    "F7YBW8_MESOW_Aakre_2015": "F7YBW8",
    "GCN4_YEAST_Staller_2018": "P03069",
    "GFP_AEQVI_Sarkisyan_2016": "P42212",
    # Train / val datasets — populated for completeness even though the
    # structure baseline only runs on test.
    "CAPSD_AAV2S_Sinai_2021": "P03135",
    "D7PM05_CLYGR_Somermeyer_2022": "D7PM05",
    "F7YBW8_MESOW_Ding_2023": "F7YBW8",
    "HIS7_YEAST_Pokusaeva_2019": "P06633",
    "PHOT_CHLRE_Chen_2023": "P25168",
    "Q6WV12_9MAXI_Somermeyer_2022": "Q6WV12",
    "Q8WTC7_9CNID_Somermeyer_2022": "Q8WTC7",
    "SPG1_STRSG_Wu_2016": "P19909",
}


@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600)
def download_alphafold_structures(
    out_dir: str = "structures",
    dataset_ids: str = "",
    overwrite: bool = False,
) -> dict:
    """Fetch AlphaFold-2 predicted PDB structures for our DMS datasets.

    Resolves UniProt accession via ``DATASET_TO_UNIPROT``, queries the AFDB
    prediction API for the latest model version, then downloads the PDB.
    Caches results on the Modal volume under ``structures/{dataset_id}.pdb``.
    Datasets without a UniProt mapping (or without an AFDB structure) are
    reported and skipped — never silently imputed.
    """
    import json

    import requests

    target = Path(VOLUME_PATH) / out_dir
    target.mkdir(parents=True, exist_ok=True)
    candidates = [d.strip() for d in dataset_ids.split(",") if d.strip()]
    if not candidates:
        candidates = list(DATASET_TO_UNIPROT.keys())
    summary = {}
    for ds_id in candidates:
        out_path = target / f"{ds_id}.pdb"
        if out_path.exists() and not overwrite:
            summary[ds_id] = {"status": "cached", "path": str(out_path)}
            continue
        accession = DATASET_TO_UNIPROT.get(ds_id)
        if accession is None:
            summary[ds_id] = {"status": "no_uniprot_mapping"}
            continue
        api = f"https://alphafold.ebi.ac.uk/api/prediction/{accession}"
        try:
            r = requests.get(api, timeout=60)
            r.raise_for_status()
            data = r.json()
            if not data:
                summary[ds_id] = {"status": "no_afdb_entry", "uniprot": accession}
                continue
            entry = data[0]
            pdb_url = entry["pdbUrl"]
            seq = entry.get("uniprotSequence", "")
            pdb_resp = requests.get(pdb_url, timeout=120)
            pdb_resp.raise_for_status()
            out_path.write_bytes(pdb_resp.content)
            summary[ds_id] = {
                "status": "downloaded",
                "uniprot": accession,
                "length": len(seq),
                "path": str(out_path),
                "pdb_url": pdb_url,
            }
            print(f"[afdb] {ds_id}: {accession} ({len(seq)} aa) → {out_path.name}")
        except Exception as exc:
            summary[ds_id] = {"status": f"error:{type(exc).__name__}:{exc}"}
            print(f"[afdb] {ds_id}: ERROR {exc}")
    volume.commit()
    return summary


@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600 * 4)
def download_proteingym_zero_shot(
    url: str = "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/zero_shot_substitutions_scores.zip",
    out_dir: str = "raw/proteingym_v1_3_zero_shot",
    overwrite: bool = False,
) -> dict:
    """Pull the 1.9 GB v1.3 zero-shot model-prediction archive (Tranception, EVE, etc.).

    Layout after extraction is roughly
    ``zero_shot/substitutions/{MODEL_NAME}/{DATASET_ID}.csv``; each CSV has a
    per-variant predicted score column whose name varies by model.
    """
    import io
    import zipfile

    import requests
    from tqdm import tqdm

    target = Path(VOLUME_PATH) / out_dir
    if target.exists() and any(target.rglob("*.csv")) and not overwrite:
        n = sum(1 for _ in target.rglob("*.csv"))
        print(f"[zero-shot] {target} already populated with {n} CSVs; skip")
        return {"path": str(target), "n_csvs": n, "skipped": True}
    target.mkdir(parents=True, exist_ok=True)
    print(f"[zero-shot] {url}")
    with requests.get(url, stream=True, timeout=1800) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        buf = io.BytesIO()
        chunk = 8 * 1024 * 1024
        with tqdm(total=total, unit="B", unit_scale=True) as pbar:
            for c in r.iter_content(chunk_size=chunk):
                if c:
                    buf.write(c)
                    pbar.update(len(c))
        buf.seek(0)
    print("[zero-shot] extracting…")
    with zipfile.ZipFile(buf) as z:
        z.extractall(target)
    volume.commit()
    csvs = list(target.rglob("*.csv"))
    top_dirs = sorted({p.parts[len(target.parts):][0] for p in csvs})[:10]
    model_dirs: list[str] = []
    for top in top_dirs:
        sub = target / top
        if sub.is_dir():
            model_dirs.extend(sorted(d.name for d in sub.iterdir() if d.is_dir())[:30])
    print(f"[zero-shot] wrote {len(csvs)} CSVs across top-level dirs: {top_dirs}")
    print(f"[zero-shot] sample model dirs: {model_dirs[:10]}")
    return {
        "path": str(target),
        "n_csvs": len(csvs),
        "skipped": False,
        "top_dirs": top_dirs,
        "sample_models": model_dirs[:10],
    }


# ---------- 2. Build processed parquet ----------

@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600)
def build_dataset(
    raw_dir: str = "raw/proteingym_v1_3",
    out_path: str = "processed/all_dms.parquet",
    fitness_normalization: str = "rank",
    threshold_mode: str = "quantile",
    threshold_value: float = 0.75,
    n_datasets: Optional[int] = None,
    min_rows: int = 100,
    max_rows: int = 200_000,
) -> dict:
    from phaseagent.proteingym import build_proteingym_dataset

    raw = Path(VOLUME_PATH) / raw_dir
    out = Path(VOLUME_PATH) / out_path
    out.parent.mkdir(parents=True, exist_ok=True)

    if not raw.exists() or not any(raw.rglob("*.csv")):
        raise FileNotFoundError(
            f"No raw CSVs at {raw}. Run `modal run modal_app.py::download_proteingym` first."
        )

    full, summary = build_proteingym_dataset(
        raw_dir=raw,
        fitness_normalization=fitness_normalization,
        threshold_mode=threshold_mode,
        threshold_value=threshold_value,
        n_datasets=n_datasets,
        min_rows=min_rows,
        max_rows=max_rows,
    )
    full.to_parquet(out, index=False)
    summary_path = Path(VOLUME_PATH) / "outputs/tables/dataset_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    volume.commit()
    return {"rows": int(len(full)), "datasets": int(len(summary)), "out": str(out)}


# ---------- 3. Phase atlas ----------

@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600 * 2)
def run_phase_atlas(
    data: str = "processed/all_dms.parquet",
    out: str = "outputs/tables/phase_boundaries.csv",
    bootstrap: int = 200,
    min_count_per_distance: int = 10,
) -> dict:
    import numpy as np
    import pandas as pd

    from phaseagent.phase import (
        bootstrap_phase_boundary,
        classify_phase_regime,
        compute_viability_by_distance,
        fit_phase_boundary,
    )

    df = pd.read_parquet(Path(VOLUME_PATH) / data)
    boundaries, v_long, bootstraps = [], [], []
    for ds_id, sub in df.groupby("dataset_id"):
        v = compute_viability_by_distance(sub, min_count_per_distance=min_count_per_distance)
        fit = fit_phase_boundary(v)
        regime = classify_phase_regime(fit["dc"], fit["alpha"], fit.get("r2"), v)
        v["dataset_id"] = ds_id
        v["dc"] = fit["dc"]
        v_long.append(v)
        boundaries.append({"dataset_id": ds_id, "regime": regime, **fit})
        if bootstrap > 0:
            boot = bootstrap_phase_boundary(
                sub, n_boot=bootstrap, min_count_per_distance=min_count_per_distance
            )
            boot["dataset_id"] = ds_id
            bootstraps.append(boot)

    out_path = Path(VOLUME_PATH) / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(boundaries).to_csv(out_path, index=False)
    pd.concat(v_long, ignore_index=True).to_csv(out_path.parent / "viability_by_distance.csv", index=False)
    if bootstraps:
        pd.concat(bootstraps, ignore_index=True).to_csv(
            out_path.parent / "bootstrap_boundaries.csv", index=False
        )
    volume.commit()
    return {"datasets": len(boundaries), "out": str(out_path)}


# ---------- 4. PhaseAgent active query ----------

@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600 * 4)
def run_phaseagent(
    data: str = "processed/all_dms.parquet",
    out: str = "outputs/tables/phaseagent_simulation.csv",
    initial_n: int = 20,
    batch_size: int = 10,
    steps: int = 20,
    repeats: int = 20,
    datasets: str = "",
) -> dict:
    import numpy as np
    import pandas as pd

    from phaseagent.agents import (
        BoundaryGreedyPolicy,
        PhaseAgentPolicy,
        RandomPolicy,
        UncertaintyShellPolicy,
        UniformShellPolicy,
        simulate_boundary_discovery,
    )

    df = pd.read_parquet(Path(VOLUME_PATH) / data)
    ds_filter = [s for s in datasets.split(",") if s.strip()]
    if ds_filter:
        df = df[df["dataset_id"].isin(ds_filter)]

    rng = np.random.default_rng(0)
    policies = [
        RandomPolicy(rng),
        UniformShellPolicy(rng),
        UncertaintyShellPolicy(rng),
        BoundaryGreedyPolicy(rng),
        PhaseAgentPolicy(rng),
    ]
    rows = []
    for ds_id, sub in df.groupby("dataset_id"):
        for pol in policies:
            sim = simulate_boundary_discovery(
                sub.reset_index(drop=True),
                policy=pol,
                initial_n=initial_n,
                batch_size=batch_size,
                n_steps=steps,
                n_repeats=repeats,
            )
            sim["dataset_id"] = ds_id
            rows.append(sim)
    out_df = pd.concat(rows, ignore_index=True)
    out_path = Path(VOLUME_PATH) / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    volume.commit()
    return {"rows": int(len(out_df)), "out": str(out_path)}


# ---------- 5. Phase-aware search ----------

@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600 * 2)
def run_phase_aware_search(
    data: str = "processed/all_dms.parquet",
    boundaries_csv: str = "outputs/tables/phase_boundaries.csv",
    out: str = "outputs/tables/search_benchmark.csv",
    budget: int = 50,
    repeats: int = 20,
    lambda_boundary: float = 0.5,
) -> dict:
    import numpy as np
    import pandas as pd

    from phaseagent.search import run_search_benchmark

    df = pd.read_parquet(Path(VOLUME_PATH) / data)
    bd = pd.read_csv(Path(VOLUME_PATH) / boundaries_csv)
    bd_dict = {
        r["dataset_id"]: {"dc": r["dc"], "alpha": r["alpha"], "dc_true": r["dc"]}
        for _, r in bd.iterrows()
    }
    datasets_dict = {ds: sub.reset_index(drop=True) for ds, sub in df.groupby("dataset_id")}
    seeds = list(range(repeats))
    out_df = run_search_benchmark(
        datasets=datasets_dict,
        boundaries=bd_dict,
        budgets=[budget],
        seeds=seeds,
        lambda_boundary=lambda_boundary,
    )
    out_path = Path(VOLUME_PATH) / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    volume.commit()
    return {"rows": int(len(out_df)), "out": str(out_path)}


# ---------- 6. PLM scoring (GPU) ----------

@app.function(image=gpu_image, volumes={VOLUME_PATH: volume}, gpu="A10G", timeout=3600 * 4)
def run_plm_scoring(
    data: str = "processed/all_dms.parquet",
    out: str = "outputs/tables/plm_scores.csv",
    model_name: str = "esm2_t33_650M_UR50D",
    max_per_dataset: int = 5000,
    batch_size: int = 4,
) -> dict:
    import pandas as pd

    from phaseagent.plm import score_dataset_plm

    df = pd.read_parquet(Path(VOLUME_PATH) / data)
    out_df = score_dataset_plm(
        df,
        model_name=model_name,
        max_per_dataset=max_per_dataset,
        batch_size=batch_size,
    )
    out_path = Path(VOLUME_PATH) / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    volume.commit()
    return {"rows": int(len(out_df)), "out": str(out_path)}


# ---------- 7. Make figures ----------

@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=1800)
def make_figures(
    tables_dir: str = "outputs/tables",
    figures_dir: str = "outputs/figures",
    data: str = "processed/all_dms.parquet",
) -> dict:
    from phaseagent.plots import make_all_figures

    tab = Path(VOLUME_PATH) / tables_dir
    fig = Path(VOLUME_PATH) / figures_dir
    fig.mkdir(parents=True, exist_ok=True)
    n = make_all_figures(tab, fig, Path(VOLUME_PATH) / data)
    volume.commit()
    return {"figures": n, "out": str(fig)}


# ---------- 8. Spectral PhaseAgent advanced path ----------

@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600)
def build_spectra(
    data: str = "processed/all_dms.parquet",
    out_dir: str = "outputs/spectral",
) -> dict:
    import pandas as pd

    from phaseagent.spectrum import (
        build_spectrum_tokens,
        compute_single_mutant_effects,
        spectrum_summary_features,
    )

    df = pd.read_parquet(Path(VOLUME_PATH) / data)
    out = Path(VOLUME_PATH) / out_dir
    out.mkdir(parents=True, exist_ok=True)
    summaries, tokens, effects = [], [], []
    for ds_id, sub in df.groupby("dataset_id"):
        summaries.append(spectrum_summary_features(sub))
        tok = build_spectrum_tokens(sub)
        eff = compute_single_mutant_effects(sub)
        tok["dataset_id"] = ds_id
        eff["dataset_id"] = ds_id
        tokens.append(tok)
        effects.append(eff)
    pd.DataFrame(summaries).to_csv(out / "spectrum_summary.csv", index=False)
    pd.concat(tokens, ignore_index=True).to_parquet(out / "spectrum_tokens.parquet", index=False)
    pd.concat(effects, ignore_index=True).to_parquet(out / "single_mutant_effects.parquet", index=False)
    volume.commit()
    return {"datasets": len(summaries), "out": str(out)}


@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600 * 2)
def run_large_deviation(
    data: str = "processed/all_dms.parquet",
    out: str = "outputs/tables/large_deviation_survival.csv",
    n_mc: int = 20_000,
    min_shells: int = 2,
) -> dict:
    import pandas as pd

    from phaseagent.eval_survival import evaluate_survival_prediction, true_survival_curve
    from phaseagent.large_deviation import additive_survival_curve

    df = pd.read_parquet(Path(VOLUME_PATH) / data)
    curves, metrics = [], []
    for ds_id, sub in df.groupby("dataset_id"):
        if sub["mutation_distance"].nunique() < min_shells:
            continue
        depths = range(0, int(sub["mutation_distance"].max()) + 1)
        curve = additive_survival_curve(sub, depths=depths, n_mc=n_mc)
        curve["dataset_id"] = ds_id
        true = true_survival_curve(sub)
        curve = curve.merge(
            true[["mutation_distance", "survival_true", "n", "stderr"]],
            on="mutation_distance",
            how="left",
        )
        curves.append(curve)
        pred = curve.rename(columns={"survival_large_deviation": "survival_pred"})
        metric = evaluate_survival_prediction(pred, true, pred_col="survival_pred")
        metrics.append({"dataset_id": ds_id, **metric})
    out_path = Path(VOLUME_PATH) / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if curves:
        pd.concat(curves, ignore_index=True).to_csv(out_path, index=False)
    else:
        pd.DataFrame().to_csv(out_path, index=False)
    pd.DataFrame(metrics).to_csv(out_path.parent / "large_deviation_metrics.csv", index=False)
    volume.commit()
    return {"datasets": len(curves), "out": str(out_path)}


@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600 * 2)
def run_survival_search(
    data: str = "processed/all_dms.parquet",
    survival_curves: str = "outputs/tables/large_deviation_survival.csv",
    boundaries_csv: str = "outputs/tables/phase_boundaries.csv",
    out: str = "outputs/tables/survival_search_frontier.csv",
    survival_col: str = "survival_large_deviation",
    budget: int = 50,
    repeats: int = 5,
    beta_grid: str = "0,0.01,0.03,0.05",
    gamma_grid: str = "0.1,0.3,0.5,1.0",
) -> dict:
    import pandas as pd

    from phaseagent.survival_search import run_survival_search_benchmark

    def parse_grid(text: str) -> list[float]:
        return [float(x) for x in text.split(",") if x.strip()]

    df = pd.read_parquet(Path(VOLUME_PATH) / data)
    curves = pd.read_csv(Path(VOLUME_PATH) / survival_curves)
    boundaries = {}
    bd_path = Path(VOLUME_PATH) / boundaries_csv
    if bd_path.exists():
        bd = pd.read_csv(bd_path)
        boundaries = {
            r["dataset_id"]: {"dc": r["dc"], "alpha": r["alpha"], "dc_true": r["dc"]}
            for _, r in bd.iterrows()
        }
    datasets = {ds: sub.reset_index(drop=True) for ds, sub in df.groupby("dataset_id")}
    curve_dict = {ds: sub.reset_index(drop=True) for ds, sub in curves.groupby("dataset_id")}
    out_df = run_survival_search_benchmark(
        datasets=datasets,
        survival_curves=curve_dict,
        boundaries=boundaries,
        budget=budget,
        seeds=range(repeats),
        beta_grid=parse_grid(beta_grid),
        gamma_grid=parse_grid(gamma_grid),
        survival_col=survival_col,
    )
    out_path = Path(VOLUME_PATH) / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    volume.commit()
    return {"rows": int(len(out_df)), "out": str(out_path)}


@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600 * 4)
def run_active_spectral_phaseagent(
    data: str = "processed/all_dms.parquet",
    out: str = "outputs/tables/active_spectral_phaseagent.csv",
    policy_name: str = "mutual_information_phaseagent",
    initial_n: int = 20,
    batch_size: int = 10,
    steps: int = 20,
    repeats: int = 10,
    datasets: str = "",
) -> dict:
    import pandas as pd

    from phaseagent.active_phaseagent import POLICIES
    from phaseagent.agents import simulate_boundary_discovery

    class FunctionPolicy:
        def __init__(self, name, fn):
            self.name = name
            self.fn = fn

        def select_batch(self, observed_df, pool_df, batch_size):
            try:
                return self.fn(observed_df, pool_df, batch_size, seed=0)
            except TypeError:
                return self.fn(observed_df, pool_df, batch_size)

    df = pd.read_parquet(Path(VOLUME_PATH) / data)
    ds_filter = [s for s in datasets.split(",") if s.strip()]
    if ds_filter:
        df = df[df["dataset_id"].isin(ds_filter)]
    policy = FunctionPolicy(policy_name, POLICIES[policy_name])
    rows = []
    for ds_id, sub in df.groupby("dataset_id"):
        if sub["mutation_distance"].nunique() < 3:
            continue
        sim = simulate_boundary_discovery(
            sub.reset_index(drop=True),
            policy=policy,
            initial_n=initial_n,
            batch_size=batch_size,
            n_steps=steps,
            n_repeats=repeats,
        )
        sim["dataset_id"] = ds_id
        rows.append(sim)
    out_path = Path(VOLUME_PATH) / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out_df.to_csv(out_path, index=False)
    volume.commit()
    return {"rows": int(len(out_df)), "out": str(out_path)}


@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=1800)
def make_advanced_figures(
    data: str = "processed/all_dms.parquet",
    single_effects: str = "outputs/spectral/single_mutant_effects.parquet",
    survival_curves: str = "outputs/tables/large_deviation_survival.csv",
    search_results: str = "outputs/tables/survival_search_frontier.csv",
    out_dir: str = "outputs/figures_advanced",
) -> dict:
    import pandas as pd

    from phaseagent.plots_advanced import (
        plot_distance_shell_histogram,
        plot_search_frontier,
        plot_spectrum_histograms,
        plot_survival_curves,
    )

    out = Path(VOLUME_PATH) / out_dir
    out.mkdir(parents=True, exist_ok=True)
    made = []
    data_path = Path(VOLUME_PATH) / data
    if data_path.exists():
        made.append(plot_distance_shell_histogram(pd.read_parquet(data_path), out / "fig1_distance_shell_histogram.png"))
    effects_path = Path(VOLUME_PATH) / single_effects
    if effects_path.exists():
        made.append(plot_spectrum_histograms(pd.read_parquet(effects_path), out / "fig2_spectrum_histograms.png"))
    curves_path = Path(VOLUME_PATH) / survival_curves
    if curves_path.exists():
        curves = pd.read_csv(curves_path)
        true_col = "survival_true" if "survival_true" in curves.columns else "survival_additive_mc"
        made.append(plot_survival_curves(curves, out / "fig3_survival_curves.png", true_col=true_col, pred_col="survival_large_deviation"))
    search_path = Path(VOLUME_PATH) / search_results
    if search_path.exists():
        made.append(plot_search_frontier(pd.read_csv(search_path), out / "fig7_search_frontier.png"))
    volume.commit()
    return {"figures": len(made), "out": str(out)}


# ---------- 9. EditGuard-Diff ICML path ----------

@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=1800)
def build_edit_splits(
    data: str = "processed/all_dms.parquet",
    out: str = "outputs/editguard/edit_splits.csv",
    seed: int = 0,
    train_frac: float = 0.5,
    val_frac: float = 0.2,
    clinical_holdout: bool = True,
) -> dict:
    """Build dataset-level train/val/test splits.

    Defaults updated for the post-audit pipeline: ``train_frac=0.5,
    val_frac=0.2`` lands roughly 50/20/30 instead of the legacy 70/15/15,
    so the held-out test set has enough datasets for paired Wilcoxon to be
    informative.
    """
    import pandas as pd

    from phaseagent.edit_splits import make_dataset_splits

    df = pd.read_parquet(Path(VOLUME_PATH) / data)
    splits = make_dataset_splits(
        df["dataset_id"].unique(),
        seed=seed,
        train_frac=train_frac,
        val_frac=val_frac,
        clinical_holdout=clinical_holdout,
    )
    out_path = Path(VOLUME_PATH) / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(out_path, index=False)
    volume.commit()
    summary = splits["split"].value_counts().to_dict()
    return {"rows": int(len(splits)), "split_counts": summary, "out": str(out_path)}


@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=1800)
def build_edit_tasks(
    data: str = "processed/all_dms.parquet",
    out: str = "outputs/editguard/edit_tasks.csv",
    budgets: str = "1,2,3,5",
    objectives: str = "novelty,fragility_aware,motif_avoidance",
    min_candidates: int = 20,
    splits_csv: str = "outputs/editguard/edit_splits.csv",
    split: Optional[str] = None,
) -> dict:
    """Build editing tasks. If ``split`` is given (train/val/test/clinical_test),
    only datasets in that split contribute tasks; output filename gets a split
    suffix unless ``out`` was explicitly specified by the caller."""
    import pandas as pd

    from phaseagent.editing_tasks import build_editing_tasks

    def parse_ints(text: str) -> list[int]:
        return [int(x) for x in text.split(",") if x.strip()]

    def parse_strs(text: str) -> list[str]:
        return [x.strip() for x in text.split(",") if x.strip()]

    df = pd.read_parquet(Path(VOLUME_PATH) / data)
    splits_df = None
    splits_path = Path(VOLUME_PATH) / splits_csv
    if splits_path.exists():
        splits_df = pd.read_csv(splits_path)
    elif split is not None:
        raise FileNotFoundError(
            f"split={split!r} requested but {splits_path} missing; run build_edit_splits first"
        )
    tasks = build_editing_tasks(
        df,
        budgets=parse_ints(budgets),
        objectives=parse_strs(objectives),
        min_candidates=min_candidates,
        splits=splits_df,
        split_filter=split,
    )
    if split is not None and out == "outputs/editguard/edit_tasks.csv":
        out = f"outputs/editguard/edit_tasks_{split}.csv"
    out_path = Path(VOLUME_PATH) / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tasks.to_csv(out_path, index=False)
    volume.commit()
    return {"rows": int(len(tasks)), "out": str(out_path)}


@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600 * 2)
def train_editguard_prior(
    data: str = "processed/all_dms.parquet",
    splits_csv: str = "outputs/editguard/edit_splits.csv",
    model_out: str = "outputs/editguard/editguard_prior.pkl",
    metrics_out: str = "outputs/editguard/editguard_prior_metrics.csv",
    n_estimators: int = 200,
) -> dict:
    import pandas as pd

    from phaseagent.edit_splits import add_splits
    from phaseagent.editguard_prior import DMSFunctionPrior, evaluate_prior

    df = pd.read_parquet(Path(VOLUME_PATH) / data)
    splits = pd.read_csv(Path(VOLUME_PATH) / splits_csv)
    labeled = add_splits(df, splits)
    train = labeled[labeled["split"] == "train"]
    val = labeled[labeled["split"] == "val"]
    prior = DMSFunctionPrior(n_estimators=n_estimators).fit(train, calibrate_df=val)
    model_path = Path(VOLUME_PATH) / model_out
    prior.save(model_path)
    rows = []
    for split, sub in labeled.groupby("split"):
        rows.append({"split": split, **evaluate_prior(prior, sub)})
    metrics_path = Path(VOLUME_PATH) / metrics_out
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    volume.commit()
    return {"model": str(model_path), "metrics": str(metrics_path)}


@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600 * 2)
def run_edit_baselines(
    data: str = "processed/all_dms.parquet",
    tasks_csv: str = "outputs/editguard/edit_tasks.csv",
    prior_path: str = "outputs/editguard/editguard_prior.pkl",
    out: str = "outputs/editguard/edit_baseline_metrics.csv",
    selections_out: str = "outputs/editguard/edit_baseline_selections.parquet",
    external_candidates: Optional[str] = None,
    k: int = 50,
    seeds: int = 5,
    max_tasks: int = 0,
    max_candidates: int = 10000,
    splits_csv: str = "outputs/editguard/edit_splits.csv",
    require_split: Optional[str] = None,
) -> dict:
    import pandas as pd

    from phaseagent.edit_baselines import run_edit_baselines as run_one
    from phaseagent.edit_eval import evaluate_methods
    from phaseagent.edit_splits import assert_tasks_in_split
    from phaseagent.editing_tasks import candidate_pool_for_task, task_from_row
    from phaseagent.editguard_prior import DMSFunctionPrior

    df = pd.read_parquet(Path(VOLUME_PATH) / data)
    tasks = pd.read_csv(Path(VOLUME_PATH) / tasks_csv)
    if require_split is not None:
        splits_path = Path(VOLUME_PATH) / splits_csv
        if not splits_path.exists():
            raise FileNotFoundError(
                f"require_split={require_split!r} but {splits_path} missing"
            )
        assert_tasks_in_split(tasks, pd.read_csv(splits_path), require_split)
    if max_tasks > 0:
        tasks = tasks.head(max_tasks)
    prior = DMSFunctionPrior.load(Path(VOLUME_PATH) / prior_path)
    external = None
    if external_candidates:
        ext_path = Path(VOLUME_PATH) / external_candidates
        external = pd.read_parquet(ext_path) if ext_path.suffix == ".parquet" else pd.read_csv(ext_path)
    metric_rows, selection_rows = [], []
    for task_idx, row in tasks.iterrows():
        task = task_from_row(row)
        pool = candidate_pool_for_task(df, task)
        if len(pool) == 0:
            continue
        for seed in range(seeds):
            if max_candidates > 0 and len(pool) > max_candidates:
                pool_seed = int(task_idx) * 1000 + seed
                pool_run = pool.sample(max_candidates, random_state=pool_seed)
            else:
                pool_run = pool
            selections = run_one(pool_run, task, prior, k=k, seed=seed, external_candidates=external)
            metrics = evaluate_methods(selections, task)
            metrics["task_idx"] = int(task_idx)
            metrics["dataset_id"] = task.dataset_id
            metrics["objective"] = task.objective
            metrics["edit_budget"] = task.edit_budget
            metrics["seed"] = seed
            metric_rows.append(metrics)
            for method, sel in selections.items():
                tmp = sel.copy()
                tmp["task_idx"] = int(task_idx)
                tmp["method"] = method
                tmp["seed"] = seed
                selection_rows.append(tmp)
    out_df = pd.concat(metric_rows, ignore_index=True) if metric_rows else pd.DataFrame()
    out_path = Path(VOLUME_PATH) / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    if selection_rows:
        pd.concat(selection_rows, ignore_index=True).to_parquet(Path(VOLUME_PATH) / selections_out, index=False)
    volume.commit()
    return {"rows": int(len(out_df)), "out": str(out_path)}


@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600 * 2)
def run_editguard_diffusion(
    data: str = "processed/all_dms.parquet",
    tasks_csv: str = "outputs/editguard/edit_tasks.csv",
    prior_path: str = "outputs/editguard/editguard_prior.pkl",
    out: str = "outputs/editguard/dms_pool_guided_metrics.csv",
    selections_out: str = "outputs/editguard/dms_pool_guided_selections.parquet",
    k: int = 50,
    seeds: int = 5,
    max_tasks: int = 0,
    max_candidates: int = 10000,
    splits_csv: str = "outputs/editguard/edit_splits.csv",
    require_split: Optional[str] = None,
) -> dict:
    """Run the DMS-pool guided sampler. The output method label is
    ``dms_pool_guided`` — this is a selection baseline, not a generative model.
    The Modal function name is kept for backwards compatibility with existing
    pipelines; ``editguard_diffusion`` is reserved for the real DPLM-backed
    sampler implemented in Phase 2 of the plan."""
    import pandas as pd

    from phaseagent.edit_eval import evaluate_edit_selection
    from phaseagent.edit_splits import assert_tasks_in_split
    from phaseagent.editing_tasks import candidate_pool_for_task, task_from_row
    from phaseagent.editguard_diffusion import DMSPoolGuidedSampler, DiffusionSampleConfig
    from phaseagent.editguard_prior import DMSFunctionPrior

    df = pd.read_parquet(Path(VOLUME_PATH) / data)
    tasks = pd.read_csv(Path(VOLUME_PATH) / tasks_csv)
    if require_split is not None:
        splits_path = Path(VOLUME_PATH) / splits_csv
        if not splits_path.exists():
            raise FileNotFoundError(
                f"require_split={require_split!r} but {splits_path} missing"
            )
        assert_tasks_in_split(tasks, pd.read_csv(splits_path), require_split)
    if max_tasks > 0:
        tasks = tasks.head(max_tasks)
    prior = DMSFunctionPrior.load(Path(VOLUME_PATH) / prior_path)
    sampler = DMSPoolGuidedSampler(prior)
    metric_rows, selection_rows = [], []
    for task_idx, row in tasks.iterrows():
        task = task_from_row(row)
        pool = candidate_pool_for_task(df, task)
        if len(pool) == 0:
            continue
        for seed in range(seeds):
            if max_candidates > 0 and len(pool) > max_candidates:
                pool_seed = int(task_idx) * 1000 + seed
                pool_run = pool.sample(max_candidates, random_state=pool_seed)
            else:
                pool_run = pool
            selected = sampler.sample(pool_run, task, DiffusionSampleConfig(n_samples=k, seed=seed))
            metric_rows.append(
                {
                    "task_idx": int(task_idx),
                    "dataset_id": task.dataset_id,
                    "objective": task.objective,
                    "edit_budget": task.edit_budget,
                    "seed": seed,
                    "method": "dms_pool_guided",
                    **evaluate_edit_selection(selected, task),
                }
            )
            tmp = selected.copy()
            tmp["task_idx"] = int(task_idx)
            tmp["seed"] = seed
            selection_rows.append(tmp)
    out_path = Path(VOLUME_PATH) / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(out_path, index=False)
    if selection_rows:
        pd.concat(selection_rows, ignore_index=True).to_parquet(Path(VOLUME_PATH) / selections_out, index=False)
    volume.commit()
    return {"rows": int(len(metric_rows)), "out": str(out_path)}


# ---------- Real ESM-2 masked-marginal baseline (Phase 2.1) ----------

@app.function(image=gpu_image, volumes={VOLUME_PATH: volume}, gpu="A10G", timeout=3600 * 4)
def run_real_esm_baseline(
    data: str = "processed/all_dms.parquet",
    tasks_csv: str = "outputs/editguard/edit_tasks_test.csv",
    prior_path: str = "outputs/editguard/editguard_prior.pkl",
    out: str = "outputs/editguard/esm2_masked_marginal_metrics.csv",
    selections_out: str = "outputs/editguard/esm2_masked_marginal_selections.parquet",
    k: int = 50,
    seeds: int = 5,
    max_tasks: int = 0,
    max_candidates: int = 10000,
    splits_csv: str = "outputs/editguard/edit_splits.csv",
    require_split: Optional[str] = "test",
    model_name: str = "esm2_t33_650M_UR50D",
    esm_batch_size: int = 4,
    dms_rerank_pool: int = 1000,
) -> dict:
    """Real ESM-2 masked-marginal baseline on a measured DMS pool.

    Two methods emitted, both selecting from the same DMS candidate pool that
    ``run_edit_baselines`` and ``run_editguard_diffusion`` use, so the
    comparison is apples-to-apples:

    1. ``esm2_masked_marginal``: score every candidate by the mean of
       per-token ESM-2 masked-marginal log-probabilities on the WT sequence,
       take top-k.
    2. ``esm2_masked_marginal_dms_rerank``: take the top
       ``dms_rerank_pool`` ESM candidates, rerank by the DMS function prior,
       take top-k. Tests whether ESM's naturalness ranking improves once
       paired with DMS guidance.

    Always evaluates on the test split unless ``require_split`` is overridden.
    Defaults to ``esm2_t33_650M_UR50D`` (650M params, A10G is sufficient).
    """
    import numpy as np
    import pandas as pd

    from phaseagent.edit_eval import evaluate_edit_selection
    from phaseagent.edit_generators import infer_wildtype_sequence
    from phaseagent.edit_splits import assert_tasks_in_split
    from phaseagent.editing_tasks import candidate_pool_for_task, task_from_row
    from phaseagent.editguard_prior import DMSFunctionPrior
    from phaseagent.mutations import parse_mutation_notation
    from phaseagent.plm import _load_esm, masked_token_log_probs

    df = pd.read_parquet(Path(VOLUME_PATH) / data)
    tasks = pd.read_csv(Path(VOLUME_PATH) / tasks_csv)
    if require_split is not None:
        splits_path = Path(VOLUME_PATH) / splits_csv
        if not splits_path.exists():
            raise FileNotFoundError(
                f"require_split={require_split!r} but {splits_path} missing"
            )
        assert_tasks_in_split(tasks, pd.read_csv(splits_path), require_split)
    if max_tasks > 0:
        tasks = tasks.head(max_tasks)

    prior_full = Path(VOLUME_PATH) / prior_path
    prior = DMSFunctionPrior.load(prior_full) if prior_full.exists() else None

    print(f"[esm-baseline] loading {model_name} once for all tasks")
    model, alphabet = _load_esm(model_name)

    # Cache: per-dataset {token: log_prob} so the same protein's tokens are
    # only scored once across all tasks/seeds.
    token_cache: dict[str, dict[str, float]] = {}

    def get_scores_for_pool(ds_id: str, wt: str, pool_df: pd.DataFrame) -> dict[str, float]:
        """Return token scores; only call ESM on tokens not already cached."""
        cache = token_cache.setdefault(ds_id, {})
        wanted = set()
        for notation in pool_df["mutation_notation"].astype(str):
            for tok in parse_mutation_notation(notation):
                wanted.add(tok)
        missing = sorted(wanted - cache.keys())
        if missing:
            scores = masked_token_log_probs(
                wt, missing, model, alphabet, batch_size=esm_batch_size
            )
            cache.update(scores)
        return cache

    wt_cache: dict[str, str] = {}
    metric_rows, selection_rows = [], []
    for task_idx, row in tasks.iterrows():
        task = task_from_row(row)
        pool = candidate_pool_for_task(df, task)
        if len(pool) == 0:
            continue
        wt = wt_cache.get(task.dataset_id)
        if wt is None:
            wt = infer_wildtype_sequence(df, task.dataset_id)
            if wt is None:
                print(f"[esm-baseline] {task.dataset_id}: cannot infer WT, skipping")
                continue
            wt_cache[task.dataset_id] = wt
            print(f"[esm-baseline] {task.dataset_id}: WT length {len(wt)}")
        for seed in range(seeds):
            if max_candidates > 0 and len(pool) > max_candidates:
                pool_seed = int(task_idx) * 1000 + seed
                pool_run = pool.sample(max_candidates, random_state=pool_seed).reset_index(drop=True)
            else:
                pool_run = pool.reset_index(drop=True)

            cache = get_scores_for_pool(task.dataset_id, wt, pool_run)
            esm_scores = []
            for notation in pool_run["mutation_notation"].astype(str):
                toks = parse_mutation_notation(notation)
                if not toks or any(t not in cache for t in toks):
                    esm_scores.append(float("nan"))
                else:
                    esm_scores.append(float(np.mean([cache[t] for t in toks])))
            scored = pool_run.assign(esm_masked_score=esm_scores).dropna(subset=["esm_masked_score"])
            if len(scored) == 0:
                continue

            esm_top = scored.nlargest(min(k, len(scored)), "esm_masked_score").copy()
            esm_top["method"] = "esm2_masked_marginal"

            rerank_n = max(min(dms_rerank_pool, len(scored)), k)
            esm_pool = scored.nlargest(rerank_n, "esm_masked_score").copy()
            if prior is not None and len(esm_pool):
                esm_pool["prior_function_prob"] = prior.predict_proba(esm_pool)
                dms_top = esm_pool.nlargest(min(k, len(esm_pool)), "prior_function_prob").copy()
                dms_top["method"] = "esm2_masked_marginal_dms_rerank"
            else:
                dms_top = esm_top.copy()
                dms_top["method"] = "esm2_masked_marginal_dms_rerank"

            for sel in (esm_top, dms_top):
                method = sel["method"].iloc[0]
                metrics = evaluate_edit_selection(sel, task)
                metric_rows.append(
                    {
                        "task_idx": int(task_idx),
                        "dataset_id": task.dataset_id,
                        "objective": task.objective,
                        "edit_budget": task.edit_budget,
                        "seed": seed,
                        "method": method,
                        **metrics,
                    }
                )
                tmp = sel.copy()
                tmp["task_idx"] = int(task_idx)
                tmp["seed"] = seed
                selection_rows.append(tmp)

    out_path = Path(VOLUME_PATH) / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(out_path, index=False)
    if selection_rows:
        pd.concat(selection_rows, ignore_index=True).to_parquet(
            Path(VOLUME_PATH) / selections_out, index=False
        )
    volume.commit()
    return {"rows": int(len(metric_rows)), "out": str(out_path), "model": model_name}


@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600)
def run_calibration_on_candidate_pools(
    data: str = "processed/all_dms.parquet",
    tasks_csv: str = "outputs/editguard/edit_tasks_test.csv",
    prior_path: str = "outputs/editguard/editguard_prior.pkl",
    metrics_out: str = "outputs/editguard/editguard_prior_calibration_pools.csv",
    bins_out: str = "outputs/editguard/editguard_prior_calibration_pools_bins.csv",
    figure_out: str = "outputs/editguard/figures/fig_calibration_pools.png",
    n_bins: int = 15,
    splits_csv: str = "outputs/editguard/edit_splits.csv",
    require_split: Optional[str] = "test",
) -> dict:
    """Compute calibration on the editing **candidate pools** the prior is
    actually used to score in benchmark tasks.

    More operationally relevant than calibration on the full DMS distribution
    because the candidate pool is filtered to ``mutation_distance ≤ edit_budget``
    and respects task constraints — exactly what scoring sees at edit time.
    """
    import pandas as pd

    from phaseagent.calibration import (
        evaluate_calibration_per_dataset,
        evaluate_calibration_per_split,
        plot_reliability_diagram,
    )
    from phaseagent.edit_splits import assert_tasks_in_split
    from phaseagent.editing_tasks import candidate_pool_for_task, task_from_row
    from phaseagent.editguard_prior import DMSFunctionPrior

    df = pd.read_parquet(Path(VOLUME_PATH) / data)
    tasks = pd.read_csv(Path(VOLUME_PATH) / tasks_csv)
    if require_split is not None:
        splits_path = Path(VOLUME_PATH) / splits_csv
        if not splits_path.exists():
            raise FileNotFoundError(
                f"require_split={require_split!r} but {splits_path} missing"
            )
        assert_tasks_in_split(tasks, pd.read_csv(splits_path), require_split)

    prior = DMSFunctionPrior.load(Path(VOLUME_PATH) / prior_path)

    pool_rows = []
    seen = set()
    for _, row in tasks.iterrows():
        task = task_from_row(row)
        pool = candidate_pool_for_task(df, task)
        if len(pool) == 0:
            continue
        # Dedupe by (dataset_id, mutation_notation) so a variant that appears
        # in multiple tasks counts once.
        for _, r in pool.iterrows():
            key = (str(r["dataset_id"]), str(r["mutation_notation"]))
            if key in seen:
                continue
            seen.add(key)
            pool_rows.append(r)
    pooled = pd.DataFrame(pool_rows)
    if len(pooled) == 0:
        raise RuntimeError("no candidate-pool variants assembled across tasks")
    pooled["pred_p"] = prior.predict_proba(pooled)
    pooled["split"] = require_split or "all"

    metrics, bin_tables = evaluate_calibration_per_split(
        pooled, pred_col="pred_p", label_col="viable", n_bins=n_bins
    )
    per_ds = evaluate_calibration_per_dataset(
        pooled, pred_col="pred_p", label_col="viable", n_bins=n_bins
    )

    metrics_path = Path(VOLUME_PATH) / metrics_out
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_combined = pd.concat([metrics, per_ds.rename(columns={"dataset_id": "split"})], ignore_index=True)
    metrics_combined.to_csv(metrics_path, index=False)

    bins_long = []
    for split_name, b in bin_tables.items():
        if len(b) == 0:
            continue
        bb = b.copy()
        bb["split"] = split_name
        bins_long.append(bb)
    bins_path = Path(VOLUME_PATH) / bins_out
    if bins_long:
        pd.concat(bins_long, ignore_index=True).to_csv(bins_path, index=False)
    else:
        pd.DataFrame().to_csv(bins_path, index=False)

    figure_path = Path(VOLUME_PATH) / figure_out
    plot_reliability_diagram(
        bin_tables,
        figure_path,
        title=f"DMS function prior — reliability on candidate pools ({require_split})",
        splits_to_plot=[require_split or "all", "pooled"],
    )
    volume.commit()

    print("[calib-pools] metrics:")
    print(metrics_combined.to_string(index=False))
    return {
        "metrics": str(metrics_path),
        "bins": str(bins_path),
        "figure": str(figure_path),
        "pooled_ece": float(metrics.set_index("split").loc["pooled", "ece"]),
    }


@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600)
def run_calibration_eval(
    data: str = "processed/all_dms.parquet",
    splits_csv: str = "outputs/editguard/edit_splits.csv",
    prior_path: str = "outputs/editguard/editguard_prior.pkl",
    metrics_out: str = "outputs/editguard/editguard_prior_calibration.csv",
    per_dataset_out: str = "outputs/editguard/editguard_prior_calibration_per_dataset.csv",
    bins_out: str = "outputs/editguard/editguard_prior_calibration_bins.csv",
    figure_out: str = "outputs/editguard/figures/fig_calibration_reliability.png",
    n_bins: int = 15,
    sample_per_split: int = 200_000,
    seed: int = 0,
) -> dict:
    """Compute calibration metrics for the trained DMS function prior.

    Predicts on every variant in train/val/test splits (sub-sampled per
    split for speed; default 200k rows per split is enough for stable ECE),
    computes ECE / Brier / log-loss per split, per-dataset, and pooled,
    and renders a reliability diagram. Defends claim C3 from the
    implementation plan.
    """
    import pandas as pd

    from phaseagent.calibration import (
        evaluate_calibration_per_dataset,
        evaluate_calibration_per_split,
        plot_reliability_diagram,
    )
    from phaseagent.edit_splits import add_splits
    from phaseagent.editguard_prior import DMSFunctionPrior

    df = pd.read_parquet(Path(VOLUME_PATH) / data)
    splits = pd.read_csv(Path(VOLUME_PATH) / splits_csv)
    labeled = add_splits(df, splits)
    prior = DMSFunctionPrior.load(Path(VOLUME_PATH) / prior_path)

    # Subsample per split for tractable inference (the prior is 130 MB and we
    # have ~700k rows total). 200k per split gives stable ECE at 15 bins.
    parts = []
    for split, sub in labeled.groupby("split"):
        if sample_per_split > 0 and len(sub) > sample_per_split:
            sub = sub.sample(sample_per_split, random_state=seed)
        parts.append(sub)
    work = pd.concat(parts, ignore_index=True).copy()

    print(f"[calib] predicting on {len(work)} variants across splits "
          f"{sorted(work['split'].unique())}")
    work["pred_p"] = prior.predict_proba(work)

    metrics, bin_tables = evaluate_calibration_per_split(
        work, pred_col="pred_p", label_col="viable", n_bins=n_bins
    )
    per_ds = evaluate_calibration_per_dataset(
        work, pred_col="pred_p", label_col="viable", n_bins=n_bins
    )

    metrics_path = Path(VOLUME_PATH) / metrics_out
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)

    per_ds_path = Path(VOLUME_PATH) / per_dataset_out
    per_ds.to_csv(per_ds_path, index=False)

    bins_long = []
    for split_name, b in bin_tables.items():
        if len(b) == 0:
            continue
        bb = b.copy()
        bb["split"] = split_name
        bins_long.append(bb)
    bins_path = Path(VOLUME_PATH) / bins_out
    if bins_long:
        pd.concat(bins_long, ignore_index=True).to_csv(bins_path, index=False)
    else:
        pd.DataFrame().to_csv(bins_path, index=False)

    figure_path = Path(VOLUME_PATH) / figure_out
    plot_reliability_diagram(bin_tables, figure_path,
                             title="DMS function prior — reliability (test/val/train/pooled)")
    volume.commit()

    print("[calib] per-split metrics:")
    print(metrics.to_string(index=False))
    return {
        "metrics": str(metrics_path),
        "per_dataset": str(per_ds_path),
        "bins": str(bins_path),
        "figure": str(figure_path),
        "test_ece": float(metrics.set_index("split").loc["test", "ece"]) if "test" in set(metrics["split"]) else float("nan"),
    }


@app.function(image=gpu_image, volumes={VOLUME_PATH: volume}, gpu="A10G", timeout=3600 * 6)
def precompute_esm_if_scores(
    data: str = "processed/all_dms.parquet",
    structures_dir: str = "structures",
    out_dir: str = "outputs/editguard/esm_if_scores",
    splits_csv: str = "outputs/editguard/edit_splits.csv",
    require_split: Optional[str] = "test",
    max_per_dataset: int = 0,
    overwrite: bool = False,
) -> dict:
    """Precompute ESM-IF1 scores for every unique variant in each dataset's
    full DMS pool, once per dataset. Caches as
    ``outputs/editguard/esm_if_scores/{dataset_id}.parquet`` on the Modal volume.

    Per-variant scoring is wrapped in a try/except so one bad sequence
    cannot kill the whole run. Writes per-dataset progress and commits the
    volume after each dataset so partial work is recoverable.
    """
    import numpy as np
    import pandas as pd

    from phaseagent.edit_splits import filter_by_split
    from phaseagent.mutations import parse_mutation_notation
    from phaseagent.structure_baselines import (
        _load_esm_if,
        load_backbone_coords,
        per_residue_log_probs,
        score_variant_per_position,
    )

    df = pd.read_parquet(Path(VOLUME_PATH) / data)
    if require_split:
        splits_path = Path(VOLUME_PATH) / splits_csv
        if not splits_path.exists():
            raise FileNotFoundError(f"missing {splits_path}")
        df = filter_by_split(df, pd.read_csv(splits_path), require_split)
    out_root = Path(VOLUME_PATH) / out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    structures_path = Path(VOLUME_PATH) / structures_dir

    print(f"[esm-if-precompute] loading model")
    model, alphabet = _load_esm_if()

    summary = {}
    for ds_id, sub in df.groupby("dataset_id"):
        ds_out = out_root / f"{ds_id}.parquet"
        if ds_out.exists() and not overwrite:
            existing = pd.read_parquet(ds_out)
            summary[ds_id] = {"status": "cached", "n_scored": int(len(existing))}
            print(f"[esm-if-precompute] {ds_id}: cached {len(existing)} variants")
            continue
        pdb_path = structures_path / f"{ds_id}.pdb"
        if not pdb_path.exists():
            summary[ds_id] = {"status": "no_pdb"}
            print(f"[esm-if-precompute] {ds_id}: no PDB; skipping")
            continue
        coords, native_seq = load_backbone_coords(str(pdb_path), chain="A")
        unique = sub.drop_duplicates(["mutation_notation"]).copy()
        unique = unique[
            unique["mutated_sequence"].notna()
            & (unique["mutated_sequence"].astype(str).str.len() == len(native_seq))
            & (unique["mutated_sequence"].astype(str).str.contains(r"\*", regex=True) == False)  # noqa: E712
        ]
        if max_per_dataset > 0 and len(unique) > max_per_dataset:
            unique = unique.sample(max_per_dataset, random_state=0)
        print(f"[esm-if-precompute] {ds_id}: scoring {len(unique)} unique variants "
              f"(WT length {len(native_seq)})")

        rows = []
        n = len(unique)
        for i, (_, row) in enumerate(unique.iterrows()):
            seq = str(row["mutated_sequence"])
            try:
                per_pos = per_residue_log_probs(model, alphabet, coords, seq)
                toks = parse_mutation_notation(str(row["mutation_notation"]))
                positions = []
                for tok in toks:
                    try:
                        positions.append(int(tok[1:-1]))
                    except ValueError:
                        continue
                s_pos, s_full = score_variant_per_position(None, per_pos, positions)
                rows.append(
                    {
                        "dataset_id": ds_id,
                        "mutation_notation": row["mutation_notation"],
                        "esm_if_score": s_pos,
                        "esm_if_score_full": s_full,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "dataset_id": ds_id,
                        "mutation_notation": row["mutation_notation"],
                        "esm_if_score": float("nan"),
                        "esm_if_score_full": float("nan"),
                        "error": f"{type(exc).__name__}:{exc}",
                    }
                )
            if (i + 1) % 200 == 0:
                print(f"[esm-if-precompute] {ds_id}: {i+1}/{n}")
        scored = pd.DataFrame(rows)
        ds_out.parent.mkdir(parents=True, exist_ok=True)
        scored.to_parquet(ds_out, index=False)
        n_ok = int(scored["esm_if_score"].notna().sum())
        summary[ds_id] = {
            "status": "scored",
            "n_unique": int(len(unique)),
            "n_ok": n_ok,
            "out": str(ds_out),
        }
        volume.commit()
        print(f"[esm-if-precompute] {ds_id}: wrote {n_ok}/{len(unique)} to {ds_out.name}")
    return summary


@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600)
def run_structure_baseline_from_cache(
    data: str = "processed/all_dms.parquet",
    tasks_csv: str = "outputs/editguard/edit_tasks_test.csv",
    scores_dir: str = "outputs/editguard/esm_if_scores",
    out: str = "outputs/editguard/esm_if_metrics.csv",
    selections_out: str = "outputs/editguard/esm_if_selections.parquet",
    k: int = 50,
    seeds: int = 5,
    max_tasks: int = 0,
    max_candidates: int = 10000,
    splits_csv: str = "outputs/editguard/edit_splits.csv",
    require_split: Optional[str] = "test",
    score_col: str = "esm_if_score",
) -> dict:
    """Build the structure-conditioned leaderboard rows from precomputed
    ESM-IF scores. Pure CPU lookup — no GPU needed once
    ``precompute_esm_if_scores`` has run."""
    import pandas as pd

    from phaseagent.edit_eval import evaluate_edit_selection
    from phaseagent.edit_splits import assert_tasks_in_split
    from phaseagent.editing_tasks import candidate_pool_for_task, task_from_row

    df = pd.read_parquet(Path(VOLUME_PATH) / data)
    tasks = pd.read_csv(Path(VOLUME_PATH) / tasks_csv)
    if require_split is not None:
        splits_path = Path(VOLUME_PATH) / splits_csv
        if not splits_path.exists():
            raise FileNotFoundError(
                f"require_split={require_split!r} but {splits_path} missing"
            )
        assert_tasks_in_split(tasks, pd.read_csv(splits_path), require_split)
    if max_tasks > 0:
        tasks = tasks.head(max_tasks)

    scores_root = Path(VOLUME_PATH) / scores_dir
    scores_cache: dict[str, pd.DataFrame] = {}

    metric_rows, selection_rows = [], []
    for task_idx, row in tasks.iterrows():
        task = task_from_row(row)
        if task.dataset_id not in scores_cache:
            sp = scores_root / f"{task.dataset_id}.parquet"
            if not sp.exists():
                print(f"[esm-if-cache] no scores for {task.dataset_id}; skipping")
                scores_cache[task.dataset_id] = None
                continue
            scores_cache[task.dataset_id] = pd.read_parquet(sp)
        scores_df = scores_cache[task.dataset_id]
        if scores_df is None:
            continue
        pool = candidate_pool_for_task(df, task)
        if len(pool) == 0:
            continue
        merged = pool.merge(
            scores_df[["mutation_notation", "esm_if_score", "esm_if_score_full"]],
            on="mutation_notation",
            how="left",
        )
        for seed in range(seeds):
            if max_candidates > 0 and len(merged) > max_candidates:
                pool_seed = int(task_idx) * 1000 + seed
                pool_run = merged.sample(max_candidates, random_state=pool_seed).reset_index(drop=True)
            else:
                pool_run = merged.reset_index(drop=True)
            scored = pool_run.dropna(subset=[score_col])
            if len(scored) == 0:
                continue
            top = scored.nlargest(min(k, len(scored)), score_col).copy()
            top["method"] = "esm_if_rerank"
            metrics = evaluate_edit_selection(top, task)
            metric_rows.append(
                {
                    "task_idx": int(task_idx),
                    "dataset_id": task.dataset_id,
                    "objective": task.objective,
                    "edit_budget": task.edit_budget,
                    "seed": seed,
                    "method": "esm_if_rerank",
                    "score_col": score_col,
                    **metrics,
                }
            )
            tmp = top.copy()
            tmp["task_idx"] = int(task_idx)
            tmp["seed"] = seed
            selection_rows.append(tmp)

    out_path = Path(VOLUME_PATH) / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(out_path, index=False)
    if selection_rows:
        pd.concat(selection_rows, ignore_index=True).to_parquet(
            Path(VOLUME_PATH) / selections_out, index=False
        )
    volume.commit()
    return {"rows": int(len(metric_rows)), "out": str(out_path)}


@app.function(image=gpu_image, volumes={VOLUME_PATH: volume}, gpu="A10G", timeout=3600 * 4)
def run_structure_baseline(
    data: str = "processed/all_dms.parquet",
    tasks_csv: str = "outputs/editguard/edit_tasks_test.csv",
    structures_dir: str = "structures",
    out: str = "outputs/editguard/esm_if_metrics.csv",
    selections_out: str = "outputs/editguard/esm_if_selections.parquet",
    k: int = 50,
    seeds: int = 5,
    max_tasks: int = 0,
    max_candidates: int = 10000,
    splits_csv: str = "outputs/editguard/edit_splits.csv",
    require_split: Optional[str] = "test",
    score_col: str = "esm_if_score",
) -> dict:
    """ESM-IF1 inverse-folding rerank baseline on the measured DMS pool.

    For each test task, loads the AlphaFold WT backbone (cached in
    ``structures/{dataset_id}.pdb``), scores every pool variant by mean
    per-residue log-likelihood under ESM-IF1, and selects the top-k by
    ``score_col`` (default ``esm_if_score`` = mean over mutated positions
    only). Outputs ``esm_if_rerank`` as the method label.

    Always evaluates on the test split unless ``require_split`` is overridden.
    """
    import numpy as np
    import pandas as pd

    from phaseagent.edit_eval import evaluate_edit_selection
    from phaseagent.edit_splits import assert_tasks_in_split
    from phaseagent.editing_tasks import candidate_pool_for_task, task_from_row
    from phaseagent.mutations import parse_mutation_notation
    from phaseagent.structure_baselines import (
        _load_esm_if,
        load_backbone_coords,
        per_residue_log_probs,
        score_variant_per_position,
    )

    df = pd.read_parquet(Path(VOLUME_PATH) / data)
    tasks = pd.read_csv(Path(VOLUME_PATH) / tasks_csv)
    if require_split is not None:
        splits_path = Path(VOLUME_PATH) / splits_csv
        if not splits_path.exists():
            raise FileNotFoundError(
                f"require_split={require_split!r} but {splits_path} missing"
            )
        assert_tasks_in_split(tasks, pd.read_csv(splits_path), require_split)
    if max_tasks > 0:
        tasks = tasks.head(max_tasks)

    print("[esm-if] loading esm_if1_gvp4_t16_142M_UR50 once for all tasks")
    model, alphabet = _load_esm_if()

    structures_path = Path(VOLUME_PATH) / structures_dir
    coords_cache: dict[str, tuple] = {}
    seq_score_cache: dict[tuple[str, str], np.ndarray] = {}

    metric_rows, selection_rows = [], []
    for task_idx, row in tasks.iterrows():
        task = task_from_row(row)
        ds_id = task.dataset_id
        if ds_id not in coords_cache:
            pdb_path = structures_path / f"{ds_id}.pdb"
            if not pdb_path.exists():
                print(f"[esm-if] no PDB for {ds_id} at {pdb_path}; skipping")
                coords_cache[ds_id] = None
                continue
            coords_cache[ds_id] = load_backbone_coords(str(pdb_path), chain="A")
        if coords_cache[ds_id] is None:
            continue
        coords, native_seq = coords_cache[ds_id]
        pool = candidate_pool_for_task(df, task)
        if len(pool) == 0:
            continue
        for seed in range(seeds):
            if max_candidates > 0 and len(pool) > max_candidates:
                pool_seed = int(task_idx) * 1000 + seed
                pool_run = pool.sample(max_candidates, random_state=pool_seed).reset_index(drop=True)
            else:
                pool_run = pool.reset_index(drop=True)
            pos_scores: list[float] = []
            full_scores: list[float] = []
            for _, prow in pool_run.iterrows():
                seq = str(prow.get("mutated_sequence", ""))
                if not seq or seq.lower() in {"nan", "none"} or len(seq) != len(native_seq):
                    pos_scores.append(float("nan"))
                    full_scores.append(float("nan"))
                    continue
                key = (ds_id, seq)
                if key not in seq_score_cache:
                    seq_score_cache[key] = per_residue_log_probs(model, alphabet, coords, seq)
                per_pos = seq_score_cache[key]
                toks = parse_mutation_notation(str(prow.get("mutation_notation", "")))
                positions = []
                for tok in toks:
                    try:
                        positions.append(int(tok[1:-1]))
                    except ValueError:
                        continue
                s_pos, s_full = score_variant_per_position(None, per_pos, positions)
                pos_scores.append(s_pos)
                full_scores.append(s_full)
            scored = pool_run.assign(
                esm_if_score=pos_scores,
                esm_if_score_full=full_scores,
            ).dropna(subset=[score_col])
            if len(scored) == 0:
                continue
            top = scored.nlargest(min(k, len(scored)), score_col).copy()
            top["method"] = "esm_if_rerank"
            metrics = evaluate_edit_selection(top, task)
            metric_rows.append(
                {
                    "task_idx": int(task_idx),
                    "dataset_id": ds_id,
                    "objective": task.objective,
                    "edit_budget": task.edit_budget,
                    "seed": seed,
                    "method": "esm_if_rerank",
                    "score_col": score_col,
                    **metrics,
                }
            )
            tmp = top.copy()
            tmp["task_idx"] = int(task_idx)
            tmp["seed"] = seed
            selection_rows.append(tmp)

    out_path = Path(VOLUME_PATH) / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(out_path, index=False)
    if selection_rows:
        pd.concat(selection_rows, ignore_index=True).to_parquet(
            Path(VOLUME_PATH) / selections_out, index=False
        )
    volume.commit()
    return {
        "rows": int(len(metric_rows)),
        "out": str(out_path),
        "n_unique_seqs_scored": len(seq_score_cache),
    }


@app.function(image=gpu_image, volumes={VOLUME_PATH: volume}, gpu="A10G", timeout=3600 * 4)
def run_editguard_dplm(
    data: str = "processed/all_dms.parquet",
    tasks_csv: str = "outputs/editguard/edit_tasks_test.csv",
    prior_path: str = "outputs/editguard/editguard_prior.pkl",
    out: str = "outputs/editguard/editguard_dplm_metrics.csv",
    selections_out: str = "outputs/editguard/editguard_dplm_selections.parquet",
    proposals_out: str = "outputs/editguard/editguard_dplm_proposals.parquet",
    k: int = 50,
    seeds: int = 5,
    n_mask_patterns: int = 10,
    samples_per_pattern: int = 20,
    beta: float = 1.0,
    temperature: float = 1.0,
    max_tasks: int = 0,
    splits_csv: str = "outputs/editguard/edit_splits.csv",
    require_split: Optional[str] = "test",
    model_name: str = "airkingbd/dplm_650m",
    batch_size: int = 8,
) -> dict:
    """DMS-guided DPLM sampler — the headline EditGuard generative method.

    For each test task: sample ``n_mask_patterns`` random K-position masks
    (respecting protected positions), run DPLM forward passes, decode
    ``samples_per_pattern`` variants per pattern by stochastic categorical
    sampling, then rerank with the DMS function prior at weight ``beta``.
    Outputs ``method = "editguard_diffusion_dplm"``.

    Generated proposals (deduplicated, with DPLM logprobs) are also saved
    so ``run_editguard_dplm_ablation`` can reuse them at multiple ``beta``
    values without re-running DPLM.
    """
    import pandas as pd

    from phaseagent.dplm_backbone import DPLMConfig, _load_dplm
    from phaseagent.edit_eval import evaluate_edit_selection, evaluate_generated_selection
    from phaseagent.edit_generators import infer_wildtype_sequence
    from phaseagent.edit_splits import assert_tasks_in_split
    from phaseagent.editing_tasks import task_from_row
    from phaseagent.editguard_dplm import (
        join_to_dms_labels,
        propose_with_dplm,
        rerank_with_classifier_guidance,
    )
    from phaseagent.editguard_prior import DMSFunctionPrior

    df = pd.read_parquet(Path(VOLUME_PATH) / data)
    tasks = pd.read_csv(Path(VOLUME_PATH) / tasks_csv)
    if require_split is not None:
        splits_path = Path(VOLUME_PATH) / splits_csv
        if not splits_path.exists():
            raise FileNotFoundError(
                f"require_split={require_split!r} but {splits_path} missing"
            )
        assert_tasks_in_split(tasks, pd.read_csv(splits_path), require_split)
    if max_tasks > 0:
        tasks = tasks.head(max_tasks)

    print(f"[dplm] loading {model_name}")
    model, tokenizer = _load_dplm(model_name)
    prior = DMSFunctionPrior.load(Path(VOLUME_PATH) / prior_path)

    wt_cache: dict[str, str] = {}
    metric_rows, selection_rows, proposal_rows = [], [], []

    for task_idx, row in tasks.iterrows():
        task = task_from_row(row)
        wt = wt_cache.get(task.dataset_id)
        if wt is None:
            wt = infer_wildtype_sequence(df, task.dataset_id)
            if wt is None:
                print(f"[dplm] {task.dataset_id}: cannot infer WT, skipping")
                continue
            wt_cache[task.dataset_id] = wt
            print(f"[dplm] {task.dataset_id}: WT length {len(wt)}")
        for seed in range(seeds):
            cfg = DPLMConfig(
                n_mask_patterns=n_mask_patterns,
                samples_per_pattern=samples_per_pattern,
                temperature=temperature,
                seed=int(task_idx) * 1000 + seed,
                batch_size=batch_size,
            )
            proposals = propose_with_dplm(wt, task, model, tokenizer, cfg)
            if len(proposals) == 0:
                continue
            # Persist raw proposals so the ablation can reuse them.
            tmp = proposals.copy()
            tmp["task_idx"] = int(task_idx)
            tmp["seed"] = seed
            tmp["objective"] = task.objective
            tmp["edit_budget"] = task.edit_budget
            proposal_rows.append(tmp)

            top = rerank_with_classifier_guidance(proposals, prior, beta=beta, k=k)
            top["task_idx"] = int(task_idx)
            top["seed"] = seed
            labeled = join_to_dms_labels(top, df)
            metric_rows.append(
                {
                    "task_idx": int(task_idx),
                    "dataset_id": task.dataset_id,
                    "objective": task.objective,
                    "edit_budget": task.edit_budget,
                    "seed": seed,
                    "method": "editguard_diffusion_dplm",
                    "beta": float(beta),
                    "n_proposals": int(len(proposals)),
                    **evaluate_generated_selection(labeled, task),
                }
            )
            selection_rows.append(labeled)

    out_path = Path(VOLUME_PATH) / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(out_path, index=False)
    if selection_rows:
        pd.concat(selection_rows, ignore_index=True).to_parquet(
            Path(VOLUME_PATH) / selections_out, index=False
        )
    if proposal_rows:
        pd.concat(proposal_rows, ignore_index=True).to_parquet(
            Path(VOLUME_PATH) / proposals_out, index=False
        )
    volume.commit()
    return {
        "rows": int(len(metric_rows)),
        "out": str(out_path),
        "n_proposals_total": int(sum(len(p) for p in proposal_rows)),
    }


@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600)
def run_editguard_dplm_ablation(
    data: str = "processed/all_dms.parquet",
    proposals_in: str = "outputs/editguard/editguard_dplm_proposals.parquet",
    tasks_csv: str = "outputs/editguard/edit_tasks_test.csv",
    prior_path: str = "outputs/editguard/editguard_prior.pkl",
    out: str = "outputs/editguard/editguard_dplm_ablation_metrics.csv",
    k: int = 50,
    beta_grid: str = "0,0.5,1.0,2.0",
    splits_csv: str = "outputs/editguard/edit_splits.csv",
    require_split: Optional[str] = "test",
) -> dict:
    """Reuse cached DPLM proposals from ``run_editguard_dplm`` and rerank at
    every value of ``beta_grid``. Defends C4 (classifier guidance recovers
    the gap between DMS-naive and DMS-conditioned generation)."""
    import pandas as pd

    from phaseagent.edit_eval import evaluate_generated_selection
    from phaseagent.edit_splits import assert_tasks_in_split
    from phaseagent.editing_tasks import task_from_row
    from phaseagent.editguard_dplm import join_to_dms_labels, rerank_with_classifier_guidance
    from phaseagent.editguard_prior import DMSFunctionPrior

    df = pd.read_parquet(Path(VOLUME_PATH) / data)
    tasks = pd.read_csv(Path(VOLUME_PATH) / tasks_csv)
    if require_split is not None:
        splits_path = Path(VOLUME_PATH) / splits_csv
        if not splits_path.exists():
            raise FileNotFoundError(
                f"require_split={require_split!r} but {splits_path} missing"
            )
        assert_tasks_in_split(tasks, pd.read_csv(splits_path), require_split)

    proposals = pd.read_parquet(Path(VOLUME_PATH) / proposals_in)
    prior = DMSFunctionPrior.load(Path(VOLUME_PATH) / prior_path)
    betas = [float(b) for b in beta_grid.split(",") if b.strip()]

    metric_rows = []
    for (task_idx, seed), group in proposals.groupby(["task_idx", "seed"]):
        task_row = tasks[tasks.index == task_idx]
        if len(task_row) == 0:
            continue
        task = task_from_row(task_row.iloc[0])
        for beta in betas:
            top = rerank_with_classifier_guidance(group, prior, beta=beta, k=k)
            top["task_idx"] = int(task_idx)
            top["seed"] = int(seed)
            labeled = join_to_dms_labels(top, df)
            metric_rows.append(
                {
                    "task_idx": int(task_idx),
                    "dataset_id": task.dataset_id,
                    "objective": task.objective,
                    "edit_budget": task.edit_budget,
                    "seed": int(seed),
                    "method": "editguard_diffusion_dplm",
                    "beta": float(beta),
                    "n_proposals": int(len(group)),
                    **evaluate_generated_selection(labeled, task),
                }
            )

    out_path = Path(VOLUME_PATH) / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(out_path, index=False)
    volume.commit()
    return {"rows": int(len(metric_rows)), "out": str(out_path), "betas": betas}


@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600)
def run_vep_baselines(
    data: str = "processed/all_dms.parquet",
    tasks_csv: str = "outputs/editguard/edit_tasks_test.csv",
    zero_shot_dir: str = "raw/proteingym_v1_3_zero_shot",
    out: str = "outputs/editguard/vep_metrics.csv",
    selections_out: str = "outputs/editguard/vep_selections.parquet",
    k: int = 50,
    seeds: int = 5,
    max_tasks: int = 0,
    max_candidates: int = 10000,
    splits_csv: str = "outputs/editguard/edit_splits.csv",
    require_split: Optional[str] = "test",
    methods: str = "tranception_l_rerank:Tranception_L,eve_ensemble_rerank:EVE_ensemble",
) -> dict:
    """Pre-computed VEP rerank baselines using ProteinGym v1.3 zero-shot scores.

    Joins the candidate pool to the per-assay zero-shot CSV by mutation
    notation and ranks by each VEP model's score. ``methods`` is a comma list
    of ``label:column`` pairs; defaults to Tranception_L (with retrieval) and
    EVE_ensemble — the two headline VEP picks per the implementation plan.

    Always evaluates on the test split unless ``require_split`` is overridden.
    Coverage (fraction of pool variants with a VEP score) is reported per
    task in case the zero-shot CSV is missing rows.
    """
    import pandas as pd

    from phaseagent.edit_eval import evaluate_edit_selection
    from phaseagent.edit_splits import assert_tasks_in_split
    from phaseagent.editing_tasks import candidate_pool_for_task, task_from_row
    from phaseagent.vep_baselines import (
        load_zero_shot_for_dataset,
        rerank_by_vep,
        vep_coverage,
    )

    df = pd.read_parquet(Path(VOLUME_PATH) / data)
    tasks = pd.read_csv(Path(VOLUME_PATH) / tasks_csv)
    if require_split is not None:
        splits_path = Path(VOLUME_PATH) / splits_csv
        if not splits_path.exists():
            raise FileNotFoundError(
                f"require_split={require_split!r} but {splits_path} missing"
            )
        assert_tasks_in_split(tasks, pd.read_csv(splits_path), require_split)
    if max_tasks > 0:
        tasks = tasks.head(max_tasks)

    pairs = []
    for entry in methods.split(","):
        entry = entry.strip()
        if not entry:
            continue
        label, _, column = entry.partition(":")
        if not column:
            raise ValueError(f"Bad method spec {entry!r}; expected label:column")
        pairs.append((label.strip(), column.strip()))
    if not pairs:
        raise ValueError("methods must contain at least one label:column pair")

    zs_dir = Path(VOLUME_PATH) / zero_shot_dir
    zs_cache: dict[str, pd.DataFrame | None] = {}

    metric_rows, selection_rows = [], []
    for task_idx, row in tasks.iterrows():
        task = task_from_row(row)
        if task.dataset_id not in zs_cache:
            zs_cache[task.dataset_id] = load_zero_shot_for_dataset(zs_dir, task.dataset_id)
        zs_df = zs_cache[task.dataset_id]
        if zs_df is None:
            print(f"[vep] no zero-shot CSV for {task.dataset_id}; skipping")
            continue
        pool = candidate_pool_for_task(df, task)
        if len(pool) == 0:
            continue
        for seed in range(seeds):
            if max_candidates > 0 and len(pool) > max_candidates:
                pool_seed = int(task_idx) * 1000 + seed
                pool_run = pool.sample(max_candidates, random_state=pool_seed).reset_index(drop=True)
            else:
                pool_run = pool.reset_index(drop=True)
            for label, column in pairs:
                if column not in zs_df.columns:
                    print(f"[vep] {column} missing from {task.dataset_id}; skipping")
                    continue
                cov = vep_coverage(pool_run, zs_df, column)
                sel = rerank_by_vep(pool_run, zs_df, label, column, k=k, higher_is_better=True)
                if len(sel) == 0:
                    continue
                metrics = evaluate_edit_selection(sel, task)
                metric_rows.append(
                    {
                        "task_idx": int(task_idx),
                        "dataset_id": task.dataset_id,
                        "objective": task.objective,
                        "edit_budget": task.edit_budget,
                        "seed": seed,
                        "method": label,
                        "vep_column": column,
                        "vep_coverage": cov,
                        **metrics,
                    }
                )
                tmp = sel.copy()
                tmp["task_idx"] = int(task_idx)
                tmp["seed"] = seed
                selection_rows.append(tmp)

    out_path = Path(VOLUME_PATH) / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(out_path, index=False)
    if selection_rows:
        pd.concat(selection_rows, ignore_index=True).to_parquet(
            Path(VOLUME_PATH) / selections_out, index=False
        )
    volume.commit()
    return {"rows": int(len(metric_rows)), "out": str(out_path), "methods": [p[0] for p in pairs]}


@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600 * 2)
def run_editguard_ablations(
    data: str = "processed/all_dms.parquet",
    tasks_csv: str = "outputs/editguard/edit_tasks.csv",
    prior_path: str = "outputs/editguard/editguard_prior.pkl",
    out: str = "outputs/editguard/editguard_ablation_metrics.csv",
    k: int = 50,
    seeds: int = 5,
    max_tasks: int = 50,
    max_candidates: int = 10000,
) -> dict:
    import numpy as np
    import pandas as pd

    from phaseagent.edit_eval import evaluate_edit_selection
    from phaseagent.editing_tasks import candidate_pool_for_task, task_from_row
    from phaseagent.editguard_prior import DMSFunctionPrior
    from phaseagent.editguard_sampling import guided_scores

    configs = {
        "default": {"alpha": 1.0, "beta": 0.5, "gamma": 2.0, "kappa": 0.0},
        "no_dms_guidance": {"alpha": 0.0, "beta": 0.5, "gamma": 2.0, "kappa": 0.0},
        "no_objective": {"alpha": 1.0, "beta": 0.0, "gamma": 2.0, "kappa": 0.0},
        "uncertainty_penalty": {"alpha": 1.0, "beta": 0.5, "gamma": 2.0, "kappa": 0.5},
        "strong_dms_guidance": {"alpha": 2.0, "beta": 0.5, "gamma": 2.0, "kappa": 0.0},
    }
    df = pd.read_parquet(Path(VOLUME_PATH) / data)
    tasks = pd.read_csv(Path(VOLUME_PATH) / tasks_csv)
    if max_tasks > 0:
        tasks = tasks.head(max_tasks)
    prior = DMSFunctionPrior.load(Path(VOLUME_PATH) / prior_path)
    metric_rows = []
    for task_idx, row in tasks.iterrows():
        task = task_from_row(row)
        pool = candidate_pool_for_task(df, task)
        if len(pool) == 0:
            continue
        for seed in range(seeds):
            if max_candidates > 0 and len(pool) > max_candidates:
                pool_seed = int(task_idx) * 1000 + seed
                pool_run = pool.sample(max_candidates, random_state=pool_seed)
            else:
                pool_run = pool
            base_scored = guided_scores(pool_run, task, prior, alpha=1.0, beta=0.5, gamma=2.0, kappa=0.0)
            for name, cfg in configs.items():
                energy = (
                    -cfg["alpha"] * np.log(base_scored["prior_function_prob"].clip(1e-6, 1.0))
                    -cfg["beta"] * base_scored["objective_score"]
                    +cfg["gamma"] * (~base_scored["constraint_satisfied"]).astype(float)
                    +cfg["kappa"] * base_scored["prior_uncertainty"]
                )
                selected = base_scored.assign(editguard_energy=energy).nsmallest(min(k, len(base_scored)), "editguard_energy")
                metric_rows.append(
                    {
                        "task_idx": int(task_idx),
                        "dataset_id": task.dataset_id,
                        "objective": task.objective,
                        "edit_budget": task.edit_budget,
                        "seed": seed,
                        "method": "dms_pool_guided",
                        "ablation": name,
                        **evaluate_edit_selection(selected, task),
                    }
                )
    out_path = Path(VOLUME_PATH) / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(out_path, index=False)
    volume.commit()
    return {"rows": int(len(metric_rows)), "out": str(out_path)}


@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600 * 2)
def run_guided_generation(
    data: str = "processed/all_dms.parquet",
    tasks_csv: str = "outputs/editguard/edit_tasks.csv",
    prior_path: str = "outputs/editguard/editguard_prior.pkl",
    out: str = "outputs/editguard/guided_generation_metrics.csv",
    selections_out: str = "outputs/editguard/guided_generation_selections.parquet",
    k: int = 50,
    n_generate: int = 500,
    seeds: int = 3,
    max_tasks: int = 20,
) -> dict:
    import pandas as pd

    from phaseagent.edit_eval import evaluate_generated_selection
    from phaseagent.edit_generators import (
        ProposalConfig,
        generate_many_then_rerank,
        guided_local_generation,
        infer_wildtype_sequence,
        join_generated_to_dms,
        observed_single_mutation_tokens,
        sample_random_edit_candidates,
    )
    from phaseagent.editing_tasks import task_from_row
    from phaseagent.editguard_prior import DMSFunctionPrior

    df = pd.read_parquet(Path(VOLUME_PATH) / data)
    tasks = pd.read_csv(Path(VOLUME_PATH) / tasks_csv)
    if max_tasks > 0:
        tasks = tasks.head(max_tasks)
    prior = DMSFunctionPrior.load(Path(VOLUME_PATH) / prior_path)
    metrics, selections = [], []
    for task_idx, row in tasks.iterrows():
        task = task_from_row(row)
        wt = infer_wildtype_sequence(df, task.dataset_id)
        if wt is None:
            continue
        allowed_tokens = observed_single_mutation_tokens(df, task)
        for seed in range(seeds):
            methods = {}
            random_props = sample_random_edit_candidates(
                task.dataset_id,
                wt,
                task,
                ProposalConfig(n_candidates=k, seed=seed),
                source="random_direct_generation",
                allowed_tokens=allowed_tokens,
            )
            random_props["method"] = "random_direct_generation"
            methods["random_direct_generation"] = random_props
            methods[f"generate_{n_generate}_then_rerank"] = generate_many_then_rerank(
                task.dataset_id,
                wt,
                task,
                prior,
                n_generate=n_generate,
                k=k,
                seed=seed,
                allowed_tokens=allowed_tokens,
            )
            guided = guided_local_generation(
                task.dataset_id,
                wt,
                task,
                prior,
                ProposalConfig(n_candidates=k, seed=seed),
                allowed_tokens=allowed_tokens,
            )
            guided["method"] = "guided_local_generation"
            methods["guided_local_generation"] = guided
            for method, generated in methods.items():
                labeled = join_generated_to_dms(generated, df)
                labeled["task_idx"] = int(task_idx)
                labeled["seed"] = seed
                labeled["method"] = method
                selections.append(labeled)
                metrics.append(
                    {
                        "task_idx": int(task_idx),
                        "dataset_id": task.dataset_id,
                        "objective": task.objective,
                        "edit_budget": task.edit_budget,
                        "seed": seed,
                        "method": method,
                        "n_generated": int(n_generate if "then_rerank" in method else k),
                        **evaluate_generated_selection(labeled, task),
                    }
                )
    out_path = Path(VOLUME_PATH) / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(out_path, index=False)
    if selections:
        pd.concat(selections, ignore_index=True).to_parquet(Path(VOLUME_PATH) / selections_out, index=False)
    volume.commit()
    return {"rows": int(len(metrics)), "out": str(out_path)}


@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=3600 * 4)
def run_guided_generation_sweep(
    data: str = "processed/all_dms.parquet",
    tasks_csv: str = "outputs/editguard/edit_tasks.csv",
    prior_path: str = "outputs/editguard/editguard_prior.pkl",
    out: str = "outputs/editguard/guided_generation_sweep_metrics.csv",
    candidate_budgets: str = "20,50,100,200,500",
    seeds: int = 5,
    max_tasks: int = 0,
    coverage_mode: str = "dms_evaluable",
) -> dict:
    import pandas as pd

    from phaseagent.edit_eval import evaluate_generated_selection
    from phaseagent.edit_generators import (
        ProposalConfig,
        generate_many_then_rerank,
        guided_local_generation,
        infer_wildtype_sequence,
        join_generated_to_dms,
        observed_single_mutation_tokens,
        plm_masked_proposal_generation,
        sample_random_edit_candidates,
    )
    from phaseagent.editing_tasks import task_from_row
    from phaseagent.editguard_prior import DMSFunctionPrior

    budgets = [int(x) for x in candidate_budgets.split(",") if x.strip()]
    df = pd.read_parquet(Path(VOLUME_PATH) / data)
    tasks = pd.read_csv(Path(VOLUME_PATH) / tasks_csv)
    if max_tasks > 0:
        tasks = tasks.head(max_tasks)
    prior = DMSFunctionPrior.load(Path(VOLUME_PATH) / prior_path)
    rows = []
    for task_idx, row in tasks.iterrows():
        task = task_from_row(row)
        wt = infer_wildtype_sequence(df, task.dataset_id)
        if wt is None:
            continue
        allowed = observed_single_mutation_tokens(df, task) if coverage_mode == "dms_evaluable" else None
        for budget in budgets:
            for seed in range(seeds):
                methods = {
                    "random_direct_generation": sample_random_edit_candidates(
                        task.dataset_id,
                        wt,
                        task,
                        ProposalConfig(n_candidates=budget, seed=seed, coverage_mode=coverage_mode),
                        source="random_direct_generation",
                        allowed_tokens=allowed,
                    ),
                    "guided_local_generation": guided_local_generation(
                        task.dataset_id,
                        wt,
                        task,
                        prior,
                        ProposalConfig(n_candidates=budget, seed=seed, coverage_mode=coverage_mode),
                        allowed_tokens=allowed,
                    ),
                    "esm2_masked_marginal_proxy": plm_masked_proposal_generation(
                        task.dataset_id,
                        wt,
                        task,
                        n_candidates=budget,
                        seed=seed,
                        allowed_tokens=allowed,
                    ),
                    f"generate_{budget * 10}_then_rerank": generate_many_then_rerank(
                        task.dataset_id,
                        wt,
                        task,
                        prior,
                        n_generate=budget * 10,
                        k=budget,
                        seed=seed,
                        allowed_tokens=allowed,
                    ),
                }
                for method, generated in methods.items():
                    labeled = join_generated_to_dms(generated, df)
                    labeled["method"] = method
                    rows.append(
                        {
                            "task_idx": int(task_idx),
                            "dataset_id": task.dataset_id,
                            "objective": task.objective,
                            "edit_budget": task.edit_budget,
                            "seed": seed,
                            "method": method,
                            "candidate_budget": budget,
                            "n_generated": int(budget * 10 if "then_rerank" in method else budget),
                            "coverage_mode": coverage_mode,
                            **evaluate_generated_selection(labeled, task),
                        }
                    )
    out_path = Path(VOLUME_PATH) / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    volume.commit()
    return {"rows": int(len(rows)), "out": str(out_path)}


@app.function(image=gpu_image, volumes={VOLUME_PATH: volume}, gpu="A10G", timeout=3600 * 4)
def run_esm_guided_generation_sweep(
    data: str = "processed/all_dms.parquet",
    tasks_csv: str = "outputs/editguard/edit_tasks.csv",
    prior_path: str = "outputs/editguard/editguard_prior.pkl",
    out: str = "outputs/editguard/esm_guided_generation_sweep_metrics.csv",
    selections_out: str = "outputs/editguard/esm_guided_generation_sweep_selections.parquet",
    candidate_budgets: str = "20,50",
    seeds: int = 1,
    max_tasks: int = 3,
    coverage_mode: str = "dms_evaluable",
    model_name: str = "esm2_t6_8M_UR50D",
    esm_batch_size: int = 4,
) -> dict:
    import pandas as pd

    from phaseagent.edit_eval import evaluate_generated_selection
    from phaseagent.edit_generators import (
        ProposalConfig,
        generate_many_then_rerank,
        guided_local_generation,
        infer_wildtype_sequence,
        join_generated_to_dms,
        observed_single_mutation_tokens,
        rerank_by_column,
        sample_random_edit_candidates,
    )
    from phaseagent.editing_tasks import task_from_row
    from phaseagent.editguard_prior import DMSFunctionPrior
    from phaseagent.plm import score_generated_esm_masked

    budgets = [int(x) for x in candidate_budgets.split(",") if x.strip()]
    df = pd.read_parquet(Path(VOLUME_PATH) / data)
    tasks = pd.read_csv(Path(VOLUME_PATH) / tasks_csv)
    if max_tasks > 0:
        tasks = tasks.head(max_tasks)
    prior = DMSFunctionPrior.load(Path(VOLUME_PATH) / prior_path)
    metric_rows, selection_rows = [], []
    for task_idx, row in tasks.iterrows():
        task = task_from_row(row)
        wt = infer_wildtype_sequence(df, task.dataset_id)
        if wt is None:
            continue
        allowed = observed_single_mutation_tokens(df, task) if coverage_mode == "dms_evaluable" else None
        for budget in budgets:
            for seed in range(seeds):
                proposal_pool = sample_random_edit_candidates(
                    task.dataset_id,
                    wt,
                    task,
                    ProposalConfig(n_candidates=budget * 10, seed=seed, coverage_mode=coverage_mode),
                    source="esm_masked_candidate_pool",
                    allowed_tokens=allowed,
                )
                proposal_pool = score_generated_esm_masked(
                    proposal_pool,
                    model_name=model_name,
                    batch_size=esm_batch_size,
                )
                esm_direct = rerank_by_column(proposal_pool, "esm_masked_score", budget, "esm2_masked_marginal")
                esm_dms = proposal_pool.copy()
                if len(esm_dms):
                    esm_dms["prior_function_prob"] = prior.predict_proba(esm_dms)
                    esm_dms = esm_dms.nlargest(min(budget, len(esm_dms)), "prior_function_prob")
                    esm_dms["method"] = "esm2_masked_marginal_dms_rerank"
                methods = {
                    "esm2_masked_marginal": esm_direct,
                    "esm2_masked_marginal_dms_rerank": esm_dms,
                    "guided_local_generation": guided_local_generation(
                        task.dataset_id,
                        wt,
                        task,
                        prior,
                        ProposalConfig(n_candidates=budget, seed=seed, coverage_mode=coverage_mode),
                        allowed_tokens=allowed,
                    ),
                    f"generate_{budget * 10}_then_rerank": generate_many_then_rerank(
                        task.dataset_id,
                        wt,
                        task,
                        prior,
                        n_generate=budget * 10,
                        k=budget,
                        seed=seed,
                        allowed_tokens=allowed,
                    ),
                }
                for method, generated in methods.items():
                    labeled = join_generated_to_dms(generated, df)
                    labeled["method"] = method
                    labeled["task_idx"] = int(task_idx)
                    labeled["seed"] = seed
                    labeled["candidate_budget"] = budget
                    selection_rows.append(labeled)
                    metric_rows.append(
                        {
                            "task_idx": int(task_idx),
                            "dataset_id": task.dataset_id,
                            "objective": task.objective,
                            "edit_budget": task.edit_budget,
                            "seed": seed,
                            "method": method,
                            "candidate_budget": budget,
                            "n_generated": int(budget * 10 if "then_rerank" in method or method.startswith("esm2") else budget),
                            "coverage_mode": coverage_mode,
                            **evaluate_generated_selection(labeled, task),
                        }
                    )
    out_path = Path(VOLUME_PATH) / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(out_path, index=False)
    if selection_rows:
        pd.concat(selection_rows, ignore_index=True).to_parquet(Path(VOLUME_PATH) / selections_out, index=False)
    volume.commit()
    return {"rows": int(len(metric_rows)), "out": str(out_path)}


@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=1800)
def run_prompted_edit_demos(
    data: str = "processed/all_dms.parquet",
    prior_path: str = "outputs/editguard/editguard_prior.pkl",
    out: str = "outputs/editguard/prompted_edit_demos.csv",
    dataset_ids: str = "",
) -> dict:
    import pandas as pd

    from phaseagent.edit_prompts import generate_prompted_edits, parse_edit_prompt
    from phaseagent.editguard_prior import DMSFunctionPrior

    df = pd.read_parquet(Path(VOLUME_PATH) / data)
    prior = DMSFunctionPrior.load(Path(VOLUME_PATH) / prior_path)
    candidates = [x.strip() for x in dataset_ids.split(",") if x.strip()]
    if not candidates:
        preferred = ("BRCA", "PTEN", "P53", "TP53", "GFP", "AAV")
        candidates = [ds for ds in df["dataset_id"].dropna().astype(str).unique() if any(p in ds.upper() for p in preferred)]
        candidates = candidates[:5] if candidates else list(df["dataset_id"].dropna().astype(str).unique()[:5])
    prompts = [
        "Generate a high-novelty variant while preserving function.",
        "Avoid DMS-fragile positions while preserving function.",
        "Remove cysteine liabilities while protecting known functional residues.",
        "Create a 2 mutation variant using single-mutant DMS guidance.",
    ]
    rows = []
    for ds_id in candidates:
        for prompt_idx, prompt in enumerate(prompts):
            req = parse_edit_prompt(ds_id, prompt, edit_budget=2 if "2 mutation" in prompt else 3, n_candidates=10)
            generated = generate_prompted_edits(df, prior, req, seed=prompt_idx)
            if len(generated) == 0:
                continue
            generated["prompt_idx"] = prompt_idx
            rows.append(generated.head(5))
    out_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out_path = Path(VOLUME_PATH) / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    volume.commit()
    return {"rows": int(len(out_df)), "out": str(out_path)}


@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=600)
def export_baseline_registry(out: str = "outputs/editguard/baseline_registry.csv") -> dict:
    from phaseagent.edit_sota import baseline_registry_frame

    out_path = Path(VOLUME_PATH) / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = baseline_registry_frame()
    df.to_csv(out_path, index=False)
    volume.commit()
    return {"rows": int(len(df)), "out": str(out_path)}


@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=600)
def run_structure_sanity_report(
    selections: str = "outputs/editguard/prompted_edit_demos.csv",
    out: str = "outputs/editguard/structure_sanity_report.csv",
    mutation_out: str = "outputs/editguard/structure_mutation_positions.csv",
) -> dict:
    import pandas as pd

    from phaseagent.edit_structure_sanity import mutation_position_table, structure_sanity_stub

    in_path = Path(VOLUME_PATH) / selections
    if not in_path.exists() or in_path.stat().st_size == 0:
        report = pd.DataFrame()
        muts = pd.DataFrame()
    else:
        generated = pd.read_csv(in_path)
        report = structure_sanity_stub(generated)
        muts = mutation_position_table(generated)
    out_path = Path(VOLUME_PATH) / out
    mut_path = Path(VOLUME_PATH) / mutation_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_path, index=False)
    muts.to_csv(mut_path, index=False)
    volume.commit()
    return {"rows": int(len(report)), "mutations": int(len(muts)), "out": str(out_path)}


@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=1800)
def make_editguard_figures(
    baseline_results: str = "outputs/editguard/edit_baseline_metrics.csv",
    diffusion_results: str = "outputs/editguard/dms_pool_guided_metrics.csv",
    out_dir: str = "outputs/editguard/figures",
) -> dict:
    import pandas as pd

    from phaseagent.edit_plots import (
        plot_ablation_bar,
        plot_editing_frontier,
        plot_coverage_vs_hit_rate,
        plot_fig2_replacement,
        plot_grouped_metric_bars,
        plot_method_boxplot,
        plot_sample_efficiency,
    )

    frames = []
    for rel in [baseline_results, diffusion_results]:
        path = Path(VOLUME_PATH) / rel
        if path.exists():
            frames.append(pd.read_csv(path))
    results = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out = Path(VOLUME_PATH) / out_dir
    out.mkdir(parents=True, exist_ok=True)
    if len(results):
        plot_fig2_replacement(results, out / "fig2_method_summary")
        plot_editing_frontier(results, out / "fig2_editing_frontier")
        plot_method_boxplot(results, out / "fig4_functional_hit_boxplot")
        if "edit_budget" in results.columns:
            plot_grouped_metric_bars(results, out / "fig2_grouped_hit_rate_by_budget")
    ablation_candidates = [
        Path(VOLUME_PATH) / "outputs/editguard/editguard_ablation_metrics.csv",
        Path(VOLUME_PATH) / "outputs/editguard/editguard_ablation_fast_metrics.csv",
        Path(VOLUME_PATH) / "outputs/editguard/editguard_ablation_smoke_metrics.csv",
    ]
    ablation_path = next((p for p in ablation_candidates if p.exists()), None)
    if ablation_path is not None:
        ablations = pd.read_csv(ablation_path)
        if len(ablations):
            plot_ablation_bar(ablations, out / "fig6_guidance_ablation")
    generation_candidates = [
        Path(VOLUME_PATH) / "outputs/editguard/guided_generation_metrics.csv",
        Path(VOLUME_PATH) / "outputs/editguard/guided_generation_fast_metrics.csv",
    ]
    generation_path = next((p for p in generation_candidates if p.exists() and p.stat().st_size > 0), None)
    if generation_path is not None:
        generation = pd.read_csv(generation_path)
        if len(generation):
            plot_sample_efficiency(generation, out / "fig3_guided_generation_sample_efficiency")
            plot_coverage_vs_hit_rate(generation, out / "fig8_label_coverage_vs_hit_rate")
    sweep_path = Path(VOLUME_PATH) / "outputs/editguard/guided_generation_sweep_metrics.csv"
    if sweep_path.exists() and sweep_path.stat().st_size > 0:
        sweep = pd.read_csv(sweep_path)
        if len(sweep):
            plot_sample_efficiency(sweep, out / "fig3_sample_efficiency_sweep")
            plot_coverage_vs_hit_rate(sweep, out / "fig8_sweep_coverage_vs_hit_rate")
    esm_sweep_path = Path(VOLUME_PATH) / "outputs/editguard/esm_guided_generation_sweep_metrics.csv"
    if esm_sweep_path.exists() and esm_sweep_path.stat().st_size > 0:
        esm_sweep = pd.read_csv(esm_sweep_path)
        if len(esm_sweep):
            plot_sample_efficiency(esm_sweep, out / "fig3_esm_sota_sample_efficiency")
            plot_coverage_vs_hit_rate(esm_sweep, out / "fig8_esm_sota_coverage_vs_hit_rate")
    volume.commit()
    return {"figures": 2 if len(results) else 0, "out": str(out)}


@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=600)
def summarize_editguard_outputs(rel_dir: str = "outputs/editguard") -> dict:
    import pandas as pd

    base = Path(VOLUME_PATH) / rel_dir
    summary = {}
    for name in [
        "edit_splits.csv",
        "edit_tasks.csv",
        "edit_tasks_train.csv",
        "edit_tasks_val.csv",
        "edit_tasks_test.csv",
        "edit_tasks_clinical_test.csv",
        "editguard_prior_metrics.csv",
        "editguard_prior_calibration.csv",
        "edit_baseline_metrics.csv",
        "dms_pool_guided_metrics.csv",
        "editguard_dplm_metrics.csv",
        "editguard_ablation_metrics.csv",
        "editguard_ablation_fast_metrics.csv",
        "editguard_ablation_smoke_metrics.csv",
        "guided_generation_metrics.csv",
        "guided_generation_fast_metrics.csv",
        "guided_generation_sweep_metrics.csv",
        "esm_guided_generation_sweep_metrics.csv",
        "prompted_edit_demos.csv",
        "structure_sanity_report.csv",
        "baseline_registry.csv",
    ]:
        path = base / name
        if not path.exists():
            print(f"[missing] {name}")
            continue
        if path.stat().st_size == 0:
            print(f"[empty] {name}")
            summary[name] = {"rows": 0, "cols": 0}
            continue
        df = pd.read_csv(path)
        summary[name] = {"rows": int(len(df)), "cols": int(len(df.columns))}
        print(f"[artifact] {name}: rows={len(df)} cols={len(df.columns)}")
        if name == "editguard_prior_metrics.csv":
            print(df.to_string(index=False))
        if "metrics" in name and "method" in df.columns and "functional_hit_rate" in df.columns:
            means = df.groupby("method")["functional_hit_rate"].agg(["mean", "count"]).sort_values("mean", ascending=False)
            print(means.to_string())
        if name.startswith("guided_generation") and "functional_hit_rate_labeled" in df.columns:
            cols = ["functional_hit_rate_labeled", "labeled_fraction", "n_generated"]
            means = df.groupby("method")[cols].agg(["mean", "count"])
            print(means.to_string())
        if name == "baseline_registry.csv":
            print(df.groupby(["tier", "status"]).size().to_string())
    fig_dir = base / "figures"
    figures = sorted(str(p.relative_to(base)) for p in fig_dir.rglob("*") if p.is_file()) if fig_dir.exists() else []
    summary["figures"] = figures
    print("[figures]")
    for fig in figures:
        print(fig)
    return summary


# ---------- 10. Pull outputs back to local ----------

@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=600)
def list_outputs(rel_dir: str = "outputs") -> list[str]:
    base = Path(VOLUME_PATH) / rel_dir
    return [str(p.relative_to(VOLUME_PATH)) for p in base.rglob("*") if p.is_file()]


@app.local_entrypoint()
def main(
    skip_download: bool = False,
    skip_plm: bool = False,
    n_datasets: int = 0,
):
    n_datasets_arg = n_datasets if n_datasets > 0 else None
    """End-to-end pipeline. Pass --skip-plm to leave the GPU step out."""
    if not skip_download:
        print("[1/7] download_proteingym")
        download_proteingym.remote()
    print("[2/7] build_dataset")
    build_dataset.remote(n_datasets=n_datasets_arg)
    print("[3/7] run_phase_atlas")
    run_phase_atlas.remote()
    print("[4/7] run_phaseagent")
    run_phaseagent.remote()
    print("[5/7] run_phase_aware_search")
    run_phase_aware_search.remote()
    if not skip_plm:
        print("[6/7] run_plm_scoring (GPU)")
        run_plm_scoring.remote()
    print("[7/7] make_figures")
    make_figures.remote()
    print("done. Pull outputs with `modal volume get phaseagent-data outputs ./outputs`.")
