"""Model 2 — the design proof. Can Model 1's predicted epistasis select
gain-of-function combinations the additive baseline cannot, AT MATCHED additive?

For held-out GB1 doubles we compare three rankings of predicted function:
  additive/global baseline:  ddG_global
  Model 1:                   ddG_global + predicted_specific_epistasis
  oracle:                    ddG_global + true_specific_epistasis
and report (A) the measured binding of the top-selected designs and (B) the
matched-additive control: within narrow additive bins, does Model 1's predicted
epistasis still order the measured function? (If yes, the gain is genuine
epistasis, not relearned additivity.)

The traceable DoD #3 artifact is `model2_oof_<assay>.csv` — the raw per-double
table (glob, measured, true_eps, pred_eps). `model2_<assay>_summary.json` holds
the paper's numbers, derived purely from that CSV via `summarize_model2`, so a
test can recompute and catch csv/summary drift (the headroom pattern). The
figure regenerates from the committed CSV via `figure_model2`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

GREY, RED, BLUE = "#9A9A9A", "#C0504D", "#3B6FB6"
AA = "ACDEFGHIKLMNPQRSTVWY"
FRACS = [0.05, 0.1, 0.2, 0.35, 0.5, 1.0]


def build_oof_table(feat, assay_csv, epochs=200):
    """Decompose the assay, train Model 1 held-out-double OOF, and merge into the
    per-double table (glob, measured, true_eps, pred_eps) the paper analyses."""
    # heavy deps (torch via train_model1) imported lazily so that importing this
    # module for summarize_model2 / figure_model2 (tests, the CI bootstrap) stays cheap.
    import sys
    sys.path.insert(0, "scripts")
    from sklearn.model_selection import KFold
    from train_model1 import load, train_fold
    from phaseagent.mutations import parse_mutation_notation
    from phaseagent.epistasis_decomposition import add_global_specific_layers, decompose_multimutants
    d = pd.read_csv(assay_csv).rename(columns={"mutant": "mutation_notation"})
    d["dataset_id"] = "GB1"
    d["mutation_distance"] = d["mutation_notation"].astype(str).apply(lambda s: len(parse_mutation_notation(s)))
    d["DMS_score"] = pd.to_numeric(d["DMS_score"], errors="coerce")
    dec = add_global_specific_layers(
        decompose_multimutants(d, label_col="DMS_score", max_distance=2), label_col="DMS_score")
    dec = dec[np.isfinite(dec["epsilon_specific"])]
    info = dec.set_index("mutation_notation")[
        ["ddG_additive", "ddG_global", "DMS_score", "epsilon_specific"]].to_dict("index")

    F_, y, pll, pos, D, use_shift = load(feat)
    ids = F_["ids"]
    oof = np.full(len(y), np.nan)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(y):
        oof[te] = train_fold(F_, y, tr, te, D, epochs=epochs)

    rows = []
    for k in range(len(y)):
        a = ids[k]
        if a.max() >= 20:
            continue
        tok1 = f"{AA[a[2]]}{int(pos[k,0])}{AA[a[0]]}"
        tok2 = f"{AA[a[3]]}{int(pos[k,1])}{AA[a[1]]}"
        rec = info.get(f"{tok1}:{tok2}")
        if rec is None:
            continue
        # pll[k] = zero-shot DPLM PLL epistasis for this double (the best-of-N
        # rerank reward model — PLM-as-reward, the standard design baseline).
        rows.append((rec["ddG_global"], rec["DMS_score"], rec["epsilon_specific"],
                     float(oof[k]), float(pll[k])))
    return pd.DataFrame(
        rows, columns=["glob", "measured", "true_eps", "pred_eps", "pll_eps"]).dropna()


def summarize_model2(df):
    """Derive Model 2's paper numbers purely from the committed per-double table
    (columns: glob, measured, true_eps, pred_eps, pll_eps).

    (A) gain-of-function recovery: within the additively-MEDIOCRE pool (below the
        median global prediction, where additive rates everything low and cannot
        tell designs apart), the mean measured binding of the top 10% reranked by
        each reward model — Model 1's predicted epistasis, the zero-shot DPLM PLL
        epistasis (best-of-N rerank, the standard PLM-as-reward design baseline),
        and the oracle — vs the additive pool mean.
    (B) matched-additive control: median over global-prediction deciles of the
        within-bin Spearman(predicted epistasis, measured) — epistasis signal
        that survives at matched additive is not relearned additivity."""
    base = df["glob"].to_numpy()
    meas = df["measured"].to_numpy()
    pred_eps = df["pred_eps"].to_numpy()
    true_eps = df["true_eps"].to_numpy()
    pll_eps = df["pll_eps"].to_numpy()

    low = base < np.median(base)
    meas_low, pe_low, te_low, pll_low = meas[low], pred_eps[low], true_eps[low], pll_eps[low]

    def topmean(score, f):
        k = max(1, int(len(score) * f))
        return float(meas_low[np.argsort(-score)[:k]].mean())

    pool_mean = float(meas_low.mean())
    top10_m1 = topmean(pe_low, 0.1)
    top10_orc = topmean(te_low, 0.1)
    top10_zs = topmean(pll_low, 0.1)

    qs = np.quantile(base, np.linspace(0, 1, 11))
    mb = []
    for i in range(10):
        sel = (base >= qs[i]) & (base <= qs[i + 1])
        if sel.sum() >= 30:
            mb.append(float(spearmanr(pred_eps[sel], meas[sel]).statistic))
    matched = float(np.nanmedian(mb)) if mb else float("nan")

    return {
        "n_doubles": int(len(df)),
        "pool_mean_binding": pool_mean,
        "gof_top10pct_model1": top10_m1,
        "gof_top10pct_zeroshot": top10_zs,
        "gof_top10pct_oracle": top10_orc,
        "gof_lift_model1": top10_m1 - pool_mean,
        "gof_lift_zeroshot": top10_zs - pool_mean,
        "matched_additive_spearman": matched,
        "n_matched_bins": int(len(mb)),
    }


def run_model2(feat, assay_csv, assay="GB1", out="outputs/epistasis", epochs=200):
    """Train Model 2's ranking, write the committed DoD #3 artifacts, return the summary."""
    df = build_oof_table(feat, assay_csv, epochs=epochs)
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"model2_oof_{assay}.csv"
    df.to_csv(csv_path, index=False)
    # summarize from the re-read CSV so the committed summary is a bit-exact pure
    # function of the committed CSV bytes (the headroom no-drift guarantee; a mean
    # of selected values is not rank-robust to a 1-ULP serialization shift).
    summ = summarize_model2(pd.read_csv(csv_path))
    summ["assay"] = assay
    (outdir / f"model2_{assay}_summary.json").write_text(json.dumps(summ, indent=2))
    return summ


def figure_model2(csv, out="paper/figures_epistasis"):
    """Regenerate fig9 from the committed per-double CSV (no retraining)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
    })
    df = pd.read_csv(csv)
    base, meas = df["glob"].to_numpy(), df["measured"].to_numpy()
    low = base < np.median(base)
    meas_low = meas[low]
    pe_low, te_low = df["pred_eps"].to_numpy()[low], df["true_eps"].to_numpy()[low]
    pll_low = df["pll_eps"].to_numpy()[low]
    pool = meas_low.mean()

    def topmean(score, f):
        k = max(1, int(len(score) * f))
        return meas_low[np.argsort(-score)[:k]].mean()

    em = [topmean(pe_low, f) for f in FRACS]
    eo = [topmean(te_low, f) for f in FRACS]
    ez = [topmean(pll_low, f) for f in FRACS]

    qs = np.quantile(base, np.linspace(0, 1, 11))
    mb = []
    for i in range(10):
        sel = (base >= qs[i]) & (base <= qs[i + 1])
        if sel.sum() >= 30:
            mb.append(spearmanr(df["pred_eps"].to_numpy()[sel], meas[sel]).statistic)
    matched = float(np.nanmedian(mb))

    fig, ax = plt.subplots(1, 2, figsize=(9.8, 4.2))
    ax[0].axhline(pool, color=GREY, ls="--", lw=1.5, label="additive baseline (rates these equal)")
    ax[0].plot([f * 100 for f in FRACS], ez, "s-", color="#7B7B7B", lw=1.4, label="best-of-N rerank (zero-shot DPLM PLL)")
    ax[0].plot([f * 100 for f in FRACS], em, "o-", color=RED, lw=2.0, label="Model 1 (rank by predicted epistasis)")
    ax[0].plot([f * 100 for f in FRACS], eo, "o--", color=BLUE, lw=1.4, label="oracle (true epistasis)")
    ax[0].set_xlabel("Top fraction selected among additively-mediocre designs (%)")
    ax[0].set_ylabel("Mean measured binding")
    ax[0].set_title("A   Gain-of-function the additive model misses", loc="left", fontsize=12, fontweight="bold")
    ax[0].legend(loc="upper right", fontsize=8.4)

    ax[1].bar(range(len(mb)), mb, color=RED, alpha=0.85)
    ax[1].axhline(0, color="#222", lw=0.8)
    ax[1].set_xlabel("Additive-prediction bin (low → high)")
    ax[1].set_ylabel("Spearman(predicted epistasis, measured)")
    ax[1].set_title(f"B   Matched-additive control (median ρ={matched:.2f})\nepistasis signal additive cannot see",
                    loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    outp = Path(out)
    # metadata CreationDate=None drops the embedded timestamp so the PDF is byte-reproducible.
    fig.savefig(outp / "fig9_model2_design.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(outp / "fig9_model2_design.png", bbox_inches="tight")
    print("saved fig9_model2_design")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--assay-csv", required=True)
    ap.add_argument("--assay", default="GB1")
    ap.add_argument("--out", default="outputs/epistasis")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--figure", action="store_true", help="also regenerate fig9 from the committed CSV")
    args = ap.parse_args()
    s = run_model2(args.feat, args.assay_csv, assay=args.assay, out=args.out, epochs=args.epochs)
    print(f"[model2] {s['n_doubles']} held-out doubles merged")
    print(f"[model2] gain-of-function recovery (additively-mediocre pool, mean binding={s['pool_mean_binding']:.2f}): "
          f"top-10% by Model1={s['gof_top10pct_model1']:.2f} (lift +{s['gof_lift_model1']:.2f})  "
          f"zero-shot rerank={s['gof_top10pct_zeroshot']:.2f} (lift +{s['gof_lift_zeroshot']:.2f})  "
          f"oracle={s['gof_top10pct_oracle']:.2f}")
    print(f"[model2] matched-additive: median within-bin Spearman(pred_eps, measured) = "
          f"{s['matched_additive_spearman']:.3f} over {s['n_matched_bins']} bins")
    print(f"[model2] wrote outputs to {args.out}/model2_{args.assay}_*")
    if args.figure:
        figure_model2(Path(args.out) / f"model2_oof_{args.assay}.csv")


if __name__ == "__main__":
    main()
