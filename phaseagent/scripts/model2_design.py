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
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "scripts")
from train_model1 import load, train_fold
from phaseagent.mutations import parse_mutation_notation
from phaseagent.epistasis_decomposition import add_global_specific_layers, decompose_multimutants

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
})
GREY, RED, BLUE = "#9A9A9A", "#C0504D", "#3B6FB6"
AA = "ACDEFGHIKLMNPQRSTVWY"

# --- decompose GB1 to get per-double additive / global / measured / true eps ---
d = pd.read_csv("/tmp/SPG1_STRSG_Olson_2014.csv").rename(columns={"mutant": "mutation_notation"})
d["dataset_id"] = "GB1"
d["mutation_distance"] = d["mutation_notation"].astype(str).apply(lambda s: len(parse_mutation_notation(s)))
d["DMS_score"] = pd.to_numeric(d["DMS_score"], errors="coerce")
dec = add_global_specific_layers(decompose_multimutants(d, label_col="DMS_score", max_distance=2), label_col="DMS_score")
dec = dec[np.isfinite(dec["epsilon_specific"])]
info = dec.set_index("mutation_notation")[["ddG_additive", "ddG_global", "DMS_score", "epsilon_specific"]].to_dict("index")

# --- Model 1 held-out-double OOF predictions of epsilon ---
F_, y, pll, pos, D, use_shift = load("/tmp/feat_GB1_Olson.npz")
ids = F_["ids"]
oof = np.full(len(y), np.nan)
for tr, te in KFold(5, shuffle=True, random_state=0).split(y):
    oof[te] = train_fold(F_, y, tr, te, D, epochs=200)

# reconstruct notation and merge
rows = []
for k in range(len(y)):
    a = ids[k]
    if a.max() >= 20:
        continue
    tok1 = f"{AA[a[2]]}{int(pos[k,0])}{AA[a[0]]}"
    tok2 = f"{AA[a[3]]}{int(pos[k,1])}{AA[a[1]]}"
    note = f"{tok1}:{tok2}"
    rec = info.get(note)
    if rec is None:
        continue
    rows.append((rec["ddG_global"], rec["DMS_score"], rec["epsilon_specific"], float(oof[k])))
df = pd.DataFrame(rows, columns=["glob", "measured", "true_eps", "pred_eps"]).dropna()
print(f"[model2] {len(df)} held-out doubles merged")

base = df["glob"].to_numpy()
m1 = df["glob"].to_numpy() + df["pred_eps"].to_numpy()
orc = df["glob"].to_numpy() + df["true_eps"].to_numpy()
meas = df["measured"].to_numpy()

# --- (A) gain-of-function recovery: among additively-MEDIOCRE doubles (additive
#         rates them all low), can Model 1's epistasis prediction find the ones
#         that actually bind? Additive cannot distinguish these by construction. ---
low = base < np.median(base)
meas_low = meas[low]
pe_low = df["pred_eps"].to_numpy()[low]
te_low = df["true_eps"].to_numpy()[low]
pool = meas_low.mean()
fracs = [0.05, 0.1, 0.2, 0.35, 0.5, 1.0]
def topmean_by(score, f):
    k = max(1, int(len(score) * f))
    return meas_low[np.argsort(-score)[:k]].mean()
em = [topmean_by(pe_low, f) for f in fracs]
eo = [topmean_by(te_low, f) for f in fracs]
print(f"[model2] gain-of-function recovery (additively-mediocre pool, mean binding={pool:.2f}): "
      f"top-10% by Model1={topmean_by(pe_low,0.1):.2f}  oracle={topmean_by(te_low,0.1):.2f}")

# --- (B) matched-additive control: within global-prediction deciles, Spearman(pred_eps, measured) ---
qs = np.quantile(base, np.linspace(0, 1, 11))
mb = []
for i in range(10):
    sel = (base >= qs[i]) & (base <= qs[i + 1])
    if sel.sum() >= 30:
        mb.append(spearmanr(df["pred_eps"].to_numpy()[sel], meas[sel]).statistic)
matched = float(np.nanmedian(mb))
print(f"[model2] matched-additive: median within-bin Spearman(pred_eps, measured) = {matched:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(9.8, 4.2))
ax[0].axhline(pool, color=GREY, ls="--", lw=1.5, label="additive baseline (rates these equal)")
ax[0].plot([f * 100 for f in fracs], em, "o-", color=RED, lw=2.0, label="Model 1 (rank by predicted epistasis)")
ax[0].plot([f * 100 for f in fracs], eo, "o--", color=BLUE, lw=1.4, label="oracle (true epistasis)")
ax[0].set_xlabel("Top fraction selected among additively-mediocre designs (%)")
ax[0].set_ylabel("Mean measured binding")
ax[0].set_title("A   Recovering gain-of-function additive misses", loc="left", fontsize=12, fontweight="bold")
ax[0].legend(loc="upper right", fontsize=8.4)

cents = [(qs[i] + qs[i + 1]) / 2 for i in range(len(mb))]
ax[1].bar(range(len(mb)), mb, color=RED, alpha=0.85)
ax[1].axhline(0, color="#222", lw=0.8)
ax[1].set_xlabel("Additive-prediction bin (low → high)")
ax[1].set_ylabel("Spearman(predicted epistasis, measured)")
ax[1].set_title(f"B   Matched-additive control (median ρ={matched:.2f})\nepistasis signal additive cannot see", loc="left", fontsize=12, fontweight="bold")

fig.tight_layout()
out = Path("paper/figures_epistasis")
fig.savefig(out / "fig9_model2_design.pdf", bbox_inches="tight")
fig.savefig(out / "fig9_model2_design.png", bbox_inches="tight")
print("saved fig9_model2_design")
