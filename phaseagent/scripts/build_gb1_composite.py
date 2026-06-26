"""Publication composite: GB1 gain-of-function epistasis = a 3D contact.

Panel A: PyMOL render of GB1 (1PGA) with the four epistatic positions, the
         gain-of-function pair G41xV54 in 3D contact (5.5 A).
Panel B: additive prediction vs measured binding; points above the diagonal are
         gain-of-function (individually weak, together strong), colored by
         specific epistasis, with the G41xV54 family circled.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from phaseagent.mutations import parse_mutation_notation
from phaseagent.epistasis_decomposition import add_global_specific_layers, decompose_multimutants

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
})

d = pd.read_csv("/tmp/SPG1_STRSG_Wu_2016.csv").rename(columns={"mutant": "mutation_notation"})
d["dataset_id"] = "GB1"
d["mutation_distance"] = d["mutation_notation"].astype(str).apply(lambda s: len(parse_mutation_notation(s)))
d["DMS_score"] = pd.to_numeric(d["DMS_score"], errors="coerce")
sing = d[d["mutation_distance"] == 1]
slk = dict(zip(sing["mutation_notation"].astype(str), sing["DMS_score"]))
dec = add_global_specific_layers(decompose_multimutants(d, label_col="DMS_score", max_distance=2), label_col="DMS_score")
b = dec[np.isfinite(dec["epsilon_specific"])].copy()
b["s1"] = b["mutation_notation"].apply(lambda n: slk.get(parse_mutation_notation(n)[0], np.nan))
b["s2"] = b["mutation_notation"].apply(lambda n: slk.get(parse_mutation_notation(n)[1], np.nan))
b = b[np.isfinite(b["s1"]) & np.isfinite(b["s2"])]

fig = plt.figure(figsize=(10.2, 4.7))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.08)

axA = fig.add_subplot(gs[0])
axA.imshow(mpimg.imread("/tmp/gb1_structure.png"))
axA.axis("off")
axA.set_title("A   The epistatic pair is a 3D contact", loc="left", fontsize=12.5, fontweight="bold")

axB = fig.add_subplot(gs[1])
add = b["ddG_additive"].to_numpy(float)
obs = b["DMS_score"].to_numpy(float)
eps = b["epsilon_specific"].to_numpy(float)
lim = [min(add.min(), obs.min()) - 0.3, max(add.max(), obs.max()) + 0.3]
axB.fill_between(lim, lim, lim[1] + 1, color="#C0504D", alpha=0.05, zorder=0)
axB.text(0.4, lim[1] - 0.3, "gain of function", color="#C0504D", fontsize=9, style="italic")
sc = axB.scatter(add, obs, c=eps, cmap="coolwarm", vmin=-3, vmax=3, s=9, alpha=0.6, linewidths=0, zorder=2)
axB.plot(lim, lim, "--", color="#222", lw=1.0, label="additive (y = x)", zorder=3)
fam = b[b["mutation_notation"].str.contains("267") & b["mutation_notation"].str.contains("280")]
axB.scatter(fam["ddG_additive"], fam["DMS_score"], s=46, facecolors="none", edgecolors="#7A1F1C", lw=1.4, label="G41 × V54 pairs", zorder=4)
gl = b[b["mutation_notation"].astype(str).str.upper().str.replace(" ", "") == "G267L:V280G"]
if len(gl):
    r = gl.iloc[0]
    axB.annotate("G41L + V54G\n0.01 + 0.27 → 3.15",
                 xy=(float(r["ddG_additive"]), float(r["DMS_score"])),
                 xytext=(lim[0] + 0.2, lim[1] - 1.4), fontsize=8.5, color="#7A1F1C",
                 arrowprops=dict(arrowstyle="->", color="#7A1F1C", lw=1.0))
axB.set_xlim(lim); axB.set_ylim(lim)
axB.set_xlabel("Additive prediction  (sum of single-mutant binding)")
axB.set_ylabel("Measured binding of the double mutant")
axB.set_title("B   Individually weak, together strong", loc="left", fontsize=12.5, fontweight="bold")
axB.legend(loc="lower right", fontsize=8.8)
cb = fig.colorbar(sc, ax=axB, shrink=0.85, pad=0.02)
cb.set_label("specific epistasis", fontsize=9)

out = Path("paper/figures_epistasis")
fig.savefig(out / "fig7_gb1_main.pdf", bbox_inches="tight")
fig.savefig(out / "fig7_gb1_main.png", bbox_inches="tight")
# also keep the standalone structure render in the figure folder
import shutil
shutil.copy("/tmp/gb1_structure.png", out / "fig7a_gb1_structure.png")
print("saved fig7_gb1_main + fig7a_gb1_structure")
