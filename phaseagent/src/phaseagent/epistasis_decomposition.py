"""Additive / specific-epistasis decomposition of multi-mutant stability.

For Megascale ``ddG_ML`` (a free energy in kcal/mol), additivity is the
*physical* null: the expected stability change of a multi-mutant is the sum of
its single-mutant ``ddG`` values. The deviation from that sum is *specific
epistasis* -- the part of the multi-mutant landscape that single-mutant scans
cannot determine, and the only part a learned model can beat the additive
baseline on:

    eps(S) = ddG_obs(S) - sum_{i in S} ddG_single(i)      # kcal/mol

This module computes that residual cleanly *within-assay* (the constituent
singles and the multi-mutant come from the same Megascale domain), and
summarizes per protein how much specific epistasis exists -- the "atlas" layer
and the go/no-go gate for the cross-protein epistasis operator (Model 1).

The complementary *global* epistasis layer (the nonlinear ddG -> fitness link
predicted in closed form from the single-mutant spectrum) lives in
``phase_validation``. This module is deliberately on the ddG free-energy scale,
where additivity is the natural null and ``eps`` is pure specific epistasis, so
that any held-out skill of a learned model on ``eps`` is genuine epistasis by
construction (the DeWitt-audit target).

Sign convention follows the rest of the repo: ``ddG_ML`` > 0 is stabilizing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .additive_baseline import AdditivePredictor
from .mutations import parse_mutation_notation
from .phase_validation import compute_spectrum_stats


def decompose_multimutants(
    df: pd.DataFrame,
    *,
    label_col: str = "ddG_ML",
    mutation_col: str = "mutation_notation",
    dataset_col: str = "dataset_id",
    distance_col: str = "mutation_distance",
    max_distance: int | None = None,
) -> pd.DataFrame:
    """Attach additive prediction and specific-epistasis residual to every
    multi-mutant row, using within-assay constituent singles.

    Parameters
    ----------
    df
        Canonical Megascale frame (see ``megascale.to_proteingym_schema``) that
        contains BOTH singles (``distance == 1``) and multi-mutants
        (``distance >= 2``) for the proteins of interest.
    label_col
        Free-energy label to decompose. ``ddG_ML`` is the physical scale where
        additivity is the natural null.

    Returns
    -------
    The multi-mutant rows (distance >= 2) with extra columns:
      - ``ddG_additive`` : sum of constituent single ddGs (NaN if any missing)
      - ``epsilon``      : observed - additive (specific epistasis, kcal/mol)
      - ``n_edits``      : number of substitutions
    Only rows whose every constituent single is measured in-assay get a finite
    ``ddG_additive`` / ``epsilon``.
    """
    dist = pd.to_numeric(df[distance_col], errors="coerce")
    singles = df[dist == 1]
    multis = df[dist >= 2].copy()
    if max_distance is not None:
        multis = multis[pd.to_numeric(multis[distance_col], errors="coerce") <= max_distance].copy()

    multis = multis.reset_index(drop=True)
    if len(singles) == 0 or len(multis) == 0:
        return multis.assign(
            ddG_additive=pd.Series(dtype=float),
            epsilon=pd.Series(dtype=float),
            n_edits=pd.Series(dtype=int),
        )

    predictor = AdditivePredictor(
        score_col=label_col,
        mutation_col=mutation_col,
        dataset_col=dataset_col,
    ).fit(singles)

    add = predictor.predict(multis)
    obs = pd.to_numeric(multis[label_col], errors="coerce").to_numpy(dtype=float)
    multis["ddG_additive"] = add
    multis["epsilon"] = obs - add
    multis["n_edits"] = multis[mutation_col].apply(
        lambda s: len(parse_mutation_notation(str(s)))
    )
    return multis


def add_global_specific_layers(
    decomposed: pd.DataFrame,
    *,
    label_col: str = "ddG_ML",
    dataset_col: str = "dataset_id",
    additive_col: str = "ddG_additive",
    min_points: int = 30,
) -> pd.DataFrame:
    """Split the additive residual into *global* and *specific* epistasis.

    Fits a per-protein monotonic global-epistasis link (the empirical,
    most-flexible counterpart of the closed-form spectrum link) mapping the
    additive latent to the observed scale, then defines:

        ddG_global       = link(sum of single ddGs)        # global epistasis (saturation)
        epsilon_specific = ddG_obs - ddG_global            # specific, residue-pair epistasis

    ``epsilon_specific`` is the DeWitt-audit target for the learned operator
    (Model 1): a residual after both additivity and a flexible monotonic link,
    so any held-out predictive skill on it is genuine specific epistasis.

    Note: the link is fit in-sample per protein, so the specific share it leaves
    is a (mild) upper bound on what a held-out link would leave; the cross-protein
    operator must be evaluated with a held-out / theory link.
    """
    from sklearn.isotonic import IsotonicRegression

    out = decomposed.copy().reset_index(drop=True)
    out["ddG_global"] = np.nan
    out["epsilon_specific"] = np.nan
    if additive_col not in out.columns:
        return out

    for _, idx in out.groupby(dataset_col).groups.items():
        sub = out.loc[idx]
        obs = pd.to_numeric(sub[label_col], errors="coerce").to_numpy(dtype=float)
        add = pd.to_numeric(sub[additive_col], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(obs) & np.isfinite(add)
        if int(m.sum()) < int(min_points):
            continue
        iso = IsotonicRegression(out_of_bounds="clip").fit(add[m], obs[m])
        pred = np.full(len(sub), np.nan)
        pred[m] = iso.predict(add[m])
        out.loc[idx, "ddG_global"] = pred
        out.loc[idx, "epsilon_specific"] = obs - pred
    return out


def protein_epistasis_atlas(
    df: pd.DataFrame,
    decomposed: pd.DataFrame | None = None,
    *,
    label_col: str = "ddG_ML",
    dataset_col: str = "dataset_id",
    distance_col: str = "mutation_distance",
    min_covered: int = 10,
) -> pd.DataFrame:
    """Per-protein summary of additive sufficiency vs. specific epistasis.

    For each domain, on its *covered* doubles/triples (those with a finite
    additive prediction), reports:
      - ``n_covered``           : usable multi-mutants
      - ``additive_r2``         : 1 - SS(eps)/SS(obs - mean_obs); additive power
      - ``epistasis_std``       : std(eps), kcal/mol -- absolute magnitude
      - ``epistasis_var_share`` : var(eps)/var(obs); fraction not explained additively
      - ``mean_eps``,``median_eps`` : directional bias (diminishing-returns sign)
      - ``m_g``,``sigma_g``,``hardness_H`` : single-mutant spectrum moments

    A protein with ``epistasis_std`` well above Megascale measurement noise
    (~0.1-0.4 kcal/mol) and a non-trivial ``epistasis_var_share`` is one where a
    learned epistasis model can, in principle, beat additive.
    """
    if decomposed is None:
        decomposed = decompose_multimutants(
            df, label_col=label_col, dataset_col=dataset_col, distance_col=distance_col
        )

    dist = pd.to_numeric(df[distance_col], errors="coerce")
    singles_all = df[dist == 1]

    rows: list[dict] = []
    for ds_id, grp in decomposed.groupby(dataset_col):
        cov = grp[np.isfinite(grp["epsilon"].to_numpy(dtype=float))]
        if len(cov) < int(min_covered):
            continue
        obs = pd.to_numeric(cov[label_col], errors="coerce").to_numpy(dtype=float)
        eps = cov["epsilon"].to_numpy(dtype=float)
        finite = np.isfinite(obs) & np.isfinite(eps)
        obs, eps = obs[finite], eps[finite]
        if len(obs) < int(min_covered):
            continue

        ss_tot = float(np.sum((obs - obs.mean()) ** 2))
        ss_res = float(np.sum(eps ** 2))
        additive_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        var_obs = float(np.var(obs, ddof=1)) if len(obs) > 1 else float("nan")
        var_eps = float(np.var(eps, ddof=1)) if len(eps) > 1 else float("nan")

        # Single-mutant spectrum for this domain (damage = -ddG, repo convention).
        s_grp = singles_all[singles_all[dataset_col] == ds_id]
        eff = -pd.to_numeric(s_grp[label_col], errors="coerce").to_numpy(dtype=float)
        eff = eff[np.isfinite(eff)]
        stats = compute_spectrum_stats(eff) if len(eff) >= 5 else None

        rows.append(
            {
                "dataset_id": str(ds_id),
                "n_covered": int(len(obs)),
                "n_singles": int(len(eff)),
                "additive_r2": float(additive_r2),
                "epistasis_std": float(np.std(eps, ddof=1)) if len(eps) > 1 else float("nan"),
                "epistasis_var_share": float(var_eps / var_obs) if var_obs and var_obs > 0 else float("nan"),
                "mean_eps": float(np.mean(eps)),
                "median_eps": float(np.median(eps)),
                "max_abs_eps": float(np.max(np.abs(eps))),
                "m_g": float(stats.m_g) if stats else float("nan"),
                "sigma_g": float(stats.sigma_g) if stats else float("nan"),
                "hardness_H": float(stats.hardness) if stats else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def epistasis_gate_summary(atlas: pd.DataFrame) -> dict:
    """Collapse the per-protein atlas into the go/no-go numbers.

    The decision the whole program hinges on: across proteins, is specific
    epistasis (a) real (magnitude >> measurement noise) and (b) widespread?
    """
    if len(atlas) == 0:
        return {"n_proteins": 0}

    def _med(col: str) -> float:
        v = pd.to_numeric(atlas[col], errors="coerce").to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        return float(np.median(v)) if v.size else float("nan")

    eps_std = pd.to_numeric(atlas["epistasis_std"], errors="coerce").to_numpy(dtype=float)
    eps_std = eps_std[np.isfinite(eps_std)]
    share = pd.to_numeric(atlas["epistasis_var_share"], errors="coerce").to_numpy(dtype=float)
    share = share[np.isfinite(share)]

    return {
        "n_proteins": int(len(atlas)),
        "total_covered_multimutants": int(pd.to_numeric(atlas["n_covered"], errors="coerce").sum()),
        "median_additive_r2": _med("additive_r2"),
        "median_epistasis_std_kcal": _med("epistasis_std"),
        "median_epistasis_var_share": _med("epistasis_var_share"),
        "frac_proteins_eps_std_gt_0p5": float(np.mean(eps_std > 0.5)) if eps_std.size else float("nan"),
        "frac_proteins_share_gt_0p1": float(np.mean(share > 0.1)) if share.size else float("nan"),
        "median_hardness_H": _med("hardness_H"),
    }
