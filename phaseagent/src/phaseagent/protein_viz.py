"""Publication-quality protein structure visualization.

Pure-Python (biotite + matplotlib) so it integrates with our existing
``nature_figures`` workflow and produces real PDF/SVG output. Three primary
helpers:

- ``render_structure_colored_by_score``: ribbon-style backbone trace with
  per-residue color from a (residue → value) mapping. Used to project rescue
  probability, ΔΔG, or epistasis-residual onto AlphaFold structures.
- ``render_local_neighborhood``: zoomed-in view around a (m1, m2) pair
  showing the surrounding contact shell — for Fig 5 case studies.
- ``contact_map``: matplotlib 2-D contact map with optional rescue overlay.

All functions return matplotlib Figures so the caller can lay them out
inside their own GridSpec, save to PDF/SVG, etc.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib import cm as mpl_cm

    _HAS_MPL = True
except Exception:  # pragma: no cover
    _HAS_MPL = False

try:
    import biotite.structure as struc
    import biotite.structure.io.pdb as pdb_io

    _HAS_BIOTITE = True
except Exception:  # pragma: no cover
    _HAS_BIOTITE = False


# Three-letter to one-letter amino-acid table (kept here to avoid extra imports).
_AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


@dataclass
class ProteinStructure:
    """Light-weight protein structure container used by the rendering helpers."""

    name: str
    coords_ca: np.ndarray            # (L, 3)
    sequence: str                    # length L
    res_ids: np.ndarray              # (L,) integer residue numbers (1-indexed)
    secondary_structure: str | None  # "HEEELL...." or None if unavailable


def load_pdb(path: str | Path, name: str | None = None) -> ProteinStructure:
    """Load a single-chain PDB into our minimal container."""
    if not _HAS_BIOTITE:
        raise ImportError("protein_viz.load_pdb requires biotite")
    p = Path(path)
    pdb_file = pdb_io.PDBFile.read(str(p))
    atoms = pdb_file.get_structure(model=1)
    # Keep only the first chain to keep things simple.
    if hasattr(atoms, "chain_id"):
        first_chain = atoms.chain_id[0]
        atoms = atoms[atoms.chain_id == first_chain]
    ca = atoms[atoms.atom_name == "CA"]
    coords = np.asarray(ca.coord, dtype=np.float32)
    res_names = list(ca.res_name)
    seq = "".join(_AA3TO1.get(r, "X") for r in res_names)
    res_ids = np.asarray(ca.res_id, dtype=int)
    return ProteinStructure(
        name=name or p.stem,
        coords_ca=coords,
        sequence=seq,
        res_ids=res_ids,
        secondary_structure=None,
    )


def _project_to_2d(coords: np.ndarray, view: str = "xy") -> np.ndarray:
    """Project (L, 3) -> (L, 2) using a simple axis projection.

    For paper figures we keep this deterministic and orthographic. Callers
    that need fancier camera angles can pre-rotate the coordinates.
    """
    if view == "xy":
        return coords[:, [0, 1]]
    if view == "xz":
        return coords[:, [0, 2]]
    if view == "yz":
        return coords[:, [1, 2]]
    raise ValueError(f"unknown view {view!r}")


def _principal_axes_view(coords: np.ndarray) -> np.ndarray:
    """Project to the two principal axes of the structure (PCA).

    Gives a deterministic, view-invariant 2-D layout that fits the page well.
    """
    centered = coords - coords.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def render_structure_colored_by_score(
    structure: ProteinStructure,
    score: np.ndarray,
    *,
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    highlight_residues: Sequence[int] = (),
    title: str = "",
    figsize: tuple[float, float] = (3.5, 3.5),
    line_width: float = 1.6,
    bg_alpha: float = 1.0,
) -> "plt.Figure":
    """Backbone trace colored by a per-residue score.

    ``score`` is a length-L array indexed by residue position (matching
    ``structure.coords_ca``). ``highlight_residues`` are 1-indexed residue
    numbers that get a red ring overlay (e.g. damaging variant + best rescue).
    """
    if not _HAS_MPL:
        raise ImportError("protein_viz.render_structure_colored_by_score requires matplotlib")
    if len(score) != len(structure.coords_ca):
        raise ValueError(
            f"score length {len(score)} != n residues {len(structure.coords_ca)}"
        )
    coords_2d = _principal_axes_view(structure.coords_ca)
    score = np.asarray(score, dtype=float)
    finite = np.isfinite(score)
    if vmin is None:
        vmin = float(np.nanpercentile(score[finite], 2)) if finite.any() else 0.0
    if vmax is None:
        vmax = float(np.nanpercentile(score[finite], 98)) if finite.any() else 1.0
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = mpl_cm.get_cmap(cmap)
    colors = cmap_obj(norm(score))

    # Draw the backbone as line segments colored by per-residue score.
    fig, ax = plt.subplots(figsize=figsize)
    segs = np.stack([coords_2d[:-1], coords_2d[1:]], axis=1)
    seg_colors = colors[:-1]
    lc = LineCollection(segs, colors=seg_colors, linewidths=line_width, alpha=bg_alpha)
    ax.add_collection(lc)
    # Plot per-residue dots so missing scores remain visible.
    ax.scatter(coords_2d[:, 0], coords_2d[:, 1], s=10, c=colors,
               edgecolor="white", linewidth=0.3, zorder=3)

    # Highlight residues (e.g. damaging + suppressor) with red rings.
    if highlight_residues:
        highlight_idx = []
        for r in highlight_residues:
            mask = structure.res_ids == int(r)
            if mask.any():
                highlight_idx.append(int(np.where(mask)[0][0]))
        if highlight_idx:
            ax.scatter(coords_2d[highlight_idx, 0], coords_2d[highlight_idx, 1],
                       s=80, facecolor="none", edgecolor="#D55E00",
                       linewidth=1.5, zorder=4, label="highlighted")

    # Cosmetics.
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=8)
    sm = mpl_cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.ax.tick_params(labelsize=6)
    return fig


def render_local_neighborhood(
    structure: ProteinStructure,
    center_residues: Sequence[int],
    radius: float = 8.0,
    *,
    color_by_score: np.ndarray | None = None,
    cmap: str = "RdBu_r",
    figsize: tuple[float, float] = (3.0, 3.0),
    title: str = "",
) -> "plt.Figure":
    """Zoomed-in 2-D view of the local neighborhood around given residues.

    Useful for the "case study close-up" panel in Fig 5: showing how a
    putative rescue mutation sits relative to the damaging variant in 3D.
    """
    if not _HAS_MPL:
        raise ImportError("matplotlib required")
    coords = structure.coords_ca
    centers = []
    for r in center_residues:
        mask = structure.res_ids == int(r)
        if mask.any():
            centers.append(int(np.where(mask)[0][0]))
    if not centers:
        raise ValueError("no center residues found in structure")
    in_neighborhood = np.zeros(len(coords), dtype=bool)
    for c in centers:
        d = np.linalg.norm(coords - coords[c], axis=-1)
        in_neighborhood |= d <= float(radius)
    in_neighborhood[centers] = True
    sub_coords = coords[in_neighborhood]
    sub_idx = np.where(in_neighborhood)[0]
    coords_2d = _principal_axes_view(sub_coords)

    fig, ax = plt.subplots(figsize=figsize)
    if color_by_score is not None and len(color_by_score) == len(coords):
        score_sub = np.asarray(color_by_score, dtype=float)[sub_idx]
        finite = np.isfinite(score_sub)
        vmin = float(np.nanpercentile(score_sub[finite], 5)) if finite.any() else 0.0
        vmax = float(np.nanpercentile(score_sub[finite], 95)) if finite.any() else 1.0
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        colors = mpl_cm.get_cmap(cmap)(norm(score_sub))
    else:
        colors = np.tile(np.array([0.5, 0.5, 0.5, 1.0]), (len(coords_2d), 1))
        norm = None
    ax.scatter(coords_2d[:, 0], coords_2d[:, 1], s=30, c=colors,
               edgecolor="black", linewidth=0.4)
    # Highlight the center residues by mapping them back.
    center_local_idx = []
    for c in centers:
        mask = sub_idx == c
        if mask.any():
            center_local_idx.append(int(np.where(mask)[0][0]))
    if center_local_idx:
        ax.scatter(coords_2d[center_local_idx, 0], coords_2d[center_local_idx, 1],
                   s=140, facecolor="none", edgecolor="#D55E00", linewidth=1.5, zorder=5)
        for li, c in zip(center_local_idx, centers):
            res_num = int(structure.res_ids[c])
            res_aa = structure.sequence[c] if c < len(structure.sequence) else "?"
            ax.annotate(f"{res_aa}{res_num}", (coords_2d[li, 0], coords_2d[li, 1]),
                        textcoords="offset points", xytext=(8, 6), fontsize=7,
                        fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=8)
    return fig


def contact_map(
    structure: ProteinStructure,
    *,
    contact_radius: float = 8.0,
    overlay_pairs: Sequence[tuple[int, int]] = (),
    figsize: tuple[float, float] = (3.5, 3.5),
    title: str = "",
) -> "plt.Figure":
    """Per-residue contact map; overlay pairs are 1-indexed (m1, m2) tuples."""
    if not _HAS_MPL:
        raise ImportError("matplotlib required")
    coords = structure.coords_ca
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    contact = (dist <= float(contact_radius)).astype(float)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(contact, cmap="Greys", origin="lower", aspect="equal", vmin=0, vmax=1)
    ax.set_xlabel("residue position")
    ax.set_ylabel("residue position")
    if title:
        ax.set_title(title, fontsize=8)
    res_to_idx = {int(r): i for i, r in enumerate(structure.res_ids)}
    for r1, r2 in overlay_pairs:
        i1 = res_to_idx.get(int(r1))
        i2 = res_to_idx.get(int(r2))
        if i1 is None or i2 is None:
            continue
        ax.scatter([i2], [i1], s=60, facecolor="none", edgecolor="#D55E00", linewidth=1.2)
        ax.scatter([i1], [i2], s=60, facecolor="none", edgecolor="#0072B2", linewidth=1.2)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label(f"|Cα-Cα| ≤ {contact_radius:.0f} Å", fontsize=6)
    return fig
