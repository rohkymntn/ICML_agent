"""Optional structure ingestion and contact-graph helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_structure(path: str | Path):
    """Load a PDB structure with Biopython when available."""
    try:
        from Bio.PDB import PDBParser
    except Exception as exc:  # pragma: no cover
        raise ImportError("Biopython is required for structure loading.") from exc
    parser = PDBParser(QUIET=True)
    return parser.get_structure(Path(path).stem, str(path))


def extract_ca_coordinates(structure) -> pd.DataFrame:
    """Extract C-alpha coordinates from a Biopython structure."""
    rows = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" not in residue:
                    continue
                atom = residue["CA"]
                resseq = residue.get_id()[1]
                rows.append(
                    {
                        "chain_id": chain.id,
                        "residue_number": int(resseq),
                        "residue_name": residue.get_resname(),
                        "x": float(atom.coord[0]),
                        "y": float(atom.coord[1]),
                        "z": float(atom.coord[2]),
                    }
                )
    return pd.DataFrame(rows)


def build_contact_graph(coords: pd.DataFrame, cutoff: float = 8.0) -> pd.DataFrame:
    """Build an undirected residue contact graph from C-alpha coordinates."""
    required = {"x", "y", "z"}
    if not required.issubset(coords.columns) or len(coords) == 0:
        return pd.DataFrame(columns=["i", "j", "distance"])
    xyz = coords[["x", "y", "z"]].to_numpy(dtype=float)
    rows = []
    for i in range(len(coords)):
        diff = xyz[i + 1 :] - xyz[i]
        dist = np.sqrt(np.sum(diff**2, axis=1))
        hits = np.where(dist <= cutoff)[0]
        for h in hits:
            j = i + 1 + int(h)
            rows.append({"i": i, "j": j, "distance": float(dist[h])})
    return pd.DataFrame(rows)


def compute_contact_map(coords: pd.DataFrame, cutoff: float = 8.0) -> np.ndarray:
    """Return a binary residue contact map."""
    n = len(coords)
    mat = np.zeros((n, n), dtype=int)
    edges = build_contact_graph(coords, cutoff=cutoff)
    for _, e in edges.iterrows():
        i, j = int(e["i"]), int(e["j"])
        mat[i, j] = 1
        mat[j, i] = 1
    return mat


def map_positions_to_residues(mutation_positions, coords: pd.DataFrame) -> pd.DataFrame:
    """Map 1-indexed mutation positions onto available residue rows by order."""
    pos = np.asarray(list(mutation_positions), dtype=int)
    rows = []
    for p in pos:
        idx = p - 1
        if 0 <= idx < len(coords):
            row = coords.iloc[idx].to_dict()
            row["position"] = int(p)
            row["mapped"] = True
        else:
            row = {"position": int(p), "mapped": False}
        rows.append(row)
    return pd.DataFrame(rows)


def map_mutations_to_structure(mutations: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    """Join mutation token rows with approximate residue coordinates."""
    if "position" not in mutations.columns:
        raise ValueError("mutations must contain a 'position' column")
    mapping = map_positions_to_residues(mutations["position"].unique(), coords)
    return mutations.merge(mapping, on="position", how="left")


def export_epistasis_field_to_pdb_bfactor(
    pdb_in: str | Path,
    pdb_out: str | Path,
    scores: dict[int, float],
) -> None:
    """Write a PDB with per-position scores stored in the B-factor column."""
    structure = load_structure(pdb_in)
    for model in structure:
        for chain in model:
            ordered = [res for res in chain if "CA" in res]
            for idx, residue in enumerate(ordered, start=1):
                val = float(scores.get(idx, 0.0))
                for atom in residue:
                    atom.set_bfactor(val)
    try:
        from Bio.PDB import PDBIO
    except Exception as exc:  # pragma: no cover
        raise ImportError("Biopython is required for PDB export.") from exc
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(pdb_out))
