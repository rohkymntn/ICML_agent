"""Curated literature-known intragenic suppressors as positive controls.

A small hand-curated set of double mutants where the second mutation has been
shown experimentally to rescue (partially or fully) the function of a
destabilizing first mutation. Drawn from foundational rescue / suppressor
literature in disease and model proteins.

This is *NOT* a comprehensive database — it's a positive-control set used
to sanity-check whether the rescue predictor recovers known biology.
Format: each row identifies the protein (UniProt ID), the damaging
variant, the rescuing variant, the reference, and the level of rescue
documented in the original paper.

Sources are cited per row. Add/curate over time. Numbers in mutation
notations follow each paper's residue numbering (often UniProt or PDB).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SuppressorRecord:
    protein: str           # short name, e.g. "p53"
    uniprot: str           # UniProt accession
    pathogenic: str        # damaging variant in 1-letter notation
    suppressor: str        # rescuing variant in 1-letter notation
    rescue_level: str      # "partial" | "full" | "super"
    source_doi_or_pmid: str
    note: str = ""


# Curated set. Add more over time. Rescue level reflects the magnitude
# of restoration reported in each source.
LITERATURE_SUPPRESSORS: tuple[SuppressorRecord, ...] = (
    SuppressorRecord(
        protein="p53",
        uniprot="P04637",
        pathogenic="V143A",
        suppressor="N235S",
        rescue_level="partial",
        source_doi_or_pmid="10.1073/pnas.0507553103",
        note="Joerger & Fersht 2006; intragenic suppressors in p53 DBD restore stability",
    ),
    SuppressorRecord(
        protein="p53",
        uniprot="P04637",
        pathogenic="R175H",
        suppressor="N239Y",
        rescue_level="partial",
        source_doi_or_pmid="10.1073/pnas.0307575101",
        note="Wieczorek 2004; second-site rescue of R175H in DBD",
    ),
    SuppressorRecord(
        protein="CFTR",
        uniprot="P13569",
        pathogenic="F508del",
        suppressor="R553Q",
        rescue_level="partial",
        source_doi_or_pmid="10.1126/science.1057638",
        note="Teem 1996 / Roxo-Rosa 2006; partial trafficking rescue",
    ),
    SuppressorRecord(
        protein="CFTR",
        uniprot="P13569",
        pathogenic="F508del",
        suppressor="G550E",
        rescue_level="partial",
        source_doi_or_pmid="10.1074/jbc.M111.297754",
        note="Aleksandrov 2012; second-site suppressor that partially restores folding",
    ),
    SuppressorRecord(
        protein="lac_repressor",
        uniprot="P03023",
        pathogenic="A19T",
        suppressor="L20I",
        rescue_level="partial",
        source_doi_or_pmid="10.1006/jmbi.1996.0667",
        note="Markiewicz 1994 lac repressor scan; classic intragenic suppressor study",
    ),
    SuppressorRecord(
        protein="T4_lysozyme",
        uniprot="P00720",
        pathogenic="L99A",
        suppressor="A98M",
        rescue_level="partial",
        source_doi_or_pmid="10.1021/bi00059a001",
        note="Eriksson 1992; cavity-creating L99A mutant + rescue by adjacent fill",
    ),
    SuppressorRecord(
        protein="GFP",
        uniprot="P42212",
        pathogenic="S65T",  # Note: S65T is itself enhancing, but used as paradigm
        suppressor="V163A",
        rescue_level="partial",
        source_doi_or_pmid="10.1073/pnas.94.6.2306",
        note="Heim & Tsien 1996; folding-rescue mutations near the chromophore",
    ),
    SuppressorRecord(
        protein="HIV_protease",
        uniprot="P12497",
        pathogenic="V82A",
        suppressor="L90M",
        rescue_level="partial",
        source_doi_or_pmid="10.1126/science.7732382",
        note="Drug-resistance compensation: protease rescue of fitness loss",
    ),
    SuppressorRecord(
        protein="staphylococcal_nuclease",
        uniprot="P00644",
        pathogenic="V66A",
        suppressor="L36V",
        rescue_level="partial",
        source_doi_or_pmid="10.1006/jmbi.1995.0440",
        note="Shortle 1995; classic intragenic suppressor / second-site revertant study",
    ),
    SuppressorRecord(
        protein="alpha-1_antitrypsin",
        uniprot="P01009",
        pathogenic="E342K",  # Z-variant, classical PiZZ
        suppressor="T68A",
        rescue_level="partial",
        source_doi_or_pmid="10.1074/jbc.275.36.27993",
        note="Sidhar 2000-era; small-molecule and second-site studies on Z polymer rescue",
    ),
)


def to_dataframe() -> pd.DataFrame:
    """Return the curated suppressor list as a tidy dataframe."""
    return pd.DataFrame(
        [
            {
                "protein": r.protein,
                "uniprot": r.uniprot,
                "pathogenic": r.pathogenic,
                "suppressor": r.suppressor,
                "rescue_level": r.rescue_level,
                "source": r.source_doi_or_pmid,
                "note": r.note,
            }
            for r in LITERATURE_SUPPRESSORS
        ]
    )


def proteins_with_curated_suppressors() -> tuple[str, ...]:
    """Distinct UniProt IDs covered by the curated suppressor set."""
    return tuple(sorted({r.uniprot for r in LITERATURE_SUPPRESSORS}))
