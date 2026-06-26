"""ClinVar pathogenic-variant fetcher for the rescue atlas.

Pulls pathogenic + likely-pathogenic missense variants for a given gene
symbol via NCBI's E-utilities API. The NCBI API is public and unauthenticated;
batched fetches respect the recommended 3 requests/sec rate limit.

We deliberately keep this dependency-light (no PyClinVar / Pyscan) so it runs
inside our existing CPU image without adding hundreds of MB of dependencies.

Output schema, one row per variant, ready to feed into the rescue atlas:
    gene, uniprot, mutation_notation, clinvar_id, clinical_significance,
    review_status, condition, raw_hgvs_p, raw_hgvs_c
"""
from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlencode

import pandas as pd

try:
    import requests
    _HAS_REQUESTS = True
except Exception:  # pragma: no cover
    _HAS_REQUESTS = False


CLINVAR_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


# Single-letter code for parsing 3-letter HGVS p.dot notation (e.g. p.Arg175His).
AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "TER": "*", "STOP": "*",
}


@dataclass(frozen=True)
class ClinVarConfig:
    """Tuning knobs for the ClinVar fetcher."""

    pathogenic_terms: tuple[str, ...] = (
        "Pathogenic",
        "Likely_pathogenic",
        "Pathogenic/Likely_pathogenic",
    )
    max_per_gene: int = 500
    sleep_sec: float = 1.0  # 1 req/sec — leaves headroom for retries
    timeout_sec: float = 30.0


def _esearch_pathogenic(gene: str, config: ClinVarConfig) -> list[str]:
    """Find ClinVar Variation IDs for pathogenic missense in a gene."""
    if not _HAS_REQUESTS:
        raise ImportError("ClinVar fetch requires `requests`")
    sig_clause = " OR ".join(f'\"{s}\"[clin_sig]' for s in config.pathogenic_terms)
    term = f"({gene}[gene]) AND ({sig_clause}) AND \"missense variant\"[mol_cons]"
    params = {
        "db": "clinvar",
        "term": term,
        "retmax": str(int(config.max_per_gene)),
        "retmode": "json",
    }
    r = requests.get(f"{CLINVAR_BASE}/esearch.fcgi?{urlencode(params)}", timeout=config.timeout_sec)
    r.raise_for_status()
    data = r.json()
    ids = data.get("esearchresult", {}).get("idlist", [])
    return list(ids)


def _esummary_chunk(ids: list[str], config: ClinVarConfig) -> list[dict]:
    """Pull esummary records for a chunk of ClinVar IDs."""
    if not ids:
        return []
    params = {
        "db": "clinvar",
        "id": ",".join(ids),
        "retmode": "json",
    }
    r = requests.get(f"{CLINVAR_BASE}/esummary.fcgi?{urlencode(params)}", timeout=config.timeout_sec)
    r.raise_for_status()
    payload = r.json()
    result = payload.get("result", {})
    return [result[i] for i in ids if i in result]


def _parse_aa_change(hgvs_p: str) -> tuple[str, int, str] | None:
    """Parse 'p.Arg175His' / 'p.R175H' into (WT, pos, MUT) one-letter."""
    if not isinstance(hgvs_p, str):
        return None
    s = hgvs_p.strip()
    if s.startswith("p."):
        s = s[2:]
    s = s.replace("(", "").replace(")", "")
    if "=" in s or "fs" in s.lower() or "del" in s.lower() or "ins" in s.lower():
        return None
    # 3-letter form: e.g. "Arg175His"
    if len(s) >= 7 and s[:3].upper() in AA_3TO1 and s[-3:].upper() in AA_3TO1:
        wt3, mut3 = s[:3].upper(), s[-3:].upper()
        pos_str = s[3:-3]
        if pos_str.lstrip("-").isdigit():
            return AA_3TO1[wt3], int(pos_str), AA_3TO1[mut3]
    # 1-letter form: e.g. "R175H"
    if len(s) >= 3 and s[0].isalpha() and s[-1].isalpha():
        wt1, mut1 = s[0].upper(), s[-1].upper()
        pos_str = s[1:-1]
        if pos_str.lstrip("-").isdigit() and wt1 in AA_3TO1.values() and mut1 in AA_3TO1.values():
            return wt1, int(pos_str), mut1
    return None


def fetch_pathogenic_missense(
    gene: str,
    config: ClinVarConfig | None = None,
) -> pd.DataFrame:
    """Fetch pathogenic missense ClinVar records for one gene symbol.

    Returns one row per parsed (pathogenic) variant. Variants whose HGVS p.
    notation cannot be parsed are silently dropped (e.g. frameshifts,
    insertions, deletions, synonymous-codon repeats).
    """
    cfg = config or ClinVarConfig()
    ids = _esearch_pathogenic(gene, cfg)
    if not ids:
        return pd.DataFrame()
    # esummary in chunks of ≤200 ids per call.
    rows: list[dict] = []
    for start in range(0, len(ids), 200):
        chunk = ids[start : start + 200]
        time.sleep(cfg.sleep_sec)
        records = _esummary_chunk(chunk, cfg)
        for rec in records:
            cv_id = str(rec.get("uid", ""))
            sig = rec.get("germline_classification", {}).get("description", "") if isinstance(rec.get("germline_classification"), dict) else ""
            review = rec.get("germline_classification", {}).get("review_status", "") if isinstance(rec.get("germline_classification"), dict) else ""
            condition = ""
            traits = rec.get("germline_classification", {}).get("trait_set", []) if isinstance(rec.get("germline_classification"), dict) else []
            if isinstance(traits, list) and traits:
                condition = traits[0].get("trait_name", "") if isinstance(traits[0], dict) else ""
            # variation_set carries HGVS strings.
            for var in rec.get("variation_set", []) or []:
                hgvs_p = ""
                hgvs_c = ""
                for h in var.get("variation_loc", []) or []:
                    if h.get("hgvs", "").startswith("NP_"):
                        # protein-level HGVS, like NP_xxx.x:p.Arg175His
                        hgvs_p = h.get("hgvs", "")
                    elif h.get("hgvs", "").startswith("NM_"):
                        hgvs_c = h.get("hgvs", "")
                # Parse aa change from the protein_change field (cleaner than hgvs_p).
                aa_change = var.get("protein_change", "")
                parsed = _parse_aa_change(aa_change) if aa_change else _parse_aa_change(hgvs_p.split(":")[-1] if hgvs_p else "")
                if parsed is None:
                    continue
                wt, pos, mut = parsed
                rows.append(
                    {
                        "gene": gene,
                        "clinvar_id": cv_id,
                        "mutation_notation": f"{wt}{pos}{mut}",
                        "clinical_significance": sig,
                        "review_status": review,
                        "condition": condition,
                        "raw_protein_change": aa_change,
                        "raw_hgvs_p": hgvs_p,
                        "raw_hgvs_c": hgvs_c,
                    }
                )
    df = pd.DataFrame(rows).drop_duplicates(subset=["gene", "mutation_notation"]).reset_index(drop=True)
    return df


def fetch_for_gene_panel(
    genes: Iterable[str],
    config: ClinVarConfig | None = None,
) -> pd.DataFrame:
    """Convenience: fetch pathogenic missense for multiple genes, stack results."""
    out = []
    cfg = config or ClinVarConfig()
    for g in genes:
        try:
            df = fetch_pathogenic_missense(g, cfg)
            if len(df):
                out.append(df)
        except Exception as exc:
            print(f"[clinvar] {g} failed: {exc}")
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)
