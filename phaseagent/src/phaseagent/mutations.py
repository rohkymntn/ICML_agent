"""Parse mutation notations and compute mutation distance."""
from __future__ import annotations

import re
from typing import List

_TOKEN_SEP = re.compile(r"[,:\s;|]+")
_MUT = re.compile(r"^[A-Za-z\*]\d+[A-Za-z\*]$")


def parse_mutation_notation(s) -> List[str]:
    """Return list of substitution tokens from a mutation notation string.

    Handles separators ',', ':', whitespace, ';', '|'. WT/wildtype/empty -> [].
    """
    if s is None:
        return []
    s = str(s).strip()
    if not s or s.lower() in {"wt", "wildtype", "wild-type", "nan", "none"}:
        return []
    tokens = [t for t in _TOKEN_SEP.split(s) if t]
    return [t for t in tokens if _MUT.match(t)]


def mutation_distance_from_notation(s) -> int:
    """Number of substitution tokens; WT -> 0."""
    return len(parse_mutation_notation(s))


def mutation_distance_from_sequences(wt: str, seq: str) -> int:
    """Hamming distance between WT and mutated sequence (must be equal length)."""
    if len(wt) != len(seq):
        raise ValueError(f"WT length {len(wt)} != mutated length {len(seq)}")
    return sum(1 for a, b in zip(wt, seq) if a != b)


def apply_substitutions(wt: str, mutation_notation: str) -> str:
    """Construct mutated sequence by applying substitution tokens to WT."""
    if mutation_notation is None:
        return wt
    s = str(mutation_notation).strip()
    if not s or s.lower() in {"wt", "wildtype", "wild-type", "nan", "none"}:
        return wt
    out = list(wt)
    for tok in parse_mutation_notation(s):
        ref, pos, alt = tok[0], int(tok[1:-1]), tok[-1]
        if 1 <= pos <= len(wt) and out[pos - 1] == ref:
            out[pos - 1] = alt
            continue
        if 0 <= pos < len(wt) and out[pos] == ref:
            out[pos] = alt
            continue
        raise ValueError(
            f"Mismatch applying {tok!r}: position {pos} in WT length {len(wt)}"
        )
    return "".join(out)
