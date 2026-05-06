"""WT-derived edit proposal generators for EditGuard.

These generators create new candidates from a wild-type sequence instead of
selecting from a fixed measured DMS pool. DMS labels are joined only afterward
for evaluation when a generated mutation happens to be measured.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from .editing_tasks import EditingTask, constraint_satisfied, objective_score
from .editguard_prior import DMSFunctionPrior
from .mutations import apply_substitutions, parse_mutation_notation
from .spectrum import AMINO_ACIDS


@dataclass(frozen=True)
class ProposalConfig:
    n_candidates: int = 500
    max_budget: int | None = None
    proposal_batch: int = 64
    guidance_temperature: float = 0.25
    seed: int = 0
    coverage_mode: str = "dms_evaluable"


def infer_wildtype_sequence(df: pd.DataFrame, dataset_id: str) -> str | None:
    """Infer a dataset WT sequence from canonical ProteinGym columns."""
    sub = df[df["dataset_id"].astype(str) == str(dataset_id)] if "dataset_id" in df.columns else df
    if "wildtype_sequence" in sub.columns:
        vals = sub["wildtype_sequence"].dropna().astype(str)
        vals = vals[vals.str.len() > 0]
        vals = vals[~vals.str.lower().isin({"none", "nan"})]
        if len(vals):
            return vals.iloc[0]
    wt_rows = sub[sub.get("mutation_distance", pd.Series(dtype=int)) == 0]
    if len(wt_rows) and "mutated_sequence" in wt_rows.columns:
        vals = wt_rows["mutated_sequence"].dropna().astype(str)
        vals = vals[vals.str.len() > 0]
        if len(vals):
            return vals.iloc[0]
    if {"mutated_sequence", "mutation_notation"}.issubset(sub.columns):
        for _, row in sub.dropna(subset=["mutated_sequence", "mutation_notation"]).iterrows():
            seq = str(row["mutated_sequence"])
            if not seq or seq.lower() in {"nan", "none"}:
                continue
            toks = parse_mutation_notation(row["mutation_notation"])
            if not toks:
                continue
            chars = list(seq)
            ok = True
            for tok in toks:
                ref, pos, alt = tok[0].upper(), int(tok[1:-1]), tok[-1].upper()
                if not (1 <= pos <= len(chars)) or chars[pos - 1].upper() != alt:
                    ok = False
                    break
                chars[pos - 1] = ref
            if ok:
                return "".join(chars)
    return None


def _parse_used_positions(mutation_notation: str) -> set[int]:
    out = set()
    for tok in parse_mutation_notation(mutation_notation):
        try:
            out.add(int(tok[1:-1]))
        except ValueError:
            continue
    return out


def _append_token(mutation_notation: str, token: str) -> str:
    toks = parse_mutation_notation(mutation_notation)
    toks.append(token)
    return ",".join(toks)


def _max_unique_from_tokens(n_tokens: int, max_budget: int) -> int:
    total = 0
    for k in range(1, min(n_tokens, max_budget) + 1):
        total += math.comb(n_tokens, k)
    return int(total)


def observed_single_mutation_tokens(dms_df: pd.DataFrame, task: EditingTask) -> tuple[str, ...]:
    """Return task-compatible single-mutant tokens observed in the assay."""
    sub = dms_df[dms_df["dataset_id"].astype(str) == task.dataset_id].copy()
    if len(sub) == 0:
        return ()
    protected = set(task.protected_positions)
    forbidden = set(a.upper() for a in task.forbidden_residues)
    tokens = []
    for notation in sub["mutation_notation"].astype(str):
        toks = parse_mutation_notation(notation)
        if len(toks) != 1:
            continue
        tok = toks[0]
        try:
            pos = int(tok[1:-1])
        except ValueError:
            continue
        if pos in protected or tok[-1].upper() in forbidden:
            continue
        tokens.append(tok)
    return tuple(sorted(set(tokens)))


def token_effect_table(dms_df: pd.DataFrame, task: EditingTask) -> pd.DataFrame:
    """Build task-compatible single-token table with observed DMS effects."""
    sub = dms_df[dms_df["dataset_id"].astype(str) == task.dataset_id].copy()
    rows = []
    for _, row in sub.iterrows():
        toks = parse_mutation_notation(row.get("mutation_notation", ""))
        if len(toks) != 1:
            continue
        tok = toks[0]
        try:
            pos = int(tok[1:-1])
        except ValueError:
            continue
        if pos in set(task.protected_positions) or tok[-1].upper() in set(task.forbidden_residues):
            continue
        rows.append(
            {
                "token": tok,
                "position": pos,
                "wt_aa": tok[0].upper(),
                "mut_aa": tok[-1].upper(),
                "fitness_norm": row.get("fitness_norm", np.nan),
                "viable": row.get("viable", np.nan),
            }
        )
    return pd.DataFrame(rows).drop_duplicates("token")


def _candidate_row(dataset_id: str, wt_sequence: str, mutation_notation: str, source: str, generated_rank: int) -> dict:
    mutated = apply_substitutions(wt_sequence, mutation_notation)
    distance = len(parse_mutation_notation(mutation_notation))
    return {
        "dataset_id": dataset_id,
        "variant_id": mutation_notation or "WT",
        "mutation_notation": mutation_notation or "WT",
        "mutation_distance": distance,
        "wildtype_sequence": wt_sequence,
        "mutated_sequence": mutated,
        "proposal_source": source,
        "generated_rank": int(generated_rank),
    }


def sample_random_edit_candidates(
    dataset_id: str,
    wt_sequence: str,
    task: EditingTask,
    config: ProposalConfig | None = None,
    source: str = "random_proposal",
    allowed_tokens: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Generate random substitution candidates satisfying hard task constraints."""
    config = config or ProposalConfig()
    rng = np.random.default_rng(config.seed)
    max_budget = int(config.max_budget or task.edit_budget)
    max_budget = max(1, min(max_budget, task.edit_budget, len(wt_sequence)))
    protected = set(task.protected_positions)
    forbidden = set(a.upper() for a in task.forbidden_residues)
    editable_positions = [i for i in range(1, len(wt_sequence) + 1) if i not in protected]
    if not editable_positions:
        return pd.DataFrame()

    seen, rows, attempts = set(), [], 0
    target_n = config.n_candidates
    if allowed_tokens:
        target_n = min(target_n, _max_unique_from_tokens(len(allowed_tokens), max_budget))
    max_attempts = max(target_n * 50, 1000)
    while len(rows) < target_n and attempts < max_attempts:
        attempts += 1
        budget = int(rng.integers(1, max_budget + 1))
        if allowed_tokens:
            eligible = [
                tok for tok in allowed_tokens
                if int(tok[1:-1]) in editable_positions and tok[-1].upper() not in forbidden
            ]
            if not eligible:
                break
            chosen = rng.choice(eligible, size=min(budget, len(eligible)), replace=False)
            toks = sorted(map(str, chosen), key=lambda tok: int(tok[1:-1]))
            if len({int(tok[1:-1]) for tok in toks}) != len(toks):
                continue
        else:
            positions = rng.choice(editable_positions, size=min(budget, len(editable_positions)), replace=False)
            toks = []
            for pos in sorted(int(p) for p in positions):
                wt_aa = wt_sequence[pos - 1].upper()
                aas = [aa for aa in AMINO_ACIDS if aa != wt_aa and aa not in forbidden]
                if not aas:
                    continue
                alt = str(rng.choice(aas))
                toks.append(f"{wt_aa}{pos}{alt}")
        if not toks:
            continue
        notation = ",".join(toks)
        if notation in seen:
            continue
        seen.add(notation)
        rows.append(_candidate_row(dataset_id, wt_sequence, notation, source, len(rows)))
    return pd.DataFrame(rows)


def _proposal_extensions(
    dataset_id: str,
    wt_sequence: str,
    task: EditingTask,
    current_notation: str,
    rng: np.random.Generator,
    n: int,
    allowed_tokens: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    used = _parse_used_positions(current_notation)
    protected = set(task.protected_positions)
    forbidden = set(a.upper() for a in task.forbidden_residues)
    editable = [i for i in range(1, len(wt_sequence) + 1) if i not in protected and i not in used]
    if not editable:
        return pd.DataFrame()
    rows, seen, attempts = [], set(), 0
    while len(rows) < n and attempts < n * 20:
        attempts += 1
        if allowed_tokens:
            eligible = [
                tok for tok in allowed_tokens
                if int(tok[1:-1]) in editable and tok[-1].upper() not in forbidden
            ]
            if not eligible:
                break
            token = str(rng.choice(eligible))
        else:
            pos = int(rng.choice(editable))
            wt_aa = wt_sequence[pos - 1].upper()
            aas = [aa for aa in AMINO_ACIDS if aa != wt_aa and aa not in forbidden]
            if not aas:
                continue
            token = f"{wt_aa}{pos}{str(rng.choice(aas))}"
        notation = _append_token(current_notation, token)
        if notation in seen:
            continue
        seen.add(notation)
        rows.append(_candidate_row(dataset_id, wt_sequence, notation, "guided_local_step", len(rows)))
    return pd.DataFrame(rows)


def guided_local_generation(
    dataset_id: str,
    wt_sequence: str,
    task: EditingTask,
    prior: DMSFunctionPrior,
    config: ProposalConfig | None = None,
    allowed_tokens: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Generate edits by applying prior guidance at each mutation step."""
    config = config or ProposalConfig()
    rng = np.random.default_rng(config.seed)
    if allowed_tokens:
        pool = sample_random_edit_candidates(
            dataset_id,
            wt_sequence,
            task,
            ProposalConfig(
                n_candidates=min(
                    max(config.n_candidates * 20, config.n_candidates),
                    _max_unique_from_tokens(len(allowed_tokens), max(1, int(config.max_budget or task.edit_budget))),
                ),
                max_budget=config.max_budget,
                seed=config.seed,
                coverage_mode=config.coverage_mode,
            ),
            source="guided_pool_generation",
            allowed_tokens=allowed_tokens,
        )
        if len(pool) == 0:
            return pool
        p = prior.predict_proba(pool)
        obj = objective_score(pool, task.objective)
        ok = constraint_satisfied(pool, task).astype(float)
        pool["guided_generation_score"] = np.log(np.clip(p, 1e-6, 1.0)) + 0.5 * obj + 2.0 * ok
        out = pool.nlargest(min(config.n_candidates, len(pool)), "guided_generation_score").copy()
        out["proposal_source"] = "guided_local_generation"
        out["generated_rank"] = np.arange(len(out))
        return out.reset_index(drop=True)
    rows, seen = [], set()
    max_budget = int(config.max_budget or task.edit_budget)
    max_budget = max(1, min(max_budget, task.edit_budget, len(wt_sequence)))

    attempts = 0
    max_attempts = max(config.n_candidates * 50, 1000)
    while len(rows) < config.n_candidates and attempts < max_attempts:
        attempts += 1
        current = ""
        for _step in range(max_budget):
            proposals = _proposal_extensions(
                dataset_id,
                wt_sequence,
                task,
                current,
                rng,
                n=config.proposal_batch,
                allowed_tokens=allowed_tokens,
            )
            if len(proposals) == 0:
                break
            p = prior.predict_proba(proposals)
            obj = objective_score(proposals, task.objective)
            ok = constraint_satisfied(proposals, task).astype(float)
            logits = np.log(np.clip(p, 1e-6, 1.0)) + 0.5 * obj + 2.0 * ok
            logits = logits / max(config.guidance_temperature, 1e-6)
            logits = logits - float(np.max(logits))
            probs = np.exp(logits)
            probs = probs / probs.sum()
            idx = int(rng.choice(np.arange(len(proposals)), p=probs))
            current = str(proposals.iloc[idx]["mutation_notation"])
        if current and current not in seen:
            seen.add(current)
            rows.append(_candidate_row(dataset_id, wt_sequence, current, "guided_local_generation", len(rows)))
        if len(seen) >= config.n_candidates:
            break
    return pd.DataFrame(rows)


def join_generated_to_dms(generated: pd.DataFrame, dms_df: pd.DataFrame) -> pd.DataFrame:
    """Attach measured DMS labels to generated candidates when available."""
    label_cols = [
        "dataset_id",
        "mutation_notation",
        "fitness_norm",
        "viable",
        "fitness_raw",
    ]
    labels = dms_df[[c for c in label_cols if c in dms_df.columns]].copy()
    labels = labels.drop_duplicates(["dataset_id", "mutation_notation"])
    out = generated.merge(labels, on=["dataset_id", "mutation_notation"], how="left", indicator="dms_label_status")
    out["has_dms_label"] = out["dms_label_status"].eq("both")
    out = out.drop(columns=["dms_label_status"])
    return out


def generate_many_then_rerank(
    dataset_id: str,
    wt_sequence: str,
    task: EditingTask,
    prior: DMSFunctionPrior,
    n_generate: int,
    k: int,
    seed: int = 0,
    allowed_tokens: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Generate many random edits, then select by DMS prior."""
    proposals = sample_random_edit_candidates(
        dataset_id,
        wt_sequence,
        task,
        ProposalConfig(n_candidates=n_generate, seed=seed),
        source="generate_many_random",
        allowed_tokens=allowed_tokens,
    )
    if len(proposals) == 0:
        return proposals
    proposals["prior_function_prob"] = prior.predict_proba(proposals)
    proposals["prior_fitness"] = prior.predict_fitness(proposals)
    out = proposals.nlargest(min(k, len(proposals)), "prior_function_prob").copy()
    out["method"] = f"generate_{n_generate}_then_rerank"
    return out.reset_index(drop=True)


def rerank_by_column(
    candidates: pd.DataFrame,
    score_col: str,
    k: int,
    method_name: str,
) -> pd.DataFrame:
    """Select top candidates by an existing score column."""
    if len(candidates) == 0 or score_col not in candidates.columns:
        return candidates.iloc[:0].copy()
    out = candidates.dropna(subset=[score_col]).nlargest(min(k, len(candidates)), score_col).copy()
    out["method"] = method_name
    return out.reset_index(drop=True)


def aa_frequency_proposal_baseline(
    dataset_id: str,
    wt_sequence: str,
    task: EditingTask,
    n_candidates: int,
    seed: int = 0,
    allowed_tokens: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Static amino-acid frequency baseline.

    Scores random edits by ``log P_swissprot(mutant_aa)`` using a hard-coded
    SwissProt-derived AA frequency table. This is *not* a PLM and should not
    be presented as one — for a real ESM-2 masked-marginal baseline, use
    ``plm.masked_token_log_probs`` (loads the ESM-2 checkpoint on GPU).

    The output method label is ``aa_frequency_proposal``.
    """
    aa_freq = {
        "A": 0.0825, "R": 0.0553, "N": 0.0406, "D": 0.0545, "C": 0.0137,
        "Q": 0.0393, "E": 0.0675, "G": 0.0707, "H": 0.0227, "I": 0.0596,
        "L": 0.0966, "K": 0.0584, "M": 0.0242, "F": 0.0386, "P": 0.0470,
        "S": 0.0656, "T": 0.0534, "W": 0.0108, "Y": 0.0292, "V": 0.0687,
    }
    proposals = sample_random_edit_candidates(
        dataset_id,
        wt_sequence,
        task,
        ProposalConfig(n_candidates=max(n_candidates * 5, n_candidates), seed=seed),
        source="aa_frequency_proposal",
        allowed_tokens=allowed_tokens,
    )
    if len(proposals) == 0:
        return proposals
    scores = []
    for notation in proposals["mutation_notation"].astype(str):
        toks = parse_mutation_notation(notation)
        if not toks:
            scores.append(0.0)
        else:
            scores.append(float(np.mean([np.log(aa_freq.get(tok[-1].upper(), 1e-4)) for tok in toks])))
    proposals["aa_freq_score"] = scores
    proposals = proposals.nlargest(min(n_candidates, len(proposals)), "aa_freq_score").copy()
    proposals["method"] = "aa_frequency_proposal"
    return proposals.reset_index(drop=True)


# Backwards-compatibility alias for any external callers.
plm_masked_proposal_generation = aa_frequency_proposal_baseline


class DPLMInfillingAdapter:
    """Interface placeholder for a public DPLM masked-infilling backbone."""

    def __init__(self, model=None):
        self.model = model

    @property
    def available(self) -> bool:
        return self.model is not None

    def sample(self, *args, **kwargs) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError(
                "DPLM adapter requires a loaded public DPLM model with masked "
                "infilling and fixed-position control hooks."
            )
        raise NotImplementedError("Wire DPLM sampling hooks here once checkpoint API is installed.")
