import numpy as np
import pandas as pd
import pytest

from phaseagent.edit_baselines import run_edit_baselines
from phaseagent.edit_eval import evaluate_methods
from phaseagent.edit_generators import (
    ProposalConfig,
    generate_many_then_rerank,
    guided_local_generation,
    infer_wildtype_sequence,
    join_generated_to_dms,
    observed_single_mutation_tokens,
    plm_masked_proposal_generation,
)
from phaseagent.edit_prompts import generate_prompted_edits, parse_edit_prompt
from phaseagent.edit_splits import (
    VALID_SPLITS,
    assert_tasks_in_split,
    filter_by_split,
    make_dataset_splits,
)
from phaseagent.edit_sota import baseline_registry_frame
from phaseagent.edit_structure_sanity import mutation_position_table, structure_sanity_stub
from phaseagent.editing_tasks import EditingTask, build_editing_tasks, candidate_pool_for_task
from phaseagent.editguard_diffusion import DiffusionSampleConfig, MeasuredPoolEditDiffusion
from phaseagent.editguard_prior import DMSFunctionPrior, FEATURE_DIM, evaluate_prior, featurize_variants


def _toy_df():
    rows = []
    for ds in ["BRCA1_HUMAN_Test", "GFP_TEST"]:
        for i in range(1, 8):
            for aa, fit in [("A", 0.9), ("D", 0.2), ("G", 0.7)]:
                rows.append(
                    {
                        "dataset_id": ds,
                        "variant_id": f"M{i}{aa}",
                        "mutation_notation": f"M{i}{aa}",
                        "mutation_distance": 1,
                        "wildtype_sequence": "M" * 10,
                        "mutated_sequence": "M" * (i - 1) + aa + "M" * (10 - i),
                        "fitness_norm": fit,
                        "viable": int(fit >= 0.5),
                    }
                )
        rows.append(
            {
                "dataset_id": ds,
                "variant_id": "M1A,M2A",
                "mutation_notation": "M1A,M2A",
                "mutation_distance": 2,
                "wildtype_sequence": "M" * 10,
                "mutated_sequence": "AAMMMMMMMM",
                "fitness_norm": 0.8,
                "viable": 1,
            }
        )
    return pd.DataFrame(rows)


def test_editing_tasks_and_splits():
    df = _toy_df()
    tasks = build_editing_tasks(df, budgets=[1, 2], objectives=["novelty"], min_candidates=3)
    assert len(tasks) >= 2
    splits = make_dataset_splits(df["dataset_id"].unique(), clinical_holdout=True)
    assert "clinical_test" in set(splits["split"])


def test_prior_baselines_and_diffusion_surrogate():
    df = _toy_df()
    train = df[df["dataset_id"] == "GFP_TEST"]
    test = df[df["dataset_id"] == "BRCA1_HUMAN_Test"]
    prior = DMSFunctionPrior(n_estimators=5).fit(train)
    metrics = evaluate_prior(prior, test)
    assert metrics["n"] == len(test)

    task = EditingTask(dataset_id="BRCA1_HUMAN_Test", objective="novelty", edit_budget=2)
    pool = candidate_pool_for_task(df, task)
    selections = run_edit_baselines(pool, task, prior, k=5, seed=0)
    result = evaluate_methods(selections, task)
    assert set(result["method"]) == set(selections)

    sampler = MeasuredPoolEditDiffusion(prior)
    sampled = sampler.sample(pool, task, DiffusionSampleConfig(n_samples=5, seed=0))
    assert len(sampled) == 5
    assert "editguard_energy" in sampled.columns


def test_guided_generation_from_wildtype():
    df = _toy_df()
    train = df[df["dataset_id"] == "GFP_TEST"]
    prior = DMSFunctionPrior(n_estimators=5).fit(train)
    task = EditingTask(dataset_id="BRCA1_HUMAN_Test", objective="novelty", edit_budget=1)
    wt = infer_wildtype_sequence(df, task.dataset_id)
    assert wt == "M" * 10
    allowed = observed_single_mutation_tokens(df, task)
    assert len(allowed) > 0

    guided = guided_local_generation(
        task.dataset_id,
        wt,
        task,
        prior,
        ProposalConfig(n_candidates=5, seed=0),
        allowed_tokens=allowed,
    )
    assert len(guided) == 5
    labeled = join_generated_to_dms(guided, df)
    assert "has_dms_label" in labeled.columns

    reranked = generate_many_then_rerank(
        task.dataset_id,
        wt,
        task,
        prior,
        n_generate=20,
        k=5,
        seed=0,
        allowed_tokens=allowed,
    )
    assert len(reranked) == 5
    assert "prior_function_prob" in reranked.columns

    aa_freq = plm_masked_proposal_generation(task.dataset_id, wt, task, n_candidates=5, seed=0, allowed_tokens=allowed)
    assert len(aa_freq) == 5
    # Output is the AA-frequency baseline, not a real PLM. Method label and
    # score column reflect that honestly.
    assert "aa_freq_score" in aa_freq.columns
    assert (aa_freq["method"] == "aa_frequency_proposal").all()


def test_prompted_interface_and_registry():
    df = _toy_df()
    prior = DMSFunctionPrior(n_estimators=5).fit(df[df["dataset_id"] == "GFP_TEST"])
    request = parse_edit_prompt(
        "BRCA1_HUMAN_Test",
        "Remove cysteine liabilities with 1 mutation while preserving function.",
        n_candidates=4,
    )
    assert request.objective == "motif_avoidance"
    assert request.edit_budget == 1
    generated = generate_prompted_edits(df, prior, request, seed=0)
    assert len(generated) <= 4
    assert "prior_function_prob" in generated.columns

    registry = baseline_registry_frame()
    assert {"name", "tier", "status"}.issubset(registry.columns)
    assert "guided_local_generation" in set(registry["name"])


def test_featurize_variants_has_no_variant_identifier():
    """Two variants with identical chemistry/positions must produce identical
    feature vectors regardless of dataset_id or token order. Guards against
    the notation/dataset-id hash features that previously enabled
    train-set memorization."""
    df = pd.DataFrame([
        {"dataset_id": "DS_A", "mutation_notation": "M1A,L3V", "mutation_distance": 2,
         "wildtype_sequence": "MXLY", "fitness_norm": 0.7, "viable": 1},
        {"dataset_id": "DS_B", "mutation_notation": "M1A,L3V", "mutation_distance": 2,
         "wildtype_sequence": "MXLY", "fitness_norm": 0.7, "viable": 1},
        {"dataset_id": "DS_A", "mutation_notation": "L3V,M1A", "mutation_distance": 2,
         "wildtype_sequence": "MXLY", "fitness_norm": 0.7, "viable": 1},
    ])
    X = featurize_variants(df)
    assert X.shape == (3, FEATURE_DIM)
    assert np.allclose(X[0], X[1]), "different dataset_ids must not change features"
    assert np.allclose(X[0], X[2]), "token reorder must not change features"


def test_split_helpers_assert_split_membership():
    splits = pd.DataFrame({
        "dataset_id": ["A", "B", "C"],
        "split": ["train", "val", "test"],
    })
    test_tasks = pd.DataFrame({"dataset_id": ["C"], "objective": ["novelty"], "edit_budget": [1]})
    assert_tasks_in_split(test_tasks, splits, "test")  # passes
    mixed = pd.DataFrame({"dataset_id": ["A", "C"], "objective": ["novelty"] * 2, "edit_budget": [1, 1]})
    with pytest.raises(ValueError, match="not in split"):
        assert_tasks_in_split(mixed, splits, "test")
    sub = filter_by_split(pd.DataFrame({"dataset_id": ["A", "B", "C"], "x": [1, 2, 3]}), splits, "val")
    assert list(sub["dataset_id"]) == ["B"]


def test_structure_sanity_report_helpers():
    df = _toy_df().head(3)
    report = structure_sanity_stub(df)
    assert "frac_hydrophobic" in report.columns
    assert "esmfold_plddt" in report.columns
    muts = mutation_position_table(df)
    assert {"position", "wt_aa", "mut_aa"}.issubset(muts.columns)
