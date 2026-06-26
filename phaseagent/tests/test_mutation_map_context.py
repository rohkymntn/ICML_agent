import numpy as np
import pandas as pd
import pytest

from phaseagent.editing_tasks import EditingTask
from phaseagent.context_generation import measured_pool_additive_oracle
from phaseagent.mutation_map_context import CONTEXT_FEATURE_DIM, build_mutation_map_context


def _toy_dms():
    wt = "ACDE"
    rows = [
        {"dataset_id": "toy", "mutation_notation": "A1G", "mutation_distance": 1, "fitness_norm": 0.9, "wildtype_sequence": wt},
        {"dataset_id": "toy", "mutation_notation": "A1V", "mutation_distance": 1, "fitness_norm": 0.4, "wildtype_sequence": wt},
        {"dataset_id": "toy", "mutation_notation": "C2A", "mutation_distance": 1, "fitness_norm": 0.8, "wildtype_sequence": wt},
        {"dataset_id": "toy", "mutation_notation": "D3W", "mutation_distance": 1, "fitness_norm": 0.1, "wildtype_sequence": wt},
        {"dataset_id": "toy", "mutation_notation": "A1G,C2A", "mutation_distance": 2, "fitness_norm": 0.95, "viable": 1, "wildtype_sequence": wt},
        {"dataset_id": "toy", "mutation_notation": "A1V,D3W", "mutation_distance": 2, "fitness_norm": 0.05, "viable": 0, "wildtype_sequence": wt},
    ]
    return pd.DataFrame(rows)


def test_mutation_map_context_shape_and_missing_values():
    df = _toy_dms()
    singles = df[df["mutation_distance"] == 1]
    ctx = build_mutation_map_context(singles, dataset_id="toy", strict_singles=True)
    assert ctx.score_matrix.shape == (4, 20)
    assert ctx.observed_mask.shape == (4, 20)
    assert ctx.features_for_positions([1, 2]).shape == (2, CONTEXT_FEATURE_DIM)
    assert np.isfinite(ctx.score_matrix).all()
    assert ctx.stats.n_observed == 4
    assert 0 < ctx.stats.coverage < 1


def test_mutation_map_context_rejects_multimutant_leakage():
    with pytest.raises(ValueError, match="multi-mutant"):
        build_mutation_map_context(_toy_dms(), dataset_id="toy", strict_singles=True)


def test_additive_oracle_respects_task_constraints():
    df = _toy_dms()
    ctx = build_mutation_map_context(df[df["mutation_distance"] == 1], dataset_id="toy")
    task = EditingTask(
        dataset_id="toy",
        objective="novelty",
        edit_budget=2,
        protected_positions=(3,),
        forbidden_residues=("W",),
    )
    out = measured_pool_additive_oracle(df, task, ctx, k=10)
    assert len(out) == 1
    assert out.iloc[0]["mutation_notation"] == "A1G,C2A"
    assert out.iloc[0]["method"] == "measured_pool_additive_oracle"

