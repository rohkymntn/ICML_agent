import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from phaseagent.dms_context_adapter import (
    DMSContextAdapter,
    DMSContextAdapterConfig,
    adapter_loss,
    make_training_frame,
    parse_variant_target,
)
from phaseagent.mutation_map_context import CONTEXT_FEATURE_DIM


def test_parse_variant_target_orders_positions():
    target = parse_variant_target("C2A,A1G", "ACDE")
    assert target is not None
    assert target.positions == (1, 2)
    assert len(target.aa_ids) == 2


def test_make_training_frame_filters_to_multimutants():
    df = pd.DataFrame(
        [
            {"mutation_notation": "A1G", "mutation_distance": 1, "fitness_norm": 0.8},
            {"mutation_notation": "A1G,C2A", "mutation_distance": 2, "fitness_norm": 0.9},
            {"mutation_notation": "A1G,C2A,D3E", "mutation_distance": 3, "fitness_norm": 0.2},
        ]
    )
    out = make_training_frame(df, min_distance=2, max_distance=2)
    assert len(out) == 1
    assert out.iloc[0]["mutation_notation"] == "A1G,C2A"


def test_adapter_forward_shape():
    adapter = DMSContextAdapter(DMSContextAdapterConfig(hidden_dim=8, adapter_dim=16))
    hidden = torch.randn(5, 8)
    ctx = torch.randn(5, CONTEXT_FEATURE_DIM)
    budgets = torch.tensor([2, 2, 3, 3, 5], dtype=torch.float32)
    out = adapter(hidden, ctx, budgets)
    assert out.shape == (5, 20)


def test_adapter_loss_decreases_on_toy_problem():
    torch.manual_seed(0)
    adapter = DMSContextAdapter(DMSContextAdapterConfig(hidden_dim=4, adapter_dim=32, dropout=0.0))
    opt = torch.optim.AdamW(adapter.parameters(), lr=3e-2)
    n = 64
    hidden = torch.randn(n, 4)
    ctx = torch.randn(n, CONTEXT_FEATURE_DIM)
    budgets = torch.ones(n)
    # Deterministic target depends on context so the adapter has a learnable signal.
    target = (ctx[:, 0] > 0).long()
    base = torch.zeros(n, 20)
    weights = torch.ones(n)

    def _loss_value():
        losses = adapter_loss(
            adapter,
            base_aa_logits=base,
            hidden_states=hidden,
            context_features=ctx,
            target_aa_ids=target,
            edit_budgets=budgets,
            token_weights=weights,
            pairwise_weight=0.0,
        )
        return losses["loss"]

    initial = float(_loss_value().detach())
    for _ in range(30):
        opt.zero_grad(set_to_none=True)
        loss = _loss_value()
        loss.backward()
        opt.step()
    final = float(_loss_value().detach())
    assert final < initial * 0.8

