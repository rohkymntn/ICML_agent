"""Tests for the embedding-based baselines that use the HashEmbedder."""
import numpy as np
import pandas as pd
import pytest

from phaseagent.embedding_baselines import (
    OneHotZeroShotRidgeBaseline,
    RFEmbeddingBaseline,
)
from phaseagent.embeddings import HashEmbedder, embed_dataframe


def _toy_train_test(n: int = 200, seed: int = 0):
    """Toy DMS-like frame with deterministic per-sequence target."""
    rng = np.random.default_rng(seed)
    seqs = ["M" * 60 for _ in range(n)]
    # Inject differentiation by appending a tag the embedder can learn.
    seqs = [s + "ACDE"[i % 4] for i, s in enumerate(seqs)]
    targets = np.array([0.1 if i % 4 == 0 else 0.9 for i in range(n)], dtype=float)
    notation = ["M1A"] * n
    rows = pd.DataFrame(
        {
            "dataset_id": ["P1"] * (n // 2) + ["P2"] * (n - n // 2),
            "mutation_notation": notation,
            "mutated_sequence": seqs,
            "fitness_norm": targets + rng.normal(0, 0.02, n),
        }
    )
    train = rows.iloc[: int(n * 0.7)].reset_index(drop=True)
    test = rows.iloc[int(n * 0.7) :].reset_index(drop=True)
    return train, test


def test_hash_embedder_returns_correct_shape():
    emb = HashEmbedder(embed_dim=64, seed=0)
    out = emb.embed(["MAMA", "ACDC", "GGGG"])
    assert out.shape == (3, 64)


def test_hash_embedder_is_deterministic():
    a = HashEmbedder(embed_dim=32, seed=0).embed(["MAMA"])
    b = HashEmbedder(embed_dim=32, seed=0).embed(["MAMA"])
    np.testing.assert_array_equal(a, b)


def test_hash_embedder_different_seqs_give_different_vectors():
    emb = HashEmbedder(embed_dim=64, seed=0)
    out = emb.embed(["A", "B"])
    assert not np.allclose(out[0], out[1])


def test_embed_dataframe_returns_one_row_per_input():
    df = pd.DataFrame({"mutated_sequence": ["MAMA", "ACDC"]})
    emb = HashEmbedder(embed_dim=16)
    X = embed_dataframe(df, emb)
    assert X.shape == (2, 16)


def test_rf_embedding_baseline_fits_and_predicts():
    train, test = _toy_train_test(n=200)
    emb = HashEmbedder(embed_dim=64, seed=0)
    train_X = embed_dataframe(train, emb)
    test_X = embed_dataframe(test, emb)
    model = RFEmbeddingBaseline(embedder=emb).fit(train, precomputed_embeddings=train_X)
    preds = model.predict_fitness(test, precomputed_embeddings=test_X)
    assert preds.shape == (len(test),)
    assert preds.min() >= 0.0 and preds.max() <= 1.0


def test_rf_embedding_predict_alias():
    train, test = _toy_train_test(n=80)
    emb = HashEmbedder(embed_dim=32, seed=0)
    train_X = embed_dataframe(train, emb)
    model = RFEmbeddingBaseline(embedder=emb).fit(train, precomputed_embeddings=train_X)
    a = model.predict(test, precomputed_embeddings=embed_dataframe(test, emb))
    b = model.predict_fitness(test, precomputed_embeddings=embed_dataframe(test, emb))
    np.testing.assert_array_equal(a, b)


def test_rf_embedding_unfitted_raises():
    df = pd.DataFrame({"mutated_sequence": ["MAMA"], "fitness_norm": [0.5]})
    emb = HashEmbedder(embed_dim=32)
    with pytest.raises(ValueError, match="not fitted"):
        RFEmbeddingBaseline(embedder=emb).predict_fitness(df, precomputed_embeddings=embed_dataframe(df, emb))


def test_onehot_zeroshot_ridge_baseline_handles_missing_zero_shot_col():
    train, test = _toy_train_test(n=80)
    # Drop the zero-shot column entirely.
    train = train.copy()
    test = test.copy()
    model = OneHotZeroShotRidgeBaseline(zero_shot_col="not_a_real_column", max_positions=8).fit(train)
    preds = model.predict_fitness(test)
    assert preds.shape == (len(test),)
    assert preds.min() >= 0.0 and preds.max() <= 1.0


def test_onehot_zeroshot_ridge_baseline_uses_zero_shot_col_when_present():
    train, test = _toy_train_test(n=80)
    train = train.copy()
    test = test.copy()
    train["esm2_log_likelihood"] = train["fitness_norm"] * 2 - 1  # fake correlation with target
    test["esm2_log_likelihood"] = test["fitness_norm"] * 2 - 1
    model = OneHotZeroShotRidgeBaseline(zero_shot_col="esm2_log_likelihood", max_positions=8).fit(train)
    preds = model.predict_fitness(test)
    # Should track the test fitness more closely than chance.
    from scipy.stats import spearmanr

    rho = spearmanr(test["fitness_norm"], preds, nan_policy="omit").statistic
    assert rho > 0.5
