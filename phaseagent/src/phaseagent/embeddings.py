"""ESM-2 / SaProt mean-pool embedding featurizers.

Used to plug a frozen PLM as the feature backbone for tiny downstream models
(RandomForest, LightGBM, ridge), following the EVOLVEpro and FLIP2 recipes.
The "tiny model on big embeddings" pattern is alive and winning in 2025/2026
on low-data regimes — RF on ESM-2 mean-pool was the EVOLVEpro winning config
(Jiang et al., *Science* 2025), and FLIP2 (Dallago et al. bioRxiv 2026)
showed simple ridge on one-hot + zero-shot likelihood is competitive with
fine-tuned PLMs on small data.

This module is pure-Python and torch-free at import time. The actual ESM-2
forward pass lives behind a lazy import inside the GPU Modal entrypoint.
A NumPy/test-friendly mock embedder is provided for unit tests so the
downstream regression-baseline code is testable without GPUs.

Embedding dimension reference:
- esm2_t6_8M_UR50D: 320
- esm2_t12_35M_UR50D: 480
- esm2_t30_150M_UR50D: 640
- esm2_t33_650M_UR50D: 1280
- esm2_t36_3B_UR50D: 2560
- saprot_650m: 1280 (uses 3Di-augmented vocabulary; needs structure tokens)
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EmbeddingConfig:
    """Selects which model and pooling strategy to use."""

    model_name: str = "esm2_t33_650M_UR50D"  # ESM-2 650M is FLIP2-validated default.
    pool: str = "mean"  # "mean" | "cls" | "max"
    max_len: int = 1022
    batch_size: int = 8
    cache_dir: str = "data/embeddings"
    # When True, prepend the per-position embedding delta at mutated positions
    # to the mean-pool vector. Captures local context for fitness prediction.
    include_position_delta: bool = False


class Embedder(Protocol):
    """Protocol any embedder must satisfy."""

    embed_dim: int

    def embed(self, sequences: list[str]) -> np.ndarray: ...


class HashEmbedder:
    """Deterministic, dependency-light embedder for tests.

    Maps each sequence to a fixed-dim vector via a hash of the sequence
    bytes. Not semantically meaningful but useful for verifying that the
    downstream regression / dataframe plumbing is correct.
    """

    def __init__(self, embed_dim: int = 32, seed: int = 0):
        self.embed_dim = int(embed_dim)
        self.seed = int(seed)

    def embed(self, sequences: list[str]) -> np.ndarray:
        out = np.zeros((len(sequences), self.embed_dim), dtype=np.float32)
        for i, seq in enumerate(sequences):
            digest = hashlib.sha256(f"{self.seed}:{seq}".encode()).digest()
            # Repeat / truncate the digest to the target dimension.
            buf = (digest * ((self.embed_dim // len(digest)) + 1))[: self.embed_dim]
            out[i] = np.frombuffer(buf, dtype=np.uint8).astype(np.float32) / 255.0
        return out


class ESM2Embedder:
    """Lazy ESM-2 mean-pool embedder. Constructed on first use.

    Imports torch + fair-esm only inside ``_load`` so this module remains
    importable on machines without those packages (e.g., the local laptop
    where we run unit tests).
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._model = None
        self._alphabet = None
        self._embed_dim: int | None = None

    @property
    def embed_dim(self) -> int:
        if self._embed_dim is None:
            self._load()
        assert self._embed_dim is not None
        return int(self._embed_dim)

    def _load(self):
        import esm  # noqa: F401  (heavy import, GPU image only)
        import torch

        fn = getattr(esm.pretrained, self.config.model_name)
        model, alphabet = fn()
        model = model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        self._model = model
        self._alphabet = alphabet
        # Probe embed dim with a dummy forward pass on a length-2 sequence.
        bc = alphabet.get_batch_converter()
        _, _, toks = bc([("probe", "MV")])
        device = next(model.parameters()).device
        with torch.inference_mode():
            out = model(toks.to(device), repr_layers=[model.num_layers])
        rep = out["representations"][model.num_layers]
        self._embed_dim = int(rep.shape[-1])

    def embed(self, sequences: list[str]) -> np.ndarray:
        if self._model is None:
            self._load()
        import torch

        assert self._model is not None and self._alphabet is not None
        bc = self._alphabet.get_batch_converter()
        device = next(self._model.parameters()).device
        max_len = self.config.max_len
        out_list: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(sequences), self.config.batch_size):
                chunk = [s[:max_len] for s in sequences[start : start + self.config.batch_size]]
                batch = [(f"s{i}", s) for i, s in enumerate(chunk)]
                _, _, toks = bc(batch)
                toks = toks.to(device)
                forward = self._model(toks, repr_layers=[self._model.num_layers])
                rep = forward["representations"][self._model.num_layers]
                pad_mask = toks == self._alphabet.padding_idx
                cls_mask = toks == self._alphabet.cls_idx
                eos_mask = toks == self._alphabet.eos_idx
                keep = (~pad_mask) & (~cls_mask) & (~eos_mask)
                if self.config.pool == "cls":
                    pooled = rep[:, 0, :]
                elif self.config.pool == "max":
                    masked = rep.masked_fill(~keep.unsqueeze(-1), float("-inf"))
                    pooled = masked.max(dim=1).values
                else:  # mean
                    summed = (rep * keep.unsqueeze(-1).float()).sum(dim=1)
                    counts = keep.sum(dim=1).clamp(min=1).unsqueeze(-1).float()
                    pooled = summed / counts
                out_list.append(pooled.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(out_list, axis=0)


def embed_dataframe(
    df: pd.DataFrame,
    embedder: Embedder,
    seq_col: str = "mutated_sequence",
) -> np.ndarray:
    """Run ``embedder`` over a dataframe column and return the (n, d) matrix."""
    if seq_col not in df.columns:
        raise ValueError(f"{seq_col!r} not in dataframe columns")
    seqs = df[seq_col].astype(str).tolist()
    if len(seqs) == 0:
        return np.zeros((0, embedder.embed_dim), dtype=np.float32)
    return embedder.embed(seqs)


def make_embedder(config: EmbeddingConfig | None = None) -> Embedder:
    """Build the production ESM-2 embedder. Used inside Modal GPU entrypoints."""
    cfg = config or EmbeddingConfig()
    return ESM2Embedder(cfg)
