"""Optional structure-aware models.

The frontier version can be replaced with e3nn/torch-geometric. This file keeps
the minimum viable invariant GNN dependency-light and optional.
"""
from __future__ import annotations

try:
    import torch
    from torch import nn

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    nn = object
    _HAS_TORCH = False


class _TorchRequired:
    def __init__(self, *args, **kwargs):
        raise ImportError("PyTorch is required for structure-aware neural models.")


if _HAS_TORCH:

    class InvariantContactGNN(nn.Module):
        """Small distance-gated message passing network over residue contacts."""

        def __init__(self, node_dim: int, hidden_dim: int = 128, n_layers: int = 3):
            super().__init__()
            self.node_proj = nn.Linear(node_dim, hidden_dim)
            self.message = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(2 * hidden_dim + 1, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                    for _ in range(n_layers)
                ]
            )
            self.update = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(2 * hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                    for _ in range(n_layers)
                ]
            )

        def forward(self, node_features, edge_index, edge_distance):
            h = self.node_proj(node_features)
            src, dst = edge_index
            for msg, upd in zip(self.message, self.update):
                m_in = torch.cat([h[src], h[dst], edge_distance.unsqueeze(-1)], dim=-1)
                m = msg(m_in)
                agg = torch.zeros_like(h)
                agg.index_add_(0, dst, m)
                h = h + upd(torch.cat([h, agg], dim=-1))
            return h


    class StructureResidualHead(nn.Module):
        """Pool residue embeddings and predict a survival residual curve."""

        def __init__(self, node_dim: int, depth_count: int, hidden_dim: int = 128):
            super().__init__()
            self.gnn = InvariantContactGNN(node_dim=node_dim, hidden_dim=hidden_dim)
            self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, depth_count))

        def forward(self, node_features, edge_index, edge_distance):
            h = self.gnn(node_features, edge_index, edge_distance)
            return self.head(h.mean(dim=0, keepdim=True))

else:
    InvariantContactGNN = _TorchRequired
    StructureResidualHead = _TorchRequired
