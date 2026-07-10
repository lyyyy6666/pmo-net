from __future__ import annotations

import torch
import torch.nn as nn


class AdaptiveGating(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_nodes: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_nodes = num_nodes
        self.gating_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[0]
        z_nodes = z.reshape(batch_size * self.num_nodes, self.latent_dim)
        alpha_nodes = self.gating_net(z_nodes)
        alpha_nodes = alpha_nodes.expand(-1, self.latent_dim)
        return alpha_nodes.reshape(batch_size, self.num_nodes * self.latent_dim)


