from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PiZeroConfig:
    state_dim: int = 7
    action_dim: int = 7
    hidden_dim: int = 512
    n_layers: int = 6
    dropout: float = 0.1

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class PiZeroPolicy(nn.Module):
    """A compact pi0-style residual MLP for state->action behavior cloning."""

    def __init__(self, config: PiZeroConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.state_dim, config.hidden_dim)
        self.blocks = nn.Sequential(
            *[ResidualBlock(config.hidden_dim, config.dropout) for _ in range(config.n_layers)]
        )
        self.head = nn.Linear(config.hidden_dim, config.action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(state)
        x = self.blocks(x)
        return self.head(x)


def normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (x - mean) / (std + eps)


def denormalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * (std + eps) + mean
