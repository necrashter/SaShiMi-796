import torch
from torch import nn
from S4 import *


def S4Block(signal_dim: int, state_dim: int, sequence_length: int, expansion_factor: int = 2):
    return Sequential(
        Residual(
            nn.LayerNorm(signal_dim),
            S4Base(signal_dim, state_dim, sequence_length),
            nn.GELU(),
            nn.Linear(signal_dim, signal_dim),
        ),
        Residual(
            nn.LayerNorm(signal_dim),
            nn.Linear(signal_dim, signal_dim * expansion_factor),
            nn.GELU(),
            nn.Linear(signal_dim * expansion_factor, signal_dim),
        ),
    )


class DownPool(nn.Module):
    def __init__(self, signal_dim: int, pooling_factor: int = 4, expansion_factor: int = 2):
        super().__init__()
        self.pooling_factor = pooling_factor
        self.linear = nn.Linear(
            signal_dim * pooling_factor,
            signal_dim * expansion_factor,
        )

    def forward(self, x):
        T = x.size(dim=-2)
        H = x.size(dim=-1)
        x = x.reshape(-1, T // self.pooling_factor, H * self.pooling_factor)
        return self.linear(x)


class UpPool(nn.Module):
    def __init__(self, signal_dim: int, pooling_factor: int = 4, expansion_factor: int = 2):
        super().__init__()
        self.pooling_factor = pooling_factor
        self.linear = nn.Linear(
            signal_dim * expansion_factor,
            signal_dim * pooling_factor,
        )

    def forward(self, x):
        # Use shifting
        pad = torch.zeros(x.size(dim=0), 1, x.size(dim=2), device=x.device)
        x = torch.cat([pad, x[:, 1:, :]], dim=1)
        y = self.linear(x)
        T = y.size(dim=-2)
        H = y.size(dim=-1)
        return y.reshape(-1, T * self.pooling_factor, H // self.pooling_factor)


def SaShiMi(input_dim: int,
            hidden_dim: int,
            output_dim: int,
            state_dim: int,
            sequence_length: int,
            block_count: int,
           ):
    return Sequential(
        nn.Linear(input_dim, hidden_dim),
        Residual(
            DownPool(hidden_dim),
            Residual(
                DownPool(2 * hidden_dim),
                Residual(*[
                    S4Block(4 * hidden_dim, state_dim, sequence_length // 16)
                    for _ in range(block_count)
                ]),
                UpPool(2 * hidden_dim),
            ),
            *[S4Block(2 * hidden_dim, state_dim, sequence_length // 4) for _ in range(block_count)],
            UpPool(hidden_dim),
        ),
        *[S4Block(hidden_dim, state_dim, sequence_length) for _ in range(block_count)],
        nn.Linear(hidden_dim, output_dim),
    )
