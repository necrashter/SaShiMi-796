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

    def no_shift(self, x):
        """
        Apply up-pooling without shifting.
        Equivalent to calling the layer directly in UpPool.
        """
        y = self.linear(x)
        T = y.size(dim=-2)
        H = y.size(dim=-1)
        return y.reshape(-1, T * self.pooling_factor, H // self.pooling_factor)

    def forward(self, x):
        return self.no_shift(x)


class CausalUpPool(UpPool):
    def forward(self, x):
        # Use shifting to preserve causality
        pad = torch.zeros(x.size(dim=0), 1, x.size(dim=2), device=x.device)
        # Shift all elements to the right, discard last, pad the beginning with zero.
        x = torch.cat([pad, x[:, :-1, :]], dim=1)
        return self.no_shift(x)


class CausalPooledResidual(nn.Module):
    """
    A sequential block wrapped between DownPool and UpPool layers with a residual connection
    from its beginning to the end.
    """
    def __init__(self,
                 layers,
                 signal_dim: int,
                 pooling_factor: int = 4,
                 expansion_factor: int = 2,
                ):
        super().__init__()
        self.sequential = Sequential(*layers)
        self.down_pool = DownPool(signal_dim, pooling_factor, expansion_factor)
        self.up_pool = CausalUpPool(signal_dim, pooling_factor, expansion_factor)
        self.signal_dim = signal_dim
        self.pooling_factor = pooling_factor

    def forward(self, x):
        return self.up_pool(self.sequential(self.down_pool(x))) + x

    def get_recurrent_runner(self):
        """
        Discretize the model with given L and return a function that maps state and input to
        the new state and input.
        """
        input_cache = []
        # First block will be up-pooled from zeros due to shifting.
        device = next(iter(self.parameters())).device
        first = torch.zeros(1, self.up_pool.linear.in_features, device=device)
        first = self.up_pool.no_shift(first)
        # first shape: (1, self.pooling_factor, hidden)
        output_cache = [i for i in first.squeeze(0)]
        # Instead of removing items from the beginning, we reverse the list and pop from
        # the end. This is slightly faster.
        output_cache.reverse()

        sequential = self.sequential.get_recurrent_runner()

        def f(u):
            nonlocal sequential, input_cache, output_cache
            output = output_cache.pop() + u
            input_cache.append(u)

            if len(input_cache) == self.pooling_factor:
                x = torch.stack(input_cache)
                input_cache.clear()
                output_cache.clear()
                y = self.down_pool(x).squeeze()
                y = self.up_pool.no_shift(sequential(y).unsqueeze(0))
                output_cache = [i for i in y.squeeze(0)]
                output_cache.reverse()

            return output

        return f


def SaShiMi(input_dim: int,
            hidden_dim: int,
            output_dim: int,
            state_dim: int,
            sequence_length: int,
            block_count: int,
           ):
    return Sequential(
        nn.Linear(input_dim, hidden_dim),
        CausalPooledResidual(
            signal_dim = hidden_dim,
            layers = [
                CausalPooledResidual(
                    signal_dim = hidden_dim * 2,
                    layers = [Residual(*[
                        S4Block(4 * hidden_dim, state_dim, sequence_length // 16)
                        for _ in range(block_count)
                    ])],
                ),
                *[S4Block(2 * hidden_dim, state_dim, sequence_length // 4)
                  for _ in range(block_count)],
            ],
        ),
        *[S4Block(hidden_dim, state_dim, sequence_length) for _ in range(block_count)],
        nn.Linear(hidden_dim, output_dim),
    )
