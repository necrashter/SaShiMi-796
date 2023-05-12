"""
Contains the implementation of S4 block module.
"""
import torch
from torch import nn
from . layer import S4Base


class Lambda(nn.Module):
    """
    A utility layer that applies the given function.
    """
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class Sequential(nn.Sequential):
    """
    A sequential NN block that accounts for the S4 layers when generating samples.
    Subclass of `torch.nn.Sequential`.
    """
    def get_recurrent_runner(self):
        """
        Discretize the model with given L and return a function that maps state and input to
        the new state and input.
        """
        layers = [
            layer.get_recurrent_runner() if hasattr(layer, "get_recurrent_runner") else layer
            for layer in self
        ]

        def f(u):
            for layer in layers:
                u = layer(u)
            return u

        return f

    def autoregressive_sample(self, samples: int, signal):
        """
        Sample in autoregressive fashion: feed the output of the previous iteration as input.
        - samples: Number of new samples
        - signal: Starting signal of shape LxD where L is the length, D is dimension
        """
        L = (samples + signal.size(dim=-2)) if signal is not None else samples
        f = self.get_recurrent_runner()

        # Process the given signal
        for s in signal:
            u = f(s)

        # Generate the new part
        Y = []
        for _ in range(samples):
            y = f(u)
            Y.append(y)
            u = y

        generated = torch.stack(Y).real
        if signal is not None:
            return torch.cat([signal, generated], dim=0)
        else:
            return generated


class Residual(Sequential):
    """
    A sequential block with a residual connection from its beginning to the end.
    """
    def get_recurrent_runner(self):
        """
        Discretize the model with given L and return a function that maps state and input to
        the new state and input.
        """
        layers = [
            layer.get_recurrent_runner() if hasattr(layer, "get_recurrent_runner") else layer
            for layer in self
        ]

        def f(x):
            y = x
            for layer in layers:
                y = layer(y)
            return y + x

        return f

    def forward(self, x):
        return super().forward(x) + x


def S4Block(signal_dim: int, state_dim: int, sequence_length: int, expansion_factor: int = 2):
    """
    Construct the full S4 block given in SaShiMi paper. Arguments:
    - signal_dim: Number of dimensions in the signal.
    - state_dim: Number of dimensions in inner state.
    - sequence_length: The length of the sequence on which this model will operate.
        - Can be changed later, but models trained on one sequence length perform poorly
          on another sequence length.
    - expansion_factor: The factor by which the number of dimensions will be multiplied
                        between two linear layers in the second pass.

    High-level Architecture
    -----------------------

    The architecture is as described in Appendix A.2 of "It’s Raw! Audio Generation with
    State-Space Models" paper.

    First pass:
    1. Input
    2. LayerNorm
    3. S4 Layer
    4. GELU
    5. Linear
    6. Residual connection from 1

    Second pass:
    1. Output of the first pass
    2. LayerNorm
    3. Linear
    4. GELU
    5. Linear
    6. Residual connection from 1

    All linear layers are position-wise, i.e., they operate on the signal dimensions, not
    the time dimension.
    """
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
