import torch
from torch import nn
from . block import *


class Lambda(nn.Module):
    """
    A utility layer that applies the given function.
    """
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class SequentialS4(nn.Sequential):
    def get_recurrent_runner(self, L: int):
        """
        Discretize the model with given L and return a function that maps state and input to
        the new state and input.
        """
        layers = [
            layer.get_recurrent_runner(L) if hasattr(layer, "get_recurrent_runner") else layer
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
        f = self.get_recurrent_runner(L)

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

