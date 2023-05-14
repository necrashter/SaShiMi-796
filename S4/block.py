"""
Contains the implementation of S4 block module.
"""
from torch import nn


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
