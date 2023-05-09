"""
Contains the implementation of S4 block module.
"""
from torch import nn
from . layer import S4Base


class S4Block(nn.Module):
    """
    Implementation of an S4 block.

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

    def __init__(self, signal_dim: int, state_dim: int, expansion_factor: int = 2):
        """
        Construct a full S4 block. Arguments:
        - signal_dim: Number of dimensions in the signal.
        - state_dim: Number of dimensions in inner state.
        - expansion_factor: The factor by which the number of dimensions will be multiplied
                            between two linear layers in the second pass.
        """
        super().__init__()
        # Pass1 before S4
        self.pass1pre = nn.LayerNorm(signal_dim)
        # S4 Layer
        self.s4 = S4Base(signal_dim, state_dim)
        # Pass1 after S4
        self.pass1post = nn.Sequential(
            nn.GELU(),
            nn.Linear(signal_dim, signal_dim),
        )
        # Residual connection from the beginning of pass1 to end of pass1

        self.pass2 = nn.Sequential(
            nn.LayerNorm(signal_dim),
            nn.Linear(signal_dim, signal_dim * expansion_factor),
            nn.GELU(),
            nn.Linear(signal_dim * expansion_factor, signal_dim),
        )
        # Residual connection from the beginning of pass2 to end of pass2

    def forward(self, u):
        """
        For batch size B, input sequence length L, and input dimensions D, the argument is:
        - u: Input torch.Tensor of size BxLxD or LxD
        """
        a = self.pass1post(self.s4(self.pass1pre(u))) + u
        b = self.pass2(a) + a
        return b

    def get_recurrent_runner(self, L: int):
        """
        Discretize the model with given L and return a stateful function that maps the input
        signal to output signal one sample at a time.
        """
        s4 = self.s4.get_recurrent_runner(L)

        def f(u):
            a = self.pass1post(s4(self.pass1pre(u))) + u
            b = self.pass2(a) + a
            return b

        return f

