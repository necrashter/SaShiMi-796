import torch
from torch import nn
from S4 import *


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


def S4BlockGLU(signal_dim: int, state_dim: int, sequence_length: int, expansion_factor: int = 2):
    """
    Same as S4Block, but it features a GLU layer after the last linear layer.

    See Appendix C.2.1 in "It's Raw! Audio Generation with State-Space Models":
    > On SC09, we found that swapping in a gated linear unit (GLU) in the S4 block improved
    > NLL as well as sample quality.
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
            nn.Linear(signal_dim * expansion_factor, signal_dim * 2),
            nn.GLU(),  # GLU halves the last dimension
        ),
    )


class DownPool(nn.Module):
    """
    Let p be the pooling factor and q the expansion factor.
    The down-pooling operation is:

           reshape                 linear
    (T,H) ---------> (T/p, H * p) --------> (T/p, H * q)

    Preserves the dimensions when combined with an UpPool layer with same settings.
    """
    def __init__(self, signal_dim: int, pooling_factor: int = 4, expansion_factor: int = 2):
        """
        - signal_dim: Input signal dimensions.
        - pooling_factor: Time is divided and hidden dimension is multiplied by this.
        - expansion_factor: Ratio between the hidden dimension of output and input.
        """
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

    def get_recurrent_runner(self):
        raise TypeError("DownPool cannot be used in recurrent mode by itself. " +
                        "See CausalPooledResidual.")


class UpPool(nn.Module):
    """
    Let p be the pooling factor and q the expansion factor.
    The up-pooling operation is the opposite of the down-pooling operation:

                  linear                 reshape
    (T/p, H * q) --------> (T/p, H * p) ---------> (T,H)

    Preserves the dimensions when combined with a DownPool layer with same settings.
    """
    def __init__(self, signal_dim: int, pooling_factor: int = 4, expansion_factor: int = 2):
        """
        - signal_dim: Output signal dimensions.
        - pooling_factor: Time is multiplied and hidden dimension is divided by this.
        - expansion_factor: Ratio between the hidden dimension of input and output.
        """
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

    def get_recurrent_runner(self):
        raise TypeError("UpPool cannot be used in recurrent mode by itself. " +
                        "See CausalPooledResidual.")


class CausalUpPool(UpPool):
    """
    Same as up-pooling, but shifts the input to the right and pads with zero in order to
    preserve causality.

    Note that regular up-pool breaks causality because the model can see the samples in
    the future if they are in the same block.
    """
    def forward(self, x):
        # Use shifting to preserve causality
        x = torch.nn.functional.pad(x[:, :-1, :], pad=(0, 0, 1, 0))
        return self.no_shift(x)


class CausalPooledResidual(nn.Module):
    """
    A sequential block wrapped between DownPool and UpPool layers with a residual connection
    from its beginning to the end.

    Equivalent to this for convolution:

        Residual(
            DownPool(hidden_dim),
            *sequential_blocks,
            UpPool(hidden_dim),
        )

    But is capable of running recurrently unlike the block above.
    """
    def __init__(self,
                 layers,
                 signal_dim: int,
                 pooling_factor: int = 4,
                 expansion_factor: int = 2,
                ):
        """
        - layers: List of layers in sequential block.
        - signal_dim, pooling_factor, expansion_factor: Parameters for pooling layers.
        """
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


class Embedding(torch.nn.Embedding):
    def get_recurrent_runner(self):
        """
        Returns a function that maps given single dimensional input to output.
        """
        def f(x):
            return self(x).squeeze(0)

        return f


def SaShiMi(input_dim: int,
            hidden_dim: int,
            output_dim: int,
            state_dim: int,
            sequence_length: int,
            block_count: int,
            block_class=S4Block,
            encoder=None,
            decoder=None,
           ):
    """
    Construct the SaShiMi architecture given in Figure 1 of "It’s Raw! Audio Generation with
    State-Space Models" paper.
    - input_dim: Input signal dimension.
    - hidden_dim: Signal dimension in the S4 blocks.
    - output_dim: Output signal dimension.
    - state_dim, sequence_length: Parameters for S4 blocks.
    - block_count: Number of S4 blocks in each series of S4 Blocks.
    - block_class: S4 block class. Can be S4Block or S4BlockGLU.
    - encoder: Optional encoder layer. A linear layer is constructed if not provided.
    - decoder: Optional decoder layer. A linear layer is constructed if not provided.
    """
    encoder = nn.Linear(input_dim, hidden_dim) if encoder is None else encoder
    decoder = nn.Linear(hidden_dim, output_dim) if decoder is None else decoder
    return Sequential(
        encoder,
        CausalPooledResidual(
            signal_dim = hidden_dim,
            layers = [
                CausalPooledResidual(
                    signal_dim = hidden_dim * 2,
                    layers = [Residual(*[
                        block_class(4 * hidden_dim, state_dim, sequence_length // 16)
                        for _ in range(block_count)
                    ])],
                ),
                *[block_class(2 * hidden_dim, state_dim, sequence_length // 4)
                  for _ in range(block_count)],
            ],
        ),
        *[block_class(hidden_dim, state_dim, sequence_length) for _ in range(block_count)],
        decoder,
    )
