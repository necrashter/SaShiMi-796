import torch
from torch import nn
from S4 import *
from torchaudio.functional import mu_law_encoding
from tqdm.auto import tqdm


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
    Same as S4Block, but all activations (GELU) are replaced with a GLU layer. Since GLU halves
    the signal dimensions, the output dimensions of the linear layers that precede GLU layers
    are multiplied by 2.

    See Appendix C.2.1 in "It's Raw! Audio Generation with State-Space Models":
    > On SC09, we found that swapping in a gated linear unit (GLU) in the S4 block improved
    > NLL as well as sample quality.
    """
    return Sequential(
        Residual(
            nn.LayerNorm(signal_dim),
            S4Base(signal_dim, state_dim, sequence_length),
            nn.Linear(signal_dim, signal_dim * 2),
            nn.GLU(),  # GLU halves the last dimension
        ),
        Residual(
            nn.LayerNorm(signal_dim),
            nn.Linear(signal_dim, signal_dim * expansion_factor * 2),
            nn.GLU(),  # GLU halves the last dimension
            nn.Linear(signal_dim * expansion_factor, signal_dim),
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
                x = torch.cat(input_cache, dim=-2)
                input_cache.clear()
                output_cache.clear()
                y = self.down_pool(x)
                y = self.up_pool.no_shift(sequential(y))
                output_cache = [i for i in torch.split(y, 1, dim=-2)]
                output_cache.reverse()

            return output

        return f


class Embedding(torch.nn.Embedding):
    pass


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


def generate_audio_sample(
        model,
        sample_count: int,
        batch_size: int = 1,
        priming_signal=None,
        starting_input=None,
        maxp=False,
        use_tqdm=True,
    ):
    """
    Generate an audio sample autoregressively from the model using 8-bit mu-law encoding.
    - model: Autoregressive audio model.
    - sample_count: Number of total samples in the output.
    - batch_size: Number of generated audio files.
    - priming_signal: Model will complete this signal if given.
                      The model will generate sample_count - priming_signal.size(0) samples.
                      The priming signal will be included in the output if provided.
    - starting_input: Normally, one sample of silence will be given to the model to start
                      the generation. If this argument is given, it will be used instead.
    - maxp: If true, the option with the highest probability will be selected instead of
            random sampling.
    - use_tqdm: Use tqdm library to display a progress bar.

    Returns:
    - A tensor of shape (batch_size, sample_count), containing samples in mu-law encoding.
    """
    f = model.get_recurrent_runner()
    # Pad the input with 0 sample to get started.
    device = next(model.parameters()).device
    if starting_input is None:
        starting_input = mu_law_encoding(torch.zeros(batch_size, 1, device=device), 256)
    u = f(starting_input)

    # Process the priming signal if given
    if priming_signal is not None:
        for s in priming_signal:
            u = f(s.reshape(1, -1).expand(batch_size, -1))
        primed_size = priming_signal.size(0)
    else:
        primed_size = 0

    # Generate the new part
    Y = []
    iterator = range(sample_count - primed_size)
    # Don't use tqdm while testing
    if use_tqdm:
        iterator = tqdm(iterator, leave=False)
    for _ in iterator:
        if maxp:
            p = torch.argmax(u, dim=-1)
        else:
            dist = torch.distributions.categorical.Categorical(
                probs=torch.nn.functional.softmax(u, dim=-1),
            )
            p = dist.sample()
        Y.append(p)
        u = f(p)

    generated = torch.cat(Y, dim=1)
    if priming_signal is not None:
        priming_signal = priming_signal.flatten()
        return torch.cat([priming_signal.reshape(1, -1), generated], dim=1)
    else:
        return generated
