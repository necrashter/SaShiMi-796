"""
Implementation of S4 layer with related components.
"""
import torch
import torch.nn.functional as F
from torch import nn
from . import cauchy


def init_HiPPO(signal_dim: int, state_dim: int):
    """
    Initialize the A and B tensors of HiPPO (High-Order Polynomial Projection Operator).

    Equation 2 in "Efficiently Modeling Long Sequences with Structured State Spaces".
    More specifically, Theorem 2 in "HiPPO: Recurrent Memory with Optimal Polynomial
    Projections".

    This HiPPO variant is called HiPPO-LegS. It assigns uniform weight to all history.
    """
    A = torch.zeros(state_dim, state_dim)
    for k in range(state_dim):
        # HiPPO matrix is non-zero only for n >= k
        for n in range(k, state_dim):
            if n > k:
                A[n, k] = (2*n + 1)**.5 * (2*k + 1)**.5
            else:
                A[n, k] = n + 1
    B = torch.sqrt(torch.arange(state_dim) * 2.0 + 1.0).unsqueeze(1)
    B = torch.cat([B for _ in range(signal_dim)], dim=1)
    return A, B


def init_NPLR_HiPPO(signal_dim: int, state_dim: int):
    """
    Initialize HiPPO with Normal Plus Low-Rank (NPLR) structure.
    Returns A, P, B where A and B are from init_HiPPO (A is negated).
    P is a 1D tensor such that the equation
        A = (S + 0.5 I) - P @ P.conj().T
    holds for a skew-symmetric matrix S, i.e., S.T = -S holds.
    This also means that S can be diagonalized.
    S = V @ Lambda @ V.conj().T for some diagonal matrix Lambda.
    See also: init_DPLR_HiPPO.

    See HiPPO-LegS part of Appendix C.1 in "Efficiently Modeling Long Sequences with
    Structured State Spaces".
    """
    A, B = init_HiPPO(signal_dim, state_dim)
    # A matrix from HiPPO-LegS is negated in Appendix C.1.
    A = A * -1.0
    # Note that in Appendix C.1, they add 0.5 * sqrt(2n+1) * sqrt(2k+1) to each element
    # in matrix. The matrix form of these added terms corresponds to the outer product of 
    # sqrt(0.5) * sqrt(2n+1) vector. Simplifying this, we get sqrt(n + 0.5).
    P = torch.sqrt(torch.arange(state_dim) + 0.5)
    return A, P, B


def init_DPLR_HiPPO(signal_dim: int, state_dim: int):
    """
    Initialize HiPPO with Diagonal Plus Low-Rank (DPLR) structure.
    Returns Lambda, V, P, B tensors where
    - Lambda, V, P are diagonalized version of A. They are complex tensors.
    - B is B from init_HiPPO but left-multiplied with the inverse of V. More on this
      below.

    In particular, the following equation holds:
        A = V @ (Lambda - P @ P.conj().T) @ V.conj().T
    By Lemma 3.1 in "Efficiently Modeling Long Sequences with Structured State Spaces", 
    we can directly use the inner part Lambda - P @ P.conj().T as our A matrix. To do
    this, we also need to set our B vector to V^{-1} B where B is the original B from
    init_HiPPO.

    This is done by diagonalizing the NPLR representation.
    """
    A, P, B = init_NPLR_HiPPO(signal_dim, state_dim)

    # Note that the following equality holds: A = (S + 0.5 I) - P @ P.T
    # Recover (S + 0.5 I) from that equation.
    SI = A + P.unsqueeze(1) * P.unsqueeze(0)

    # Recover S from (S + 0.5 I)
    diag = torch.diagonal(SI)
    S = SI - torch.diag(diag)
    # All values in diag must be 0.5, but it doesn't always happen due to floating
    # point inaccuracy.

    # We can now diagonalize S.
    # Since S is a skew-symmetric matrix, j*S is a Hermitian matrix where j is the
    # imaginary unit. We can get the eigenvalues of -j*S and multiply them with j
    # to get the eigenvalues of S (since j^2 = -1).
    eigenvalues, V = torch.linalg.eigh(S * -1j)
    Lambda = diag + eigenvalues * 1j

    # At this point the following equation holds:
    #   A = V @ (Lambda - (V.conj().T @ P) @ (V.conj().T @ P.conj().T)) @ V.T
    # Simply left-multiply P with V.conj().T and use that as P.
    Vc = V.conj().T
    P = Vc @ P.to(torch.complex64)

    # Now we will use Lambda - P @ P.conj().T as A.
    # Remember from Lemma 3.1 that we also need to set the new B to V^{-1} B.
    # Since B is unitary, we can use conjugate transpose as V^{-1}.
    B = Vc @ B.to(torch.complex64)

    return Lambda, V, P, B


def discretize_SSM(A, B, C, step):
    """
    Discretize the given State Space Model (SSM) with the given step size.

    Equation 3 in "Efficiently Modeling Long Sequences with Structured State Spaces".
    """
    I = torch.eye(A.shape[0])
    # Common term in A and B equations
    left = torch.linalg.inv(I - (step / 2.0) * A)
    A = left @ (I + (step / 2.0) * A)
    B = (left * step) @ B
    # C stays the same
    return A, B, C


def discretize_DPLR(Lambda, P, Q, B, C, step, L):
    """
    Discretize the given SSM in DPRL representation with the given step size.
    - Lambda, P, Q, B, C: Model parameters (Complex Tensors)
    - step: Step size
    - L: Input length

    See Appendix C.2 in "Efficiently Modeling Long Sequences with Structured State Spaces".
    """
    # Convert to matrices
    if len(P.size()) < 2:
        P = P.unsqueeze(1)
    if len(Q.size()) < 2:
        Q = Q.unsqueeze(1)
    # Conjugate transpose of Q
    Qstar = Q.conj().T
    # Build A matrix in SSM from DPLR parameters
    A = torch.diag(Lambda) - P @ Qstar
    I = torch.eye(Lambda.size(dim=0), device=Lambda.device)

    # Forward discretization
    A0 = I * (2.0 / step) + A

    # Backward discretization
    D = torch.diag(1.0 / ((2.0 / step) - Lambda))
    A1 = D - (D @ P * (1.0 / (1 + (Qstar @ D @ P))) * Qstar @ D)

    # S4 Recurrence
    Ab = A1 @ A0
    Bb = 2 * A1 @ B

    # Note that we don't learn C directly, we learn C^~, the result of the first step in
    # Algorithm 1. Therefore, we need to get the actual C bar from C^~.
    Cb = C.conj() @ torch.linalg.inv(I - torch.linalg.matrix_power(Ab, L))
    return Ab, Bb, Cb


def conv_kernel_naive(A, B, C, L):
    """
    Get convolution kernel from discretized SSM parameters.
    Naive implementation for testing.

    Equation 4 and 5 in "Efficiently Modeling Long Sequences with Structured State Spaces".
    """
    # Size of B: NxD
    # Size of C: DxN
    # Organize their shapes such that D dimension does not play a role in matrix
    # multiplication C A^i B
    C2 = C.unsqueeze(1)  # C2: Dx1xN
    B2 = B.T.unsqueeze(-1)  # B2: DxNx1
    # The resulting kernel is LxD
    return torch.cat([
        # The result of matrix multiplication is Dx1x1
        (C2 @ torch.linalg.matrix_power(A, i) @ B2).reshape(1, -1)
        for i in range(L)
    ])


def get_roots_of_unity(L: int, **kwargs):
    """
    Get the roots of unity at which the SSM generating function is evaluated.
    - L: input length.
    - **kwargs: Keyword arguments for torch.arange function.

    See Lemma C.2 in "Efficiently Modeling Long Sequences with Structured State Spaces".
    """
    return torch.exp(-2j * torch.pi * (torch.arange(L, **kwargs) / L))


def conv_kernel_DPLR(Lambda, P, Q, B, C, step, Omega):
    """
    Get convolution kernel from DPLR model parameters.
    - Lambda, P, Q, B, C: Model parameters (Complex Tensors)
    - step: Step size
    - Omega: Roots of unity at which SSM generating function is evaluated.
        - Must be generated from get_roots_of_unity(L) where L is the sequence length.

    This is almost the same as Algorithm 1 in "Efficiently Modeling Long Sequences with
    Structured State Spaces". However, the first step that transforms C is skipped. The
    model can learn the transformed C in practice.

    The resulting convolution kernel is LxD where D is the signal dimensions.
    """
    # Size of B: NxD
    # Size of C: DxN
    a0, a1, b0, b1 = [
        i.unsqueeze(1) # DxN to Dx1xN, 1 is for sequence length (broadcasted)
        for i in [
            C.conj(), Q.conj().unsqueeze(0),
            B.T, P.unsqueeze(0),
        ]
    ]

    g = (2.0 / step) * ((1.0 - Omega) / (1.0 + Omega))
    # Size of cauchy_denominator: SxN
    # Cauchy dot products
    k00, k01, k10, k11 = cauchy.cauchy_kernel(a0, a1, b0, b1, g, Lambda)

    evaluated = (2.0 / (1.0 + Omega)) * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)
    out = torch.fft.ifft(evaluated)
    # out is DxL, transpose to get LxD
    out = out.T
    return out.real


def convolve(u: torch.Tensor, K: torch.Tensor):
    """
    Apply convolution on u with kernel K. Operates on the last dimension.
    For batch size B, input sequence length L, and input dimensions D, the arguments are:
    - u: Input torch.Tensor of size BxLxD or LxD
    - K: Kernel torch.Tensor of size LxD

    Uses the Convolution Theorem: Convolution of 2 signals is the product of their
    Fourier transforms.
    """
    l_max = u.size(dim=-2)
    # Pad the time dimension, which is the second from the last
    ud = torch.fft.rfft(F.pad(u.real, pad=(0, 0, 0, l_max)), dim=-2)
    Kd = torch.fft.rfft(F.pad(K.real, pad=(0, 0, 0, l_max)), dim=-2)
    product = ud * Kd
    return torch.fft.irfft(product, dim=-2)[..., :l_max, :]


def run_recurrent_SSM(Ab, Bb, Cb, u, x0=None):
    """
    Run the discretized SSM with given parameters on input signal u with initial x0.

    For input sequence length L, and input dimensions D, the argument is:
    - u: Input torch.Tensor of size LxD

    Note that the input cannot be a batch.
    """
    x = x0 if x0 is not None else torch.zeros(Ab.size(dim=0), u.size(dim=-1))
    x = x.to(Ab.dtype)
    u = u.to(Ab.dtype)
    X, Y = [], []
    Ct = Cb.T
    for u_k in u:
        x = Ab @ x + Bb * u_k
        y = torch.sum(Ct * x, dim=0).flatten()
        X.append(x)
        Y.append(y)

    return torch.stack(X), torch.stack(Y)


class S4Base(nn.Module):
    """
    An S4 layer module. It represents an SSM in DPLR form.
    """
    def __init__(self, signal_dim: int, state_dim: int, sequence_length: int):
        """
        Initialize S4.
        - signal_dim: Number of dimensions in the signal.
        - state_dim: Number of dimensions in inner state.
        - sequence_length: The length of the sequence on which this model will operate.
            - Can be changed later, but models trained on one sequence length perform
              poorly on another sequence length.

        The parameters for A and B matrices will be tied.
        """
        super().__init__()
        # For parameter tying, we simply set the signal dimension to 1 for A and B matrices.
        # Broadcasting will take care of the rest.
        Lambda, _, P, B = init_DPLR_HiPPO(1, state_dim)
        # NOTE: If you use signal_dim here instead of 1, the same A matrix will still be
        # shared along the signal dimension. Learning a separate A matrix is for each
        # dimension is not implemented.

        # We need to store complex tensors as real tensors, otherwise some optimizers
        # (e.g. Adam) won't work. view_as_real returns an alias tensor with an additional
        # dimension (of size = 2, at the end) for real and complex parts.
        self.Lambda = nn.parameter.Parameter(torch.view_as_real(Lambda.resolve_conj()))
        self.P = nn.parameter.Parameter(torch.view_as_real(P.resolve_conj()))
        self.B = nn.parameter.Parameter(torch.view_as_real(B.resolve_conj()))
        # We don't have a separate Q parameter. We simply use P as Q.
        # See section 4.1 in "It’s Raw! Audio Generation with State-Space Models".

        # Standard normal initialization works better than xavier_normal_ initialization
        # on this parameter.
        C = torch.randn(signal_dim, state_dim, dtype=torch.complex64)
        # nn.init.xavier_normal_(C)
        self.C = nn.parameter.Parameter(torch.view_as_real(C))

        # How you Initialize this parameter doesn't affect much.
        self.D = nn.parameter.Parameter(torch.randn(signal_dim))

        # Step size is a learnable parameter but stored as log.
        self.log_step = nn.parameter.Parameter(torch.empty(1).uniform_(0.001, 0.1).log())

        # The roots of unity is cached in order to compute the convolution kernel faster.
        # Note that it's a buffer, not a parameter. It won't be trained.
        self.register_buffer("Omega", get_roots_of_unity(sequence_length))

    @property
    def sequence_length(self):
        """
        The length of the sequence on which this model will operate.
        Can be changed after construction, but models trained on one sequence length
        perform poorly on another sequence length.
        """
        return self.Omega.size(dim=0)

    @sequence_length.setter
    def sequence_length(self, value):
        self.Omega = get_roots_of_unity(value)

    def get_conv_kernel(self, Omega):
        step = self.log_step.exp()
        Lambda = torch.view_as_complex(self.Lambda)
        P = torch.view_as_complex(self.P)
        B = torch.view_as_complex(self.B)
        C = torch.view_as_complex(self.C)
        return conv_kernel_DPLR(Lambda, P, P, B, C, step, Omega)

    def discretize(self, L):
        step = self.log_step.exp()
        Lambda = torch.view_as_complex(self.Lambda)
        P = torch.view_as_complex(self.P)
        B = torch.view_as_complex(self.B)
        C = torch.view_as_complex(self.C)
        return discretize_DPLR(Lambda, P, P, B, C, step, L)

    def convolutional_forward(self, u: torch.Tensor):
        """
        Forward pass of the network with convolutional method.

        For batch size B, input sequence length L, and input dimensions D, the argument is:
        - u: Input torch.Tensor of size BxLxD or LxD
        """
        if u.size(dim=-2) != self.sequence_length:
            raise ValueError(f"This model has a sequence_length of {self.sequence_length}," +
                             f" but the length of the input is {u.size(dim=-2)}")
        K = self.get_conv_kernel(self.Omega)
        return convolve(u, K) + self.D * u

    def recurrent_forward(self, u: torch.Tensor):
        """
        Forward pass of the network with recurrent method.

        For input sequence length L, and input dimensions D, the argument is:
        - u: Input torch.Tensor of size LxD

        The input cannot be a batch for recurrent_forward.
        """
        L = u.size(dim=-2)
        if L != self.sequence_length:
            raise ValueError(f"This model has a sequence_length of {self.sequence_length}," +
                             f" but the length of the input is {L}")
        Ab, Bb, Cb = self.discretize(L)
        # States are ignored
        _, out = run_recurrent_SSM(Ab, Bb, Cb, u)
        return out + self.D * u

    def forward(self, u: torch.Tensor):
        """
        By default, forward pass is executed with convolutional method for better
        runtime performance.

        For batch size B, input sequence length L, and input dimensions D, the argument is:
        - u: Input torch.Tensor of size BxLxD or LxD
        """
        return self.convolutional_forward(u)

    def get_recurrent_runner(self):
        """
        Discretize the model and return a stateful function that maps the input
        signal to output signal one sample at a time.
        """
        Ab, Bb, Cb = self.discretize(self.sequence_length)
        Ct = Cb.T
        device = next(self.parameters()).device
        x = torch.zeros(Ab.size(dim=0), Bb.size(dim=-1), device=device, dtype=Ab.dtype)

        def f(u):
            nonlocal x
            x = Ab @ x + Bb * u
            y = torch.sum(Ct * x, dim=-2).unsqueeze(-2)
            return y.real + self.D * u

        return f
