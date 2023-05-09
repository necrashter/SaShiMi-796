"""
Implementation of S4 layer with related components.
"""
import torch
import torch.nn.functional as F
from torch import nn


def init_HiPPO(signal_dim: int, state_dim: int):
    """
    Initialize HiPPO (High-Order Polynomial Projection Operator) matrix.

    Equation 2 in "Efficiently Modeling Long Sequences with Structured State Spaces".
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
    """
    A, B = init_HiPPO(signal_dim, state_dim)
    A = A * -1.0
    # Rank 1 term to make HiPPO Normal.
    P = torch.sqrt(torch.arange(state_dim) + 0.5)
    return A, P, B


def init_DPLR_HiPPO(signal_dim: int, state_dim: int):
    """
    Initialize HiPPO with Diagonal Plus Low-Rank (DPLR) structure.
    This is done by diagonalizing the NPLR representation.
    """
    A, P, B = init_NPLR_HiPPO(signal_dim, state_dim)

    S = A + P.unsqueeze(1) * P.unsqueeze(0)

    # Check skew symmetry
    S_diag = torch.diagonal(S)
    Lambda_real = torch.mean(S_diag) * torch.ones_like(S_diag)
    assert torch.allclose(Lambda_real, S_diag, atol=1e-5)

    # Diagonalize S to V Lambda V* form.
    Lambda_imag, V = torch.linalg.eigh(S * -1j)
    Lambda = Lambda_real + Lambda_imag * 1j

    Vc = V.conj().T
    P = Vc @ P.to(torch.complex64)
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
    I = torch.eye(Lambda.size(dim=0))

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


def get_roots_of_unity(L: int):
    """
    Get the roots of unity at which the SSM generating function is evaluated.
    - L: input length.

    See Lemma C.2 in "Efficiently Modeling Long Sequences with Structured State Spaces".
    """
    return torch.exp(-2j * torch.pi * (torch.arange(L) / L))


def conv_kernel_DPLR(Lambda, P, Q, B, C, step, L):
    """
    Get convolution kernel from DPLR model parameters.
    - Lambda, P, Q, B, C: Model parameters (Complex Tensors)
    - step: Step size
    - L: Input length

    This is almost the same as Algorithm 1 in "Efficiently Modeling Long Sequences with
    Structured State Spaces". However, the first step that transforms C is skipped. The
    model can learn the transformed C in practice.

    The resulting convolution kernel is LxD where D is the signal dimensions.
    """
    # Roots of unith at which SSM generating function is evaluated.
    Omega = get_roots_of_unity(L)

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
    # Denominator in Cauchy dot product
    cauchy_denominator = g.unsqueeze(1) - Lambda
    # Size of cauchy_denominator: SxN
    # Cauchy dot products
    k00 = (a0 * b0 / cauchy_denominator).sum(dim=-1)
    k01 = (a0 * b1 / cauchy_denominator).sum(dim=-1)
    k10 = (a1 * b0 / cauchy_denominator).sum(dim=-1)
    k11 = (a1 * b1 / cauchy_denominator).sum(dim=-1)

    evaluated = (2.0 / (1.0 + Omega)) * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)
    out = torch.fft.ifft(evaluated, L)
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
    def __init__(self, signal_dim: int, state_dim: int):
        """
        Initialize S4.
        - signal_dim: Number of dimensions in the signal.
        - state_dim: Number of dimensions in inner state.
        """
        super().__init__()
        Lambda, _, P, B = init_DPLR_HiPPO(signal_dim, state_dim)
        # We need to store complex tensors as real tensors, otherwise some optimizers
        # (e.g. Adam) won't work. view_as_real returns an alias tensor with an additional
        # dimension (of size = 2, at the end) for real and complex parts.
        self.Lambda = nn.parameter.Parameter(torch.view_as_real(Lambda.resolve_conj()))
        self.P = nn.parameter.Parameter(torch.view_as_real(P.resolve_conj()))
        self.B = nn.parameter.Parameter(torch.view_as_real(B.resolve_conj()))

        C = torch.empty(1, state_dim, dtype=torch.complex64)
        nn.init.xavier_normal_(C)
        self.C = nn.parameter.Parameter(torch.view_as_real(C))

        # D is like a skip connection, hence initialize with 1's
        self.D = nn.parameter.Parameter(torch.ones(1))

        # Step size is a learnable parameter but stored as log.
        self.log_step = nn.parameter.Parameter(torch.empty(1).uniform_(0.001, 0.1))

    def get_conv_kernel(self, L):
        step = self.log_step.exp()
        Lambda = torch.view_as_complex(self.Lambda)
        P = torch.view_as_complex(self.P)
        B = torch.view_as_complex(self.B)
        C = torch.view_as_complex(self.C)
        return conv_kernel_DPLR(Lambda, P, P, B, C, step, L)

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
        L = u.size(dim=-2)
        K = self.get_conv_kernel(L)
        return convolve(u, K) + self.D * u

    def recurrent_forward(self, u: torch.Tensor):
        """
        Forward pass of the network with recurrent method.

        For input sequence length L, and input dimensions D, the argument is:
        - u: Input torch.Tensor of size LxD

        The input cannot be a batch for recurrent_forward.
        """
        L = u.size(dim=-2)
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

