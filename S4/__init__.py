# Implementation of S4 block.
# Main component in SaShiMi architecture.

import torch
import torch.nn.functional as F


def init_HiPPO(size: int):
    """
    Initialize HiPPO (High-Order Polynomial Projection Operator) matrix.

    Equation 2 in "Efficiently Modeling Long Sequences with Structured State Spaces".
    """
    hippo = torch.zeros(size, size)
    for k in range(size):
        # HiPPO matrix is non-zero only for n >= k
        for n in range(k, size):
            if n > k:
                hippo[n, k] = (2*n + 1)**.5 * (2*k + 1)**.5
            else:
                hippo[n, k] = n + 1
    return hippo


def init_NPLR_HiPPO(size: int):
    """
    Initialize HiPPO with Normal Plus Low-Rank (NPLR) structure.
    """
    neg_hippo = init_HiPPO(size) * -1.0
    # Rank 1 term to make HiPPO Normal.
    P = torch.sqrt(torch.arange(size) + 0.5)
    # HiPPO also specifies a B matrix.
    B = torch.sqrt(torch.arange(size) * 2.0 + 1.0)
    return neg_hippo, P, B


def init_DPLR_HiPPO(N):
    """
    Initialize HiPPO with Diagonal Plus Low-Rank (DPLR) structure.
    This is done by diagonalizing the NPLR representation.
    """
    A, P, B = init_NPLR_HiPPO(N)

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
    Bb = 2 * A1 @ B.unsqueeze(1)

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
    return torch.cat([C @ torch.linalg.matrix_power(A, i) @ B for i in range(L)])


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
    """
    # Roots of unith at which SSM generating function is evaluated.
    Omega = get_roots_of_unity(L)

    a0, a1 = (C.conj(), Q.conj())
    b0, b1 = (B, P)

    g = (2.0 / step) * ((1.0 - Omega) / (1.0 + Omega))
    # Denominator in Cauchy dot product
    cauchy_denominator = g.unsqueeze(1) - Lambda
    # Cauchy dot products
    k00 = (a0 * b0 / cauchy_denominator).sum(dim=-1)
    k01 = (a0 * b1 / cauchy_denominator).sum(dim=-1)
    k10 = (a1 * b0 / cauchy_denominator).sum(dim=-1)
    k11 = (a1 * b1 / cauchy_denominator).sum(dim=-1)

    evaluated = (2.0 / (1.0 + Omega)) * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)
    out = torch.fft.ifft(evaluated, L)
    return out.real


def convolve(u: torch.Tensor, K: torch.Tensor):
    """
    Apply convolution on u with kernel K. Operates on the last dimension.

    Uses the Convolution Theorem.
    Convolution of 2 signals is the product of their Fourier transforms.
    """
    l_max = u.size(dim=-1)
    ud = torch.fft.rfft(F.pad(u.real, pad=(0, l_max)), dim=-1)
    Kd = torch.fft.rfft(F.pad(K.real, pad=(0, l_max)), dim=-1)
    product = ud * Kd
    return torch.fft.irfft(product)[..., :l_max]


def run_recurrent_SSM(Ab, Bb, Cb, u, x0=None):
    """
    Run the discretized SSM with given parameters on input signal u with initial x0.
    """
    x = x0 if x0 is not None else torch.zeros(Ab.size(dim=0))
    x = x.to(Ab.dtype)
    u = u.to(Ab.dtype)
    if len(Bb.size()) < 2:
        Bb = Bb.unsqueeze(1)
    X, Y = [], []
    for u_k in u:
        x = Ab @ x + Bb @ u_k
        y = Cb @ x
        X.append(x)
        Y.append(y)

    return torch.cat(X), torch.cat(Y)

