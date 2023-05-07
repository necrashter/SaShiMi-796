# Implementation of S4 block.
# Main component in SaShiMi architecture.

import torch


def init_HiPPO(size: int):
    """
    Initialize HiPPO (High-Order Polynomial Projection Operator) matrix.
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


def conv_kernel_naive(A, B, C, L):
    """
    Get convolution kernel from discretized SSM parameters.
    Naive implementation for testing.
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

