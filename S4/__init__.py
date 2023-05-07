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

