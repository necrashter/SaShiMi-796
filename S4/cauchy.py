"""
Implementations of Cauchy kernel computation.
- Memory efficient PyKeOps implementation is used when PyKeOps is installed.
- Otherwise, the naive PyTorch implementation is used as a fallback.
"""
import torch


def naive_cauchy_kernel(a0: torch.Tensor,
                        a1: torch.Tensor,
                        b0: torch.Tensor,
                        b1: torch.Tensor,
                        g: torch.Tensor,
                        Lambda: torch.Tensor,
                        ):
    """
    Compute the Cauchy kernel with naive method.
    """
    denominator = g.unsqueeze(1) - Lambda
    # a0 to b1:    (SIGNAL, 1, STATE)
    # denominator: (1, SAMPLES, STATE)
    k00 = (a0 * b0 / denominator).sum(dim=-1)
    k01 = (a0 * b1 / denominator).sum(dim=-1)
    k10 = (a1 * b0 / denominator).sum(dim=-1)
    k11 = (a1 * b1 / denominator).sum(dim=-1)
    return k00, k01, k10, k11


try:
    from pykeops.torch import LazyTensor

    def pykeops_cauchy_kernel(a0: torch.Tensor,
                              a1: torch.Tensor,
                              b0: torch.Tensor,
                              b1: torch.Tensor,
                              g: torch.Tensor,
                              Lambda: torch.Tensor,
                              ):
        """
        Compute the Cauchy kernel using PyKeOps for better memory efficiency.
        """
        # For PyKeOps, we need two tensors of shapes (..., M, 1, D) and (..., 1, N, D).
        # D = 1 in our case.
        # Let's use M for sequence_length and N for state_dim.
        g = LazyTensor(g.view(-1, 1, 1))
        Lambda = LazyTensor(Lambda.view(1, -1, 1))
        # NOTE: Running contiguous() can slow things down. There might be a more efficient
        # way to reshape these into the form that PyKeOps wants.
        a0 = a0.view(a0.size(dim=0), 1, -1, 1).contiguous()
        a1 = a1.view(a1.size(dim=0), 1, -1, 1).contiguous()
        b0 = b0.view(b0.size(dim=0), 1, -1, 1).contiguous()
        b1 = b1.view(b1.size(dim=0), 1, -1, 1).contiguous()
        # a0 to b1:    (SIGNAL,  1, STATE, 1)
        # denominator:    (1, STATE, SAMPLES)
        denominator = g - Lambda
        k00 = (LazyTensor(a0 * b0) / denominator).sum_reduction(axis=2).squeeze(-1)
        k01 = (LazyTensor(a0 * b1) / denominator).sum_reduction(axis=2).squeeze(-1)
        k10 = (LazyTensor(a1 * b0) / denominator).sum_reduction(axis=2).squeeze(-1)
        k11 = (LazyTensor(a1 * b1) / denominator).sum_reduction(axis=2).squeeze(-1)
        return k00, k01, k10, k11

    cauchy_kernel = pykeops_cauchy_kernel

except ModuleNotFoundError:
    print("WARNING: PyKeOps not found, using the naive Cauchy kernel method.")
    cauchy_kernel = naive_cauchy_kernel


if __name__ == "__main__":
    import time

    state_dim = 256
    signal_dim = 256
    sequence_length = 1024

    dtype = torch.complex64
    device = torch.device("cuda")

    a0, a1, b0, b1 = [torch.randn(signal_dim, 1, state_dim, dtype=dtype, device=device) for _ in range(4)]
    g = torch.randn(sequence_length, dtype=dtype, device=device)
    Lambda = torch.randn(state_dim, dtype=dtype, device=device)

    print("Benchmarking...")

    start = time.time()
    naive_ks = naive_cauchy_kernel(a0, a1, b0, b1, g, Lambda)
    end = time.time()
    print("NAIVE:", end - start)

    start = time.time()
    pykeops_ks = pykeops_cauchy_kernel(a0, a1, b0, b1, g, Lambda)
    end = time.time()
    print("PYKEOPS:", end - start)

    for a, b in zip(naive_ks, pykeops_ks):
        assert type(a) == type(b)
        print("Max difference:", (a-b).abs().max().item())
