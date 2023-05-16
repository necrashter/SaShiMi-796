"""
Implementations of Cauchy kernel computation.
- Memory efficient PyKeOps implementation is used when PyKeOps is installed.
- Otherwise, the naive PyTorch implementation is used as a fallback.

NOTE: DO NOT import this module using star, i.e., `from cauchy import *`.
This module uses global variables as module attributes. It will pollute your namespace.
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


cauchy_kernel_methods = {
    "naive": naive_cauchy_kernel,
}


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

    cauchy_kernel_methods["pykeops"] = pykeops_cauchy_kernel
    print("Using PyKeOps Cauchy kernel.")
    cauchy_kernel = pykeops_cauchy_kernel

except ModuleNotFoundError:
    print("PyKeOps not found, using the naive Cauchy kernel method.")
    cauchy_kernel = naive_cauchy_kernel


def select_method(name: str, silent=False):
    global cauchy_kernel, cauchy_kernel_methods
    if name in cauchy_kernel_methods:
        cauchy_kernel = cauchy_kernel_methods[name]
        if not silent:
            print("Switched to Cauchy kernel method:", name)
    else:
        raise ValueError("Cauchy kernel method is unavailable:", name)


if __name__ == "__main__":
    import time
    import argparse

    parser = argparse.ArgumentParser(
        prog="Cauchy Kernel Benchmark",
        description="Benchmark naive vs PyKeOps implementations of Cauchy kernel computation.",
        epilog="Text at the bottom of help",
    )
    parser.add_argument(
        "-s", "--state-dim",
        type=int,
        default=256,
        help="State dimensions",
    )
    parser.add_argument(
        "-d", "--signal-dim",
        type=int,
        default=256,
        help="Signal dimensions",
    )
    parser.add_argument(
        "-l", "--length",
        type=int,
        default=4096,
        help="Sequence length",
    )

    def benchmark(state_dim: int,
                  signal_dim: int,
                  sequence_length: int,
                 ):
        dtype = torch.complex64
        device = torch.device("cuda")

        a0, a1, b0, b1 = [torch.randn(signal_dim, 1, state_dim, dtype=dtype, device=device) for _ in range(4)]
        g = torch.randn(sequence_length, dtype=dtype, device=device)
        Lambda = torch.randn(state_dim, dtype=dtype, device=device)

        naive_ks = None
        pykeops_ks = None
        print("State dimensions: ", state_dim)
        print("Signal dimensions:", signal_dim)
        print("Sequence length:  ", sequence_length)
        print("Benchmarking...")
        print()

        # For some reason running PyKeOps before naive (or without naive) improves the
        # performance of PyKeOps. The performance of naive method is not affected by order.
        try:
            start = time.time()
            pykeops_ks = pykeops_cauchy_kernel(a0, a1, b0, b1, g, Lambda)
            end = time.time()
            print("PYKEOPS:", end - start, "seconds")
        except RuntimeError as e:
            print("PYKEOPS method failed!")
            print("Error:", e)

        try:
            start = time.time()
            naive_ks = naive_cauchy_kernel(a0, a1, b0, b1, g, Lambda)
            end = time.time()
            print("NAIVE:  ", end - start, "seconds")
        except RuntimeError as e:
            print()
            print("NAIVE method failed!")
            print("Error:", e)

        print()
        if naive_ks is not None and pykeops_ks is not None:
            for a, b in zip(naive_ks, pykeops_ks):
                assert type(a) == type(b)
                print("Max difference:", (a-b).abs().max().item())

    args = parser.parse_args()
    benchmark(
        args.state_dim,
        args.signal_dim,
        args.length,
    )
