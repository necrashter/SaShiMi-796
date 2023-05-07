import unittest
from __init__ import *


class TestS4Components(unittest.TestCase):
    def test_HiPPO_NPLR_DPLR(self):
        """
        Check whether the NPLR and DPLR representations of HiPPO are equivalent.
        """
        A2, P, _ = init_NPLR_HiPPO(8)
        Lambda, V, Pc, _ = init_DPLR_HiPPO(8)
        Vc = V.conj().T
        P = P.unsqueeze(1)
        Pc = Pc.unsqueeze(1)
        Lambda = torch.diag(Lambda)

        A2 = A2.to(torch.complex64)
        A3 = V @ Lambda @ Vc - (P @ P.T)
        A4 = V @ (Lambda - Pc @ Pc.conj().T) @ Vc
        self.assertTrue(torch.allclose(A2, A3, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(A2, A4, atol=1e-5, rtol=1e-5))

    def test_conv_kernel_DPLR(self):
        """
        Test whether conv_kernel_DPLR is equivalent to conv_kernel_naive.
        """
        L = 16
        Lambda, _, P, B = init_DPLR_HiPPO(4)

        # Discretize HiPPO
        A = torch.diag(Lambda) - P.unsqueeze(1) @ P.unsqueeze(1).conj().T
        C = torch.randn(1, 4, dtype=torch.complex64)
        Ab, Bb, Cb = discretize_SSM(A, B, C, 1.0 / L)

        # Naive convolution
        a = conv_kernel_naive(Ab, Bb, Cb.conj(), L)

        # Compare to the DPLR generating function approach.
        C = (torch.eye(4) - torch.linalg.matrix_power(Ab, L)).conj().T @ Cb.flatten()
        b = conv_kernel_DPLR(Lambda, P, P, B, C, step=1.0 / L, L=L)
        self.assertTrue(torch.allclose(a.real, b.real, atol=1e-5, rtol=1e-5))

    def test_conv_and_recurrent(self):
        """
        Test whether the result of convolution is equivalent to the recurrent method.
        """
        L = 16
        Lambda, _, P, B = init_DPLR_HiPPO(4)

        # Discretize HiPPO
        A = torch.diag(Lambda) - P.unsqueeze(1) @ P.unsqueeze(1).conj().T
        C = torch.randn(1, 4, dtype=torch.complex64)
        Ab, Bb, Cb = discretize_SSM(A, B, C, 1.0 / L)

        # Recurrent method
        u = torch.randn(L)
        recurrent_out = run_recurrent_SSM(Ab, Bb, Cb.conj(), u.unsqueeze(1))[1]

        # Convolutional method
        # Naive kernel
        K_naive = conv_kernel_naive(Ab, Bb, Cb.conj(), L)
        # DPLR convolution kernel
        C = (torch.eye(4) - torch.linalg.matrix_power(Ab, L)).conj().T @ Cb.flatten()
        K = conv_kernel_DPLR(Lambda, P, P, B, C, step=1.0 / L, L=L)
        self.assertTrue(torch.allclose(K_naive.real, K, atol=1e-5, rtol=1e-5))
        conv_out = convolve(u, K)

        self.assertTrue(torch.allclose(recurrent_out.real, conv_out.real, atol=1e-5, rtol=1e-5))


if __name__ == '__main__':
    unittest.main()
