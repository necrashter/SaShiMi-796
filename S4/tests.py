import unittest
from . layer import *
from . block import *


class TestS4Components(unittest.TestCase):
    def test_HiPPO_NPLR_DPLR(self):
        """
        Check whether the NPLR and DPLR representations of HiPPO are equivalent.
        """
        A2, P, _ = init_NPLR_HiPPO(1, 8)
        Lambda, V, Pc, _ = init_DPLR_HiPPO(1, 8)
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
        step = 1.0 / L
        Lambda, _, P, B = init_DPLR_HiPPO(1, 4)

        # Discretize HiPPO
        A = torch.diag(Lambda) - P.unsqueeze(1) @ P.unsqueeze(1).conj().T
        C = torch.randn(1, 4, dtype=torch.complex64)
        Ab, Bb, Cb = discretize_SSM(A, B, C, 1.0 / L)

        # Naive convolution
        a = conv_kernel_naive(Ab, Bb, Cb.conj(), L)

        # Compare to the DPLR generating function approach.
        C = Cb @ (torch.eye(4) - torch.linalg.matrix_power(Ab, L)).conj()
        b = conv_kernel_DPLR(Lambda, P, P, B, C, step, get_roots_of_unity(L))
        self.assertTrue(torch.allclose(a.real, b.real, atol=1e-5, rtol=1e-5))

    def test_conv_and_recurrent(self):
        """
        The result of convolution must be equivalent to the recurrent method for a generic SSM.
        """
        L = 16
        Lambda, _, P, B = init_DPLR_HiPPO(1, 4)

        # Discretize HiPPO
        A = torch.diag(Lambda) - P.unsqueeze(1) @ P.unsqueeze(1).conj().T
        C = torch.randn(1, 4, dtype=torch.complex64)
        Ab, Bb, Cb = discretize_SSM(A, B, C, 1.0 / L)

        # Generate random input
        u = torch.randn(L, 1)

        # Recurrent method
        recurrent_out = run_recurrent_SSM(Ab, Bb, Cb, u)[1]

        # Convolutional method
        K = conv_kernel_naive(Ab, Bb, Cb, L)
        conv_out = convolve(u, K)

        self.assertTrue(torch.allclose(recurrent_out.real, conv_out.real, atol=1e-5, rtol=1e-5))

    def test_conv_and_recurrent_DPLR(self):
        """
        Test whether the result of convolution is equivalent to the recurrent method in DPLR.
        """
        L = 16
        step = 1.0 / L
        Lambda, _, P, B = init_DPLR_HiPPO(1, 4)
        C = torch.randn(1, 4, dtype=torch.complex64)

        # Convolution kernel
        K = conv_kernel_DPLR(Lambda, P, P, B, C, step, get_roots_of_unity(L))

        # Recurrent form
        Ab, Bb, Cb = discretize_DPLR(Lambda, P, P, B, C, step, L)
        # Compare convolution kernel computed from recurrent form with the previous one.
        K_naive = conv_kernel_naive(Ab, Bb, Cb, L)
        self.assertTrue(torch.allclose(K_naive.real, K, atol=1e-5, rtol=1e-5))

        # Generate random input
        u = torch.randn(L, 1)
        # Test both approaches
        recurrent_out = run_recurrent_SSM(Ab, Bb, Cb, u)[1]
        conv_out = convolve(u, K)
        self.assertTrue(torch.allclose(recurrent_out.real, conv_out.real, atol=1e-5, rtol=1e-5))

    def test_S4Conv_S4Recurrent_equivalence(self):
        """
        In S4 module, convolutional_forward and recurrent_forward must be equivalent.
        """
        L = 16
        s4 = S4Base(1, 4, L)
        self.assertEqual(s4.sequence_length, L)

        # Generate random input
        u = torch.randn(L, 1)
        # Test both approaches
        conv_out = s4.convolutional_forward(u)
        recurrent_out = s4.recurrent_forward(u)
        self.assertTrue(torch.allclose(recurrent_out.real, conv_out.real, atol=1e-5, rtol=1e-5))

    def test_S4Conv_batch(self):
        """
        Processing N inputs one by one must be equivalent to processing them in a batch.
        """
        L = 16
        s4 = S4Base(1, 4, L)
        self.assertEqual(s4.sequence_length, L)

        # Generate random input
        us = [torch.randn(1, L, 1) for _ in range(4)]
        combined_out = torch.cat([s4(u) for u in us], dim=0)
        batched_out = s4(torch.cat(us, dim=0))
        self.assertTrue(torch.allclose(combined_out.real, batched_out.real, atol=1e-5, rtol=1e-5))

    def test_SSM_multi_dimensional_signal(self):
        """
        Check whether SSM works with multi dimensional signal.
        """
        L = 16
        Lambda, _, P, B = init_DPLR_HiPPO(2, 4)
        C = torch.randn(2, 4, dtype=torch.complex64)
        # Discretize HiPPO
        A = torch.diag(Lambda) - P.unsqueeze(1) @ P.unsqueeze(1).conj().T
        Ab, Bb, Cb = discretize_SSM(A, B, C, 1.0 / L)
        Bb0, Bb1 = Bb[:, 0:1], Bb[:, 1:2]
        Cb0, Cb1 = Cb[0:1, :], Cb[1:2, :]
        # Convolution kernel
        K = conv_kernel_naive(Ab, Bb, Cb, L)
        K0, K1 = K[:, 0:1], K[:, 1:2]
        K0a = conv_kernel_naive(Ab, Bb0, Cb0, L)
        K1a = conv_kernel_naive(Ab, Bb1, Cb1, L)
        self.assertTrue(torch.allclose(K0, K0a, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(K1, K1a, atol=1e-5, rtol=1e-5))

        # Generate random input
        u = torch.randn(L, 2)
        u0, u1 = u[:, 0:1], u[:, 1:2]

        co0 = convolve(u0, K0)
        co1 = convolve(u1, K1)
        co = convolve(u, K)
        self.assertTrue(torch.allclose(torch.cat([co0, co1], dim=1), co, atol=1e-5, rtol=1e-5))

        so0, ro0 = run_recurrent_SSM(Ab, Bb0, Cb0, u0)
        so1, ro1 = run_recurrent_SSM(Ab, Bb1, Cb1, u1)
        so, ro = run_recurrent_SSM(Ab, Bb, Cb, u)
        self.assertTrue(torch.allclose(ro0.real, co0, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(ro1.real, co1, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(torch.cat([ro0, ro1], dim=1), ro, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(torch.cat([so0, so1], dim=-1), so, atol=1e-5, rtol=1e-5))

        self.assertTrue(torch.allclose(co.real, ro.real, atol=1e-5, rtol=1e-5))
        self.assertEqual(co.size(), u.size())
        self.assertEqual(ro.size(), u.size())

    def test_S4_multi_dimensional_signal(self):
        """
        Check whether S4 works with multi dimensional signal.
        """
        L = 16
        step = 1.0 / L
        Lambda, _, P, B = init_DPLR_HiPPO(2, 4)
        C = torch.randn(2, 4, dtype=torch.complex64)
        # Parts
        B0, B1 = B[:, 0:1], B[:, 1:2]
        C0, C1 = C[0:1, :], C[1:2, :]

        # Convolution kernel
        K  = conv_kernel_DPLR(Lambda, P, P, B, C, step, get_roots_of_unity(L))
        K0 = conv_kernel_DPLR(Lambda, P, P, B0, C0, step, get_roots_of_unity(L))
        K1 = conv_kernel_DPLR(Lambda, P, P, B1, C1, step, get_roots_of_unity(L))
        self.assertTrue(torch.allclose(K0, K[:, 0:1], atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(K1, K[:, 1:2], atol=1e-5, rtol=1e-5))

        # Recurrent form
        Ab, Bb, Cb = discretize_DPLR(Lambda, P, P, B, C, step, L)
        Ab0, Bb0, Cb0 = discretize_DPLR(Lambda, P, P, B0, C0, step, L)
        Ab1, Bb1, Cb1 = discretize_DPLR(Lambda, P, P, B1, C1, step, L)
        self.assertTrue(torch.allclose(Ab, Ab0, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(Ab0, Ab1, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(Bb0, Bb[:, 0:1], atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(Bb1, Bb[:, 1:2], atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(Cb0, Cb[0:1, :], atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(Cb1, Cb[1:2, :], atol=1e-5, rtol=1e-5))
        # Compare convolution kernel computed from recurrent form with the previous one.
        K_naive = conv_kernel_naive(Ab, Bb, Cb, L)
        self.assertTrue(torch.allclose(K_naive.real, K, atol=1e-5, rtol=1e-5))

        # Generate random input
        u = torch.randn(L, 2)
        u0, u1 = u[:, 0:1], u[:, 1:2]

        co = convolve(u, K)
        co0 = convolve(u0, K0)
        co1 = convolve(u1, K1)
        self.assertTrue(torch.allclose(torch.cat([co0, co1], dim=1), co, atol=1e-5, rtol=1e-5))

        so0, ro0 = run_recurrent_SSM(Ab, Bb0, Cb0, u0)
        so1, ro1 = run_recurrent_SSM(Ab, Bb1, Cb1, u1)
        so, ro = run_recurrent_SSM(Ab, Bb, Cb, u)
        self.assertTrue(torch.allclose(ro0.real, co0, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(ro1.real, co1, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(torch.cat([ro0, ro1], dim=1), ro, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(torch.cat([so0, so1], dim=-1), so, atol=1e-5, rtol=1e-5))

        self.assertTrue(torch.allclose(co.real, ro.real, atol=1e-5, rtol=1e-5))
        self.assertEqual(co.size(), u.size())
        self.assertEqual(ro.size(), u.size())

    def test_S4Block_dimensions(self):
        L = 16
        s4block = S4Block(2, 4, L)
        x = torch.randn(L, 2)
        y = s4block(x)
        self.assertEqual(x.size(), y.size())

    def test_recurrent_runner(self):
        L = 16
        s4 = S4Block(2, 4, L)
        u = torch.randn(L, 2)
        o = s4(u)
        f = s4.get_recurrent_runner()
        o2 = torch.stack([f(i) for i in u])
        self.assertTrue(torch.allclose(o, o2, atol=1e-5, rtol=1e-5))

    def test_priming(self):
        """
        Tests priming and sample generation.
        """
        L = 16
        s4 = S4Block(2, 4, L)

        # Generate random sample autoregressively
        f = s4.get_recurrent_runner()
        current = torch.zeros(2)
        generated = []
        for _ in range(L):
            current = f(current)
            generated.append(current)

        # Start again
        # Prime the model with a part of the generated sample
        f = s4.get_recurrent_runner()
        for i in [torch.zeros(2)] + generated[:7]:
            current = f(i)

        # Generate the rest of the sample
        gen2 = []
        for _ in range(8):
            current = f(current)
            gen2.append(current)

        # Original generation and primed generation must yield the same result
        gen1 = torch.stack(generated[8:])
        gen2 = torch.stack(gen2)
        self.assertTrue(torch.allclose(gen1, gen2, atol=1e-6, rtol=1e-6))

    def test_S4Base_sequence_length(self):
        """
        Test the sequence length and Omega property of S4Base.
        """
        L = 16
        s4 = S4Base(1, 4, L)
        self.assertEqual(s4.sequence_length, L)
        self.assertTrue(torch.allclose(s4.Omega, get_roots_of_unity(L)))

        # Changing sequence_length shall change Omega
        L = 8
        s4.sequence_length = L
        self.assertEqual(s4.sequence_length, L)
        self.assertTrue(torch.allclose(s4.Omega, get_roots_of_unity(L)))

        u = torch.randn(16, 1)
        # Feeding an input with incorrect sequence_length shall trigger ValueError
        with self.assertRaises(ValueError):
            s4(u)
