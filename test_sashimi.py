import unittest
import torch
from sashimi import *


class TestSashimiComponents(unittest.TestCase):
    def test_pooling_dimensions(self):
        """
        Check the dimensions after down-pooling and up-pooling.
        """
        x = torch.randn(1, 64, 2)
        y = DownPool(2)(x)
        self.assertEqual(list(y.size()), [1, 64 // 4, 2 * 2])
        z = UpPool(2)(y)
        self.assertEqual(list(z.size()), [1, 64, 2])
        z = CausalUpPool(2)(y)
        self.assertEqual(list(z.size()), [1, 64, 2])

    def test_pooling_recurrence_support(self):
        """
        Pooling layers don't support recurrent mode by themselves.
        """
        with self.assertRaises(TypeError):
            DownPool(2).get_recurrent_runner()
        with self.assertRaises(TypeError):
            UpPool(2).get_recurrent_runner()
        with self.assertRaises(TypeError):
            CausalUpPool(2).get_recurrent_runner()

        # Only this supports recurrent mode.
        CausalPooledResidual([torch.nn.Identity()], 2).get_recurrent_runner()

    def test_causal_pooled_residual_padding(self):
        """
        Due to shifting in causal pooling, feeding different inputs should yield the same result
        in the first block.
        """
        L = 4
        model = CausalPooledResidual([torch.nn.Identity()], 2)
        u = torch.randn(L, 2)
        o1 = model(u) - u
        u = torch.randn(L, 2)
        o2 = model(u) - u
        self.assertTrue(torch.allclose(o1, o2, atol=1e-6, rtol=1e-6))

    def test_causal_pooled_residual_conv_recurrent(self):
        """
        Test whether running the CausalPooledResidual with convolution is equivalent to recurrence.
        """
        L = 8
        model = CausalPooledResidual([torch.nn.Identity()], 2)
        u = torch.randn(L, 2)
        o = model(u)
        f = model.get_recurrent_runner()
        o2 = torch.cat([f(i.reshape(1, 1, -1)) for i in u], dim=-2)
        self.assertTrue(torch.allclose(o, o2, atol=1e-5, rtol=1e-5))

    def test_sashimi_conv_recurrent(self):
        """
        Test whether running the SaShiMi model with convolution is equivalent to recurrence.
        """
        L = 128
        model = SaShiMi(
            input_dim=2,
            hidden_dim=8,
            output_dim=2,
            state_dim=4,
            sequence_length=L,
            block_count=2,
        )
        u = torch.randn(L, 2)
        o = model(u)
        f = model.get_recurrent_runner()
        o2 = torch.cat([f(i.reshape(1, 1, -1)) for i in u], dim=-2)
        self.assertTrue(torch.allclose(o, o2, atol=1e-5, rtol=1e-5))

    def test_S4Block_dimensions(self):
        L = 16
        x = torch.randn(L, 2)

        s4block = S4Block(2, 4, L)
        y = s4block(x)
        self.assertEqual(x.size(), y.size())

        s4block = S4BlockGLU(2, 4, L)
        y = s4block(x)
        self.assertEqual(x.size(), y.size())

    def test_recurrent_runner(self):
        L = 16
        s4 = S4Block(2, 4, L)
        u = torch.randn(L, 2)
        o = s4(u)
        f = s4.get_recurrent_runner()
        o2 = torch.cat([f(i) for i in u])
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

    def test_sashimi_audio_generation(self):
        L = 64
        model = SaShiMi(
            input_dim=1,
            hidden_dim=4,
            output_dim=1,
            state_dim=4,
            sequence_length=L,
            block_count=2,
            encoder=Embedding(256, 4),
        )
        sample = generate_audio_sample(model, L, use_tqdm=False)
        self.assertEqual(list(sample.size()), [1, L])

        sample = generate_audio_sample(model, L, maxp=True, use_tqdm=False)
        self.assertEqual(list(sample.size()), [1, L])

        sample1 = generate_audio_sample(model, L, priming_signal=sample[0, :-1], maxp=True, use_tqdm=False)
        self.assertTrue(torch.allclose(sample, sample1))
        sample2 = generate_audio_sample(model, L, priming_signal=sample[0, :2], maxp=True, use_tqdm=False)
        self.assertTrue(torch.allclose(sample, sample2))

        samples = generate_audio_sample(
            model,
            L,
            batch_size=3,
            starting_input=torch.tensor([1, 2, 3]).unsqueeze(-1),
            maxp=True,
            use_tqdm=False,
        )
        sample0 = generate_audio_sample(model, L, starting_input=torch.tensor([[1]]), maxp=True, use_tqdm=False)
        sample1 = generate_audio_sample(model, L, starting_input=torch.tensor([[2]]), maxp=True, use_tqdm=False)
        sample2 = generate_audio_sample(model, L, starting_input=torch.tensor([[3]]), maxp=True, use_tqdm=False)

        self.assertTrue(torch.allclose(samples[0], sample0[0]))
        self.assertTrue(torch.allclose(samples[1], sample1[0]))
        self.assertTrue(torch.allclose(samples[2], sample2[0]))
