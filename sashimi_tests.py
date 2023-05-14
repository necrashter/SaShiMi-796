import unittest
import torch
from sashimi import *


class TestS4Components(unittest.TestCase):
    def test_sashimi_conv_recurrent(self):
        """
        Test whether running the SaShiMi model with convolution is equivalent to recurrence.
        """
        L = 256
        model = SaShiMi(
            input_dim=2,
            hidden_dim=8,
            output_dim=2,
            state_dim=4,
            sequence_length=L,
            block_count=2,
        )
        u = torch.randn(1, L, 2)
        o = model(u)
        f = model.get_recurrent_runner()
        o2 = torch.stack([f(i) for i in u])
        self.assertTrue(torch.allclose(o, o2, atol=1e-5, rtol=1e-5))
