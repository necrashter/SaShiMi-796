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

if __name__ == '__main__':
    unittest.main()
