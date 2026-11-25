import unittest

import torch

from src.ovg_e import decode_gated, rear_base


class EGeometryTest(unittest.TestCase):
    def test_zero_gate_returns_exact_ovg_and_small_gate_converges(self):
        front = torch.tensor([[[[.318745, 1.217345]]]])
        back = (.45*front).log()
        residual = torch.full((1,2,1,2), .75)
        scales = torch.full_like(residual, .08)
        front_logit = torch.full_like(front, -15.)
        last = float("inf")
        for gate in (1e-3, 1e-4, 1e-5):
            logits = torch.full_like(front, torch.logit(torch.tensor(gate)).item())
            out = decode_gated(front,back,torch.ones_like(front),residual,scales,front_logit,logits)
            error = (out["decoded"]-front).abs().max().item()
            self.assertGreater(error, 0.)
            self.assertLess(error, last)
            self.assertFalse(out["selected_back"].any())
            last = error
        out = decode_gated(front,back,torch.ones_like(front),residual,scales,front_logit,
                           torch.full_like(front, -torch.inf))
        self.assertTrue(torch.equal(out["decoded"],front))
        self.assertTrue(torch.equal(out["log_mass"][:,1:].exp(),torch.zeros_like(front)))

    def test_source_specific_margins_and_triangulation_priority(self):
        front = torch.ones((1,1,1,4))
        rear = .7 * front
        prior = torch.tensor([[[[1.,1.,1.,1.]], [[.35,.60,.35,.60]], [[.85,.85,.96,.96]]]])
        base,source,valid = rear_base(front,rear,prior,torch.ones_like(prior))
        self.assertEqual(source.flatten().tolist(),[2,2,1,0])
        torch.testing.assert_close(base.exp().flatten(),torch.tensor([.85,.85,.35,.70]))
        self.assertTrue(valid.all())


if __name__ == "__main__":
    unittest.main()
