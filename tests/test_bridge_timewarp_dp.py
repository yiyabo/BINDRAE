import unittest

import torch

from scripts.evaluate_stage2_bridge_timewarp import (
    force_endpoint_costs,
    monotone_dp,
)


class BridgeTimewarpDynamicProgrammingTest(unittest.TestCase):
    def test_transition_penalty_prefers_smooth_monotone_progress(self):
        cost = force_endpoint_costs(torch.zeros(4, 7, 1))
        unregularized = monotone_dp(cost)[:, 0]
        regularized = monotone_dp(cost, transition_weight=1.0)[:, 0]
        self.assertTrue(torch.equal(unregularized, torch.tensor([0, 6, 6, 6])))
        self.assertTrue(torch.equal(regularized, torch.tensor([0, 2, 4, 6])))
        self.assertTrue((regularized[1:] >= regularized[:-1]).all().item())


if __name__ == "__main__":
    unittest.main()
