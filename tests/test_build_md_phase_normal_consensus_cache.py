import unittest

import numpy as np

from scripts.build_md_phase_normal_consensus_cache import weighted_vector_consensus


class MdPhaseNormalConsensusTest(unittest.TestCase):
    def test_identical_replicas_keep_target_and_confidence(self):
        values = np.array([[[1.0, -2.0]], [[1.0, -2.0]]])
        weights = np.ones((2, 1))
        mean, confidence, valid, agreement = weighted_vector_consensus(
            values, weights
        )
        np.testing.assert_allclose(mean, [[1.0, -2.0]])
        np.testing.assert_allclose(confidence, [1.0])
        np.testing.assert_array_equal(valid, [True])
        np.testing.assert_allclose(agreement, [1.0])

    def test_opposite_replicas_cancel_and_downweight_target(self):
        values = np.array([[[1.0, 0.0]], [[-1.0, 0.0]]])
        weights = np.ones((2, 1))
        mean, confidence, valid, agreement = weighted_vector_consensus(
            values, weights
        )
        np.testing.assert_allclose(mean, [[0.0, 0.0]])
        np.testing.assert_allclose(confidence, [0.0])
        np.testing.assert_array_equal(valid, [True])
        np.testing.assert_allclose(agreement, [0.0])

    def test_requires_replica_support(self):
        values = np.array([[[2.0]], [[7.0]], [[9.0]]])
        weights = np.array([[1.0], [0.0], [0.0]])
        mean, confidence, valid, _ = weighted_vector_consensus(
            values, weights, min_support_fraction=0.5, min_support_count=2
        )
        np.testing.assert_allclose(mean, [[0.0]])
        np.testing.assert_allclose(confidence, [0.0])
        np.testing.assert_array_equal(valid, [False])


if __name__ == "__main__":
    unittest.main()
