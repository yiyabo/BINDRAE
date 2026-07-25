import numpy as np

from scripts.evaluate_ca_baseline_md_reference_paths import (
    interpolate_path,
    inverse_smoothstep,
    projected_tau_from_ca,
)


def test_inverse_smoothstep_round_trip():
    tau = np.linspace(0.0, 1.0, 101)
    progress = 3.0 * tau**2 - 2.0 * tau**3
    recovered = inverse_smoothstep(progress)
    np.testing.assert_allclose(recovered, tau, atol=1.0e-8)


def test_projected_tau_recovers_residue_specific_phase():
    apo = np.zeros((2, 3), dtype=np.float64)
    holo = np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    tau = np.array(
        [[0.0, 0.0], [0.2, 0.8], [0.65, 0.35], [1.0, 1.0]],
        dtype=np.float64,
    )
    progress = 3.0 * tau**2 - 2.0 * tau**3
    path = apo[None] + progress[..., None] * (holo - apo)[None]
    recovered = projected_tau_from_ca(path, apo, holo)
    np.testing.assert_allclose(recovered, tau, atol=1.0e-8)


def test_interpolate_path_preserves_endpoints():
    path = np.array(
        [
            [[0.0, 0.0, 0.0]],
            [[0.5, 1.0, 0.0]],
            [[1.0, 2.0, 0.0]],
        ]
    )
    target_times = np.linspace(0.0, 1.0, 21)
    interpolated = interpolate_path(path, target_times)
    np.testing.assert_allclose(interpolated[0], path[0])
    np.testing.assert_allclose(interpolated[-1], path[-1])
    np.testing.assert_allclose(interpolated[10], path[1])
