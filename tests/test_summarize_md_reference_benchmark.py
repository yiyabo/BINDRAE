import numpy as np

from scripts.summarize_md_reference_benchmark import paired_bootstrap


def test_paired_bootstrap_lower_is_better_improvement():
    primary = np.array([1.0, 2.0, 3.0])
    comparator = np.array([2.0, 3.0, 4.0])
    result = paired_bootstrap(
        primary,
        comparator,
        higher_is_better=False,
        samples=1000,
        rng=np.random.default_rng(7),
    )
    assert result["improvement"] == 1.0
    assert result["ci95_low"] == 1.0
    assert result["ci95_high"] == 1.0
    assert result["win_rate"] == 1.0


def test_paired_bootstrap_higher_is_better_improvement():
    primary = np.array([0.7, 0.8, 0.9])
    comparator = np.array([0.5, 0.6, 0.7])
    result = paired_bootstrap(
        primary,
        comparator,
        higher_is_better=True,
        samples=1000,
        rng=np.random.default_rng(11),
    )
    np.testing.assert_allclose(result["improvement"], 0.2)
    assert result["ci95_low"] > 0.0
