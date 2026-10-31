"""Unit and validation tests for stochastic simulation module."""
import numpy as np
import pytest

from headway.distributions import ALPHA, DISTS, T_MAX
from headway.simulation import (
    compare_empirical_and_simulated,
    compute_summary_statistics,
    sample_distribution,
    validate_simulated_against_theoretical,
)


def test_simulation_respects_observation_bounds():
    P = DISTS["Proposed"]
    theta = (1.5, 0.6)
    samples = sample_distribution(P, theta, size=5000, seed=42)

    assert len(samples) == 5000
    assert np.all(samples >= ALPHA)
    assert np.all(samples <= T_MAX)


def test_simulation_reproducibility():
    P = DISTS["Proposed"]
    theta = (2.0, 0.5)
    s1 = sample_distribution(P, theta, size=1000, seed=99)
    s2 = sample_distribution(P, theta, size=1000, seed=99)
    s3 = sample_distribution(P, theta, size=1000, seed=100)

    assert np.array_equal(s1, s2)
    assert not np.array_equal(s1, s3)


def test_simulation_summary_statistics():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = compute_summary_statistics(data, "Test")

    assert stats.sample_size == 5
    assert abs(stats.mean - 3.0) < 1e-9
    assert abs(stats.median - 3.0) < 1e-9
    assert abs(stats.min_val - 1.0) < 1e-9
    assert abs(stats.max_val - 5.0) < 1e-9


def test_simulation_validates_against_theoretical():
    for name, theta in [
        ("Proposed", (2.0, 0.5)),
        ("Weibull", (1.6, 2.8)),
        ("Log-logistic", (2.8, 2.2)),
        ("Gamma", (2.5, 1.1)),
    ]:
        dist = DISTS[name]
        samples = sample_distribution(dist, theta, size=12000, seed=42)
        val = validate_simulated_against_theoretical(dist, theta, samples)

        assert val["valid"], f"Simulation validation failed for {name}: {val}"
        assert val["theory_ks_D"] < 0.02
        assert val["theory_ks_p"] > 0.05
        assert val["theory_wasserstein"] < 0.05


def test_empirical_vs_simulated_comparison():
    dist = DISTS["Weibull"]
    theta = (1.8, 3.2)
    emp = sample_distribution(dist, theta, size=8000, seed=1)
    sim = sample_distribution(dist, theta, size=8000, seed=2)

    comp = compare_empirical_and_simulated(emp, sim)
    assert comp["ks_pvalue"] > 0.05
    assert comp["wasserstein_distance"] < 0.05
    assert comp["mean_diff"] < 0.1
