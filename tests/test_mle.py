"""Unit tests for Maximum Likelihood Estimation (MLE) research extension."""
import numpy as np
import pytest

from headway.distributions import ALPHA, DISTS, EDGES, T_MAX
from headway.mle import fit_all_mle, fit_mle
from headway.simulation import sample_distribution


def test_mle_recovers_proposed_parameters():
    P = DISTS["Proposed"]
    true_a, true_b = 2.4, 0.52
    data = sample_distribution(P, (true_a, true_b), size=30000, seed=42)
    counts = np.histogram(data, bins=EDGES)[0].astype(float)

    res = fit_mle(P, counts, data=data, tmin=data.min())

    assert res.converged
    assert abs(res.theta[0] - true_a) < 0.15
    assert abs(res.theta[1] - true_b) < 0.04
    assert res.runtime_seconds < 2.0
    assert res.metrics["kl"] < 0.01


def test_mle_recovers_weibull_parameters():
    W = DISTS["Weibull"]
    true_alpha, true_beta = 1.6, 3.0
    data = sample_distribution(W, (true_alpha, true_beta), size=30000, seed=42)
    counts = np.histogram(data, bins=EDGES)[0].astype(float)

    res = fit_mle(W, counts, data=data, tmin=data.min())

    assert res.converged
    assert abs(res.theta[0] - true_alpha) < 0.15
    assert abs(res.theta[1] - true_beta) < 0.25
    assert res.metrics["kl"] < 0.01


def test_fit_all_mle_runs_on_all_distributions():
    P = DISTS["Proposed"]
    data = sample_distribution(P, (2.0, 0.5), size=10000, seed=42)
    counts = np.histogram(data, bins=EDGES)[0].astype(float)

    results = fit_all_mle(counts, data=data, tmin=data.min(), workers=1)

    assert len(results) == len(DISTS)
    for name, res in results.items():
        assert res.converged
        assert np.all(np.isfinite(res.theta))
        assert res.runtime_seconds > 0.0
        assert res.metrics["kl"] >= 0.0
