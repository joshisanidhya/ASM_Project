"""Unit tests for MCMC implementation and diagnostics."""
import numpy as np
import pytest

from headway.diagnostics import effective_sample_size, split_rhat, summarize_posterior
from headway.distributions import DISTS, EDGES
from headway.fit import fit_all
from headway.mcmc import metropolis
from headway.simulation import sample_distribution


def test_metropolis_preserves_full_history_and_acceptance():
    # Gaussian target N(2, 1)
    logp = lambda x: -0.5 * (x[0] - 2.0) ** 2
    rng = np.random.default_rng(42)
    output = metropolis(logp, [0.0], n_iter=1000, n_warmup=400, rng=rng)

    assert output.full_draws.shape == (1000, 1)
    assert output.warmup.shape == (400, 1)
    assert output.draws.shape == (600, 1)
    assert 0.10 <= output.acceptance_rate <= 0.50
    assert abs(np.mean(output.draws) - 2.0) < 0.25


def test_effective_sample_size():
    rng = np.random.default_rng(42)
    # White noise draws should have ESS close to sample size
    draws = rng.normal(size=(2, 2000, 1))
    ess = effective_sample_size(draws)
    assert ess[0] > 1000.0


def test_summarize_posterior():
    rng = np.random.default_rng(42)
    draws = rng.normal(loc=3.0, scale=0.5, size=(2, 2500, 2))
    summary = summarize_posterior(draws, ["param1", "param2"], acceptance_rate=0.25)

    assert len(summary) == 2
    assert abs(summary[0]["mean"] - 3.0) < 0.1
    assert abs(summary[0]["sd"] - 0.5) < 0.1
    assert summary[0]["converged"]
    assert summary[0]["acceptance_rate"] == 0.25
