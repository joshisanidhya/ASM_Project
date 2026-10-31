"""Core tests for headway distribution mathematics, normalization, and MCMC."""
import numpy as np
import pytest
from scipy.integrate import quad

from headway.distributions import ALPHA, DISTS, EDGES, T_MAX
from headway.fit import fit_all

P = DISTS["Proposed"]


def test_closed_form_matches_numerical_integration():
    # every branch of paper eq. (2): a above, inside and below the window start
    for a, b in [(0.2, 0.6), (2.0, 0.5), (6.0, 0.8)]:
        f = lambda t: b ** abs(t - a)
        total = quad(f, ALPHA, 200, points=[a] if a > ALPHA else None)[0]
        t1, t2 = 1.0, 3.0
        want = quad(f, t1, t2, points=[a] if t1 < a < t2 else None)[0] / total
        F = lambda t: P.cdf(np.array([t]), a, b)[0]
        got = (F(t2) - F(t1)) / (P.cdf(np.array([200.0]), a, b)[0] - F(ALPHA))
        assert abs(got - want) < 1e-6, (a, b, got, want)


def test_eq2_case1_literal():
    a, b, t1, t2 = 2.0, 0.6, 1.0, 3.0
    paper = (b ** (a - t1) + b ** (t2 - a) - 2) / (b ** (a - ALPHA) - 2)
    F = lambda t: P.cdf(np.array([t]), a, b)[0]
    ours = (F(t2) - F(t1)) / (P.cdf(np.array([1e4]), a, b)[0] - F(ALPHA))
    assert abs(paper - ours) < 1e-9


def test_all_seven_distributions_integrate_to_one():
    test_params = {
        "Proposed": (2.0, 0.5),
        "Shifted Log-normal": (0.5, 0.8, 0.2),
        "Weibull": (1.5, 2.5),
        "Log-logistic": (2.5, 1.8),
        "Gamma": (2.0, 1.0),
        "Burr": (3.0, 1.0, 1.5),
        "Shifted Exponential": (0.5, 0.4),
    }
    for name, theta in test_params.items():
        dist = DISTS[name]
        # Test truncated PDF integrates to 1.0 on [ALPHA, T_MAX]
        val, _ = quad(lambda t: dist.trunc_pdf(np.array([t]), theta)[0], ALPHA, T_MAX)
        assert abs(val - 1.0) < 1e-4, f"{name} truncated PDF integral is {val}, expected 1.0"

        # Test bin probabilities sum to 1.0
        p = dist.bin_probs(theta)
        assert len(p) == 49
        assert abs(p.sum() - 1.0) < 1e-6, f"{name} bin probs sum is {p.sum()}, expected 1.0"


def test_derivative_of_cdf_matches_pdf():
    dist = DISTS["Proposed"]
    theta = (2.2, 0.54)
    eps = 1e-5
    for t_val in [0.8, 1.5, 2.2, 3.5, 10.0]:
        num_deriv = (dist.cdf(np.array([t_val + eps]), *theta)[0] - dist.cdf(np.array([t_val - eps]), *theta)[0]) / (2 * eps)
        exact_pdf = dist.pdf(np.array([t_val]), *theta)[0]
        assert abs(num_deriv - exact_pdf) < 1e-3, f"Mismatch at t={t_val}: deriv={num_deriv}, pdf={exact_pdf}"


def test_mcmc_recovers_proposed_parameters():
    a, b = 2.5, 0.55
    grid = np.linspace(ALPHA, T_MAX, 20001)
    cdf = P.trunc_cdf(grid, (a, b))
    rng = np.random.default_rng(1)
    data = np.interp(rng.random(100_000), cdf, grid)
    counts = np.histogram(data, bins=EDGES)[0].astype(float)
    theta, _ = fit_all(counts, data.min(), n_iter=3000, n_warmup=1500,
                       names=["Proposed"], workers=1)["Proposed"]
    assert abs(theta[0] - a) < 0.1 and abs(theta[1] - b) < 0.02, theta


def test_parallel_fit_matches_serial():
    rng = np.random.default_rng(2)
    data = np.clip(rng.gamma(2.5, 1.2, 50_000), ALPHA, T_MAX)
    counts = np.histogram(data, bins=EDGES)[0].astype(float)
    kw = dict(n_iter=600, n_warmup=300, names=["Proposed", "Gamma"])
    ser, par = fit_all(counts, data.min(), workers=1, **kw), fit_all(counts, data.min(), workers=2, **kw)
    for n in kw["names"]:
        assert np.allclose(ser[n][0], par[n][0])
