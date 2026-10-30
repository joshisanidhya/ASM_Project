import numpy as np
from scipy.integrate import quad

from headway.distributions import ALPHA, DISTS, EDGES, T_MAX
from headway.fit import fit

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


def test_mcmc_recovers_proposed_parameters():
    a, b = 2.5, 0.55
    grid = np.linspace(ALPHA, T_MAX, 20001)
    cdf = P.trunc_cdf(grid, (a, b))
    rng = np.random.default_rng(1)
    data = np.interp(rng.random(100_000), cdf, grid)
    counts = np.histogram(data, bins=EDGES)[0].astype(float)
    theta, _ = fit(P, counts, data.min(), n_iter=3000, n_warmup=1500)
    assert abs(theta[0] - a) < 0.1 and abs(theta[1] - b) < 0.02, theta
