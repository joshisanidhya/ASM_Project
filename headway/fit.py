"""Posterior over distribution parameters given binned headway counts."""
import numpy as np
from scipy.optimize import minimize

from .distributions import CENTERS, Dist
from .mcmc import metropolis


def make_logpost(dist: Dist, counts, tmin):
    """Binned (multinomial) log-posterior.

    ponytail: likelihood uses the 49 half-second bin counts, not the raw samples,
    so a step costs 49 CDF evals instead of n. Exact enough for the paper's 0.5 s
    resolution; use raw samples if sub-bin shape ever matters.
    """
    def logpost(theta):
        lp = dist.logprior(theta, tmin)
        if not np.isfinite(lp):
            return -np.inf
        with np.errstate(all="ignore"):
            p = dist.raw_bin_probs(theta)
        if not np.all(np.isfinite(p)) or np.any(p < 0):
            return -np.inf
        return lp + float(counts @ np.log(np.maximum(p, 1e-300)))
    return logpost


def fit(dist: Dist, counts, tmin, n_iter=10000, n_warmup=5000, chains=2, seed=0):
    """Return posterior-mean parameters (paper: mean of the post-warmup draws)."""
    cdf = np.cumsum(counts) / counts.sum()
    median, mean = CENTERS[np.searchsorted(cdf, 0.5)], CENTERS @ counts / counts.sum()
    logpost = make_logpost(dist, counts, tmin)

    # MCMC chains start at the MAP so 5000 warmup draws only have to tune the proposal.
    nll = lambda th: -v if np.isfinite(v := logpost(th)) else 1e300
    x0 = minimize(nll, dist.init(median, mean), method="Nelder-Mead",
                  options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000}).x

    rng = np.random.default_rng(seed)
    draws = np.vstack([metropolis(logpost, x0, n_iter, n_warmup, rng) for _ in range(chains)])
    return draws.mean(axis=0), draws
