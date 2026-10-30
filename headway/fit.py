"""Posterior over distribution parameters given binned headway counts."""
import numpy as np
from scipy.optimize import minimize

from .distributions import CENTERS, DISTS, Dist
from .mcmc import metropolis
from .parallel import pmap


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


# Tasks take distribution *names* (Dist holds lambdas, which can't be pickled to workers).
def _map_start(task):
    name, counts, tmin = task
    dist = DISTS[name]
    logpost = make_logpost(dist, counts, tmin)
    cdf = np.cumsum(counts) / counts.sum()
    median, mean = CENTERS[np.searchsorted(cdf, 0.5)], CENTERS @ counts / counts.sum()
    nll = lambda th: -v if np.isfinite(v := logpost(th)) else 1e300
    return minimize(nll, dist.init(median, mean), method="Nelder-Mead",
                    options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000}).x


def _chain(task):
    name, counts, tmin, x0, c, n_iter, n_warmup, seed = task
    logpost = make_logpost(DISTS[name], counts, tmin)
    return metropolis(logpost, x0, n_iter, n_warmup, np.random.default_rng([seed, c]))


def fit_all(counts, tmin, n_iter=10000, n_warmup=5000, chains=2, seed=0,
            names=None, workers=None):
    """{name: (posterior-mean theta, draws)} for each distribution.

    draws has shape (chains, n_iter - n_warmup, n_params), warmup already dropped.
    Chains start at the MAP so the warmup only has to tune the proposal. Every
    (distribution, chain) pair is independent, so they all run in parallel.
    """
    names = list(names or DISTS)
    x0s = pmap(_map_start, [(n, counts, tmin) for n in names], workers)
    draws = pmap(_chain, [(n, counts, tmin, x0, c, n_iter, n_warmup, seed)
                          for n, x0 in zip(names, x0s) for c in range(chains)], workers)
    out = {}
    for i, n in enumerate(names):
        d = np.stack(draws[i * chains:(i + 1) * chains])
        out[n] = (d.reshape(-1, d.shape[-1]).mean(axis=0), d)  # paper: mean of post-warmup draws
    return out
