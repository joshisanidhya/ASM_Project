"""Posterior parameter estimation via Markov Chain Monte Carlo (paper section 3.2).

Paper settings:
- 2 independent chains
- 10,000 iterations per chain
- 5,000 warmup iterations
- 5,000 retained posterior draws per chain
- Parameter estimate = mean of all retained posterior draws
- Random walk Metropolis-Hastings
"""
from collections import namedtuple
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from .distributions import CENTERS, DISTS, Dist
from .mcmc import ChainOutput, metropolis
from .parallel import pmap


class FitResult(tuple):
    """Fit output for one distribution.

    Behaves as a 2-tuple (theta, draws) for full backward compatibility,
    while providing attributes: theta, draws, full_draws, acceptance_rates.
    """
    def __new__(cls, theta: np.ndarray, draws: np.ndarray, full_draws: np.ndarray = None, acceptance_rates: Sequence[float] = None):
        inst = super().__new__(cls, (theta, draws))
        inst.theta = theta
        inst.draws = draws
        inst.full_draws = full_draws if full_draws is not None else draws
        inst.acceptance_rates = list(acceptance_rates or [])
        return inst


def make_logpost(dist: Dist, counts: np.ndarray, tmin: float) -> Callable[[Sequence[float]], float]:
    """Construct the binned multinomial log-posterior function.

    ln p(theta | counts) = ln p(theta) + sum_{k=1}^49 O_k ln(p_k(theta))
    where p_k(theta) is the model probability in bin k.
    """
    def logpost(theta: Sequence[float]) -> float:
        lp = dist.logprior(theta, tmin)
        if not np.isfinite(lp):
            return -np.inf
        if not dist.check_params(theta):
            return -np.inf
        with np.errstate(all="ignore"):
            p = dist.raw_bin_probs(theta)
        if not np.all(np.isfinite(p)) or np.any(p < 0.0) or p.sum() <= 0.0:
            return -np.inf
        # Multinomial log-likelihood over the 49 bins
        safe_p = np.maximum(p, 1e-300)
        return lp + float(np.dot(counts, np.log(safe_p)))

    return logpost


def _map_start(task: Tuple[str, np.ndarray, float]) -> np.ndarray:
    """Find Maximum A Posteriori (MAP) estimate via Nelder-Mead to initialize MCMC."""
    name, counts, tmin = task
    dist = DISTS[name]
    logpost = make_logpost(dist, counts, tmin)
    total_counts = float(counts.sum())
    cdf = np.cumsum(counts) / total_counts
    median = float(CENTERS[np.searchsorted(cdf, 0.5)])
    mean = float(np.dot(CENTERS, counts) / total_counts)
    init_guess = dist.init(median, mean)

    def nll(th):
        val = logpost(th)
        return -val if np.isfinite(val) else 1e300

    res = minimize(
        nll,
        init_guess,
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000},
    )
    return res.x if res.success and np.isfinite(logpost(res.x)) else np.array(init_guess, dtype=float)


def _chain(task: Tuple[str, np.ndarray, float, np.ndarray, int, int, int, int]) -> ChainOutput:
    """Worker task executing one independent MCMC chain."""
    name, counts, tmin, x0, chain_idx, n_iter, n_warmup, seed = task
    dist = DISTS[name]
    logpost = make_logpost(dist, counts, tmin)
    rng = np.random.default_rng([seed, chain_idx, 100])

    # Chain 0 starts at MAP; Chain 1+ starts slightly perturbed to verify convergence
    start = np.array(x0, dtype=float)
    if chain_idx > 0:
        perturb = 1.0 + 0.05 * (1.0 if chain_idx % 2 == 1 else -1.0)
        perturbed_start = start * perturb
        if np.isfinite(logpost(perturbed_start)):
            start = perturbed_start

    return metropolis(logpost, start, n_iter=n_iter, n_warmup=n_warmup, rng=rng)


def fit_all(
    counts: np.ndarray,
    tmin: float,
    n_iter: int = 10000,
    n_warmup: int = 5000,
    chains: int = 2,
    seed: int = 42,
    names: Optional[Sequence[str]] = None,
    workers: Optional[int] = None,
) -> Dict[str, FitResult]:
    """Fit all specified distributions via parallel MCMC.

    Args:
        counts: Binned headway counts (49 bins).
        tmin: Minimum observed headway in dataset.
        n_iter: Total iterations per chain (paper: 10,000).
        n_warmup: Warmup iterations per chain (paper: 5,000).
        chains: Number of independent chains (paper: 2).
        seed: Random seed for reproducibility.
        names: Distribution names to fit (defaults to all 7).
        workers: Number of parallel worker processes.

    Returns:
        Dict mapping distribution name to FitResult(theta, draws, full_draws, acceptance_rates).
    """
    names = list(names or DISTS.keys())
    x0s = pmap(_map_start, [(n, counts, tmin) for n in names], workers)

    tasks = [
        (n, counts, tmin, x0, c, n_iter, n_warmup, seed)
        for n, x0 in zip(names, x0s)
        for c in range(chains)
    ]
    chain_outputs: List[ChainOutput] = pmap(_chain, tasks, workers)

    out = {}
    for i, n in enumerate(names):
        chain_slice = chain_outputs[i * chains : (i + 1) * chains]
        post_draws = np.stack([co.draws for co in chain_slice])        # (chains, n_post, d)
        full_draws = np.stack([co.full_draws for co in chain_slice])    # (chains, n_iter, d)
        acc_rates = [co.acceptance_rate for co in chain_slice]
        # Paper: point estimate is the mean of all post-warmup draws across both chains
        flat = post_draws.reshape(-1, post_draws.shape[-1])
        theta = flat.mean(axis=0)
        out[n] = FitResult(theta=theta, draws=post_draws, full_draws=full_draws, acceptance_rates=acc_rates)

    return out
