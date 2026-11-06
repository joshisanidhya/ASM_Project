"""Table 1 PDF likelihoods and priors, sampled with Metropolis-Hastings (section 3.2).

Defaults: two chains, 10,000 iterations, 5,000 warmup, pooled posterior means.
The 49 histogram bins are for evaluation. Repeated observations are counted exactly,
without rounding or subsampling, to reduce the cost of the observation likelihood.
"""
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, gammaln, log_expit

from .distributions import ALPHA, T_MAX, DISTS, Dist
from .mcmc import metropolis
from .parallel import pmap


class FitResult(tuple):
    """Point estimates and chains; also supports ``theta, draws = result``."""

    def __new__(cls, theta, draws, full_draws=None, acceptance_rates=None):
        inst = super().__new__(cls, (theta, draws))
        inst.theta = theta
        inst.draws = draws
        inst.full_draws = full_draws if full_draws is not None else draws
        inst.acceptance_rates = list(acceptance_rates or [])
        return inst


def make_logpost(dist: Dist, data, frequencies=None):
    """log prior + sum_i log f(h_i | theta), using the PDFs in paper Table 1.

    No upper truncation normalizer is added: the paper filters data at 25 s but lists
    unconditioned baseline PDFs and a proposed PDF normalized on [0.5, infinity).
    It does not publish its likelihood code; this is the direct PDF-based interpretation.
    ``frequencies`` optionally holds exact multiplicities of sorted unique observations.
    """
    if frequencies is None:
        x, weights = np.unique(np.asarray(data, dtype=float), return_counts=True)
    else:
        x, weights = np.asarray(data, dtype=float), np.asarray(frequencies, dtype=float)
    n = float(weights.sum())
    total = float(np.dot(weights, x))
    log_x = np.log(x)
    sum_log_x = float(np.dot(weights, log_x))
    prefix_n = np.r_[0.0, np.cumsum(weights)]
    prefix_x = np.r_[0.0, np.cumsum(weights * x)]
    tmin = float(x[0])

    def logpost(theta):
        if not dist.check_params(theta):
            return -np.inf
        lp = dist.logprior(theta, tmin)
        if not np.isfinite(lp):
            return -np.inf
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            if dist.name == "Proposed":
                a, b = theta
                rate = -np.log(b)
                if a <= ALPHA:
                    ll = n * np.log(rate) - rate * (total - n * ALPHA)
                else:
                    i = np.searchsorted(x, a, side="right")
                    absolute_sum = total - 2 * prefix_x[i] + a * (2 * prefix_n[i] - n)
                    ll = n * (np.log(rate) - np.log(2 - np.exp(-rate * (a - ALPHA)))) - rate * absolute_sum
            elif dist.name == "Shifted Log-normal":
                mu, sigma, shift = theta
                logs = np.log(x - shift)
                ll = -n * np.log(sigma * np.sqrt(2 * np.pi)) - np.dot(weights, logs + 0.5 * ((logs - mu) / sigma) ** 2)
            elif dist.name == "Weibull":
                shape, scale = theta
                ll = n * (np.log(shape) - shape * np.log(scale)) + (shape - 1) * sum_log_x
                ll -= np.dot(weights, np.exp(shape * (log_x - np.log(scale))))
            elif dist.name == "Log-logistic":
                shape, scale = theta
                ll = n * (np.log(shape) - shape * np.log(scale)) + (shape - 1) * sum_log_x
                ll -= 2 * np.dot(weights, np.logaddexp(0, shape * (log_x - np.log(scale))))
            elif dist.name == "Gamma":
                shape, rate = theta
                ll = n * (shape * np.log(rate) - gammaln(shape)) + (shape - 1) * sum_log_x - rate * total
            elif dist.name == "Burr":
                shape, tail, scale = theta
                ll = n * (np.log(shape) + np.log(tail) - shape * np.log(scale)) + (shape - 1) * sum_log_x
                ll -= (tail + 1) * np.dot(weights, np.logaddexp(0, shape * (log_x - np.log(scale))))
            elif dist.name == "Shifted Exponential":
                rate, shift = theta
                ll = n * np.log(rate) - rate * (total - n * shift)
            else:
                raise ValueError(f"No PDF likelihood for {dist.name}")
        value = lp + ll
        return float(value) if np.isfinite(value) else -np.inf

    return logpost


@dataclass
class ParameterTransform:
    """Log / logit coordinates avoid hard boundaries without changing the posterior."""

    bounds: Sequence

    def forward(self, theta):
        z = np.array(theta, dtype=float)
        for j, (lo, hi) in enumerate(self.bounds):
            if np.isfinite(lo) and np.isfinite(hi):
                z[j] = np.log(theta[j] - lo) - np.log(hi - theta[j])
            elif np.isfinite(lo):
                z[j] = np.log(theta[j] - lo)
        return z

    def inverse(self, z):
        theta = np.array(z, dtype=float)
        jacobian = 0.0
        with np.errstate(over="ignore", invalid="ignore"):
            for j, (lo, hi) in enumerate(self.bounds):
                if np.isfinite(lo) and np.isfinite(hi):
                    # Subtract from the nearer boundary to retain precision near min(headway).
                    theta[j] = hi - (hi - lo) * expit(-z[j]) if z[j] > 0 else lo + (hi - lo) * expit(z[j])
                    jacobian += np.log(hi - lo) + log_expit(z[j]) + log_expit(-z[j])
                elif np.isfinite(lo):
                    theta[j] = lo + np.exp(z[j])
                    jacobian += z[j]
        return theta, float(jacobian)


def _target(name, values, frequencies):
    dist = DISTS[name]
    bounds = [(lo, float(values[0]) if p == "gamma" else hi)
              for p, (lo, hi) in zip(dist.params, dist.bounds)]
    transform = ParameterTransform(bounds)
    logpost = make_logpost(dist, values, frequencies)

    def transformed(z):
        theta, log_jacobian = transform.inverse(z)
        value = logpost(theta) + log_jacobian
        return value if np.isfinite(value) else -np.inf

    return transformed, transform


def _initial_state(task):
    """Optimization initializes MH only; estimates always come from posterior draws."""
    name, values, frequencies = task
    logp, transform = _target(name, values, frequencies)
    n = frequencies.sum()
    mean = float(np.dot(values, frequencies) / n)
    median = values[np.searchsorted(np.cumsum(frequencies), n / 2)]
    theta = DISTS[name].init(median, mean)
    if name == "Shifted Exponential":
        rate = 1 / max(mean - values[0], 1e-6)
        theta = (rate, values[0] - min(1 / (n * rate), (values[0] + 10) / 2))
    z = transform.forward(theta)

    def objective(y):
        lp = logp(y)
        return -lp if np.isfinite(lp) else 1e300

    res = minimize(objective, z, method="Nelder-Mead",
                   options={"xatol": 1e-7, "fatol": 1e-7, "maxiter": 5000})
    if np.isfinite(logp(res.x)) and objective(res.x) < objective(z):
        z = res.x

    # Local curvature scales proposals for million-observation fits and narrow shift posteriors.
    d, step = len(z), 1e-3
    directions = np.eye(d) * step
    hessian = np.empty((d, d))
    for i in range(d):
        for j in range(d):
            ei, ej = directions[i], directions[j]
            hessian[i, j] = (objective(z + ei + ej) - objective(z + ei - ej)
                             - objective(z - ei + ej) + objective(z - ei - ej)) / (4 * step ** 2)
    if np.all(np.isfinite(hessian)):
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        floor = max(1e-6, float(np.max(np.abs(eigenvalues))) * 1e-8)
        cov = (eigenvectors * (1 / np.maximum(eigenvalues, floor))) @ eigenvectors.T
    else:
        cov = np.eye(d) * 0.01
    return z, cov


def _chain(task):
    name, values, frequencies, initial, chain_idx, n_iter, n_warmup, seed = task
    logp, transform = _target(name, values, frequencies)
    rng = np.random.default_rng([seed, chain_idx, list(DISTS).index(name)])
    z, cov = initial
    start = z.copy()
    for _ in range(20):
        candidate = z + 2 * (np.linalg.cholesky(cov) @ rng.standard_normal(len(z)))
        if np.isfinite(logp(candidate)):
            start = candidate
            break
    output = metropolis(logp, start, n_iter, n_warmup, rng, initial_cov=cov)
    output.full_draws = np.array([transform.inverse(row)[0] for row in output.full_draws])
    output.warmup = output.full_draws[:n_warmup]
    output.draws = output.full_draws[n_warmup:]
    return output


def fit_all(data, n_iter=10000, n_warmup=5000, chains=2, seed=42, names=None, workers=None):
    """Fit observed headways using Table 1 PDFs, priors and posterior-mean estimates."""
    data = np.asarray(data, dtype=float)
    if (data.ndim != 1 or len(data) < 2 or not np.all(np.isfinite(data))
            or np.any(data < ALPHA) or np.any(data > T_MAX)):
        raise ValueError("Expected at least two finite headways in [0.5, 25] seconds")
    if chains < 1 or not 0 <= n_warmup <= n_iter - 4:
        raise ValueError("Need at least one chain and four retained draws; 0 <= warmup <= iters - 4")
    values, frequencies = np.unique(data, return_counts=True)
    names = list(DISTS if names is None else names)
    starts = pmap(_initial_state, [(name, values, frequencies) for name in names], workers)
    tasks = [(name, values, frequencies, start, ch, n_iter, n_warmup, seed)
             for name, start in zip(names, starts) for ch in range(chains)]
    outputs = pmap(_chain, tasks, workers)
    fits = {}
    for i, name in enumerate(names):
        selected = outputs[i * chains:(i + 1) * chains]
        draws = np.stack([c.draws for c in selected])
        fits[name] = FitResult(draws.mean(axis=(0, 1)), draws,
                              np.stack([c.full_draws for c in selected]),
                              [c.acceptance_rate for c in selected])
    return fits
