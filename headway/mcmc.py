"""Adaptive random-walk Metropolis-Hastings (paper section 3.2).

Paper: 2 chains x 10000 iterations, first 5000 warmup, parameter = mean of the
remaining draws. The paper uses PyMC3/Theano, which has no Python 3.13 build, so
the same Metropolis-Hastings scheme is written directly in numpy.
"""
import numpy as np


def metropolis(logp, x0, n_iter, n_warmup, rng):
    """One chain. Proposal covariance and step scale are tuned during warmup only."""
    d = len(x0)
    x, lp = np.array(x0, float), logp(x0)
    cov = np.diag((0.01 * np.maximum(np.abs(x), 0.1)) ** 2)
    L = np.linalg.cholesky(cov)
    scale = 2.38 / np.sqrt(d)
    draws = np.empty((n_iter, d))
    acc = 0
    for i in range(n_iter):
        y = x + scale * L @ rng.standard_normal(d)
        lq = logp(y)
        if np.log(rng.random()) < lq - lp:
            x, lp, acc = y, lq, acc + 1
        draws[i] = x
        if i < n_warmup and (i + 1) % 100 == 0:
            r = acc / 100
            scale *= 1.2 if r > 0.35 else 0.8 if r < 0.15 else 1.0
            acc = 0
            if (i + 1) % 500 == 0:  # re-estimate covariance from the recent window
                w = draws[max(0, i - 1000):i + 1]
                if (w.std(axis=0) > 0).all():
                    try:
                        L = np.linalg.cholesky(np.atleast_2d(np.cov(w.T)) + 1e-14 * np.eye(d))
                    except np.linalg.LinAlgError:
                        pass
    return draws[n_warmup:]
