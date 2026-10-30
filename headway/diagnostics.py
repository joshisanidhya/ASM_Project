"""Convergence check for the MCMC chains."""
import numpy as np


def split_rhat(draws):
    """Split-R-hat per parameter for draws of shape (chains, n, params).

    Each chain is cut in half so within-chain drift also shows up. ~1.00 means the
    chains agree; above ~1.05 they have not converged.
    """
    c, n, d = draws.shape
    h = n // 2
    x = draws[:, :2 * h].reshape(c, 2, h, d).reshape(2 * c, h, d)
    W = x.var(axis=1, ddof=1).mean(axis=0)
    B = h * x.mean(axis=1).var(axis=0, ddof=1)
    with np.errstate(all="ignore"):
        return np.sqrt(((h - 1) / h * W + B / h) / W)
