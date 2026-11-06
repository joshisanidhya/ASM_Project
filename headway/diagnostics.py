"""Convergence diagnostics and uncertainty quantification for MCMC chains.

Implements:
- Gelman-Rubin Split R-hat (Gelman et al. BDA3)
- Effective Sample Size (ESS) via initial positive sequence autocorrelation
- Bayesian Credible Intervals (95% equal-tailed intervals)
- Acceptance rate summary and convergence check flags
"""
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def split_rhat(draws: np.ndarray) -> np.ndarray:
    """Compute Split-R-hat per parameter for draws of shape (chains, n_draws, n_params).

    Splits each chain in half to detect both non-stationary drift and between-chain differences.
    Values near 1.00 (< 1.05) indicate acceptable convergence.
    """
    draws = np.asarray(draws, dtype=float)
    if draws.ndim == 2:
        draws = draws[None, ...]
    c, n, d = draws.shape
    if n < 4:
        return np.full(d, np.nan)
    h = n // 2
    # Split each chain in two: (2 * c, h, d)
    x = draws[:, : 2 * h].reshape(c, 2, h, d).reshape(2 * c, h, d)
    # Within-chain variance W
    W = x.var(axis=1, ddof=1).mean(axis=0)
    # Between-chain variance B
    B = h * x.mean(axis=1).var(axis=0, ddof=1)
    with np.errstate(all="ignore"):
        var_plus = ((h - 1.0) / h) * W + (B / h)
        rhat = np.sqrt(var_plus / W)
    # Constant chains cannot establish convergence, even if they are stuck at the same value.
    return np.where((W > 0) & np.isfinite(rhat), rhat, np.inf)


def effective_sample_size(draws: np.ndarray) -> np.ndarray:
    """Compute Effective Sample Size (ESS) per parameter across multiple chains.

    Uses paired initial-positive, monotone autocorrelations and between-chain variance.
    """
    draws = np.asarray(draws, dtype=float)
    if draws.ndim == 2:
        draws = draws[None, ...]
    c, n, d = draws.shape
    total_samples = c * n
    ess = np.zeros(d, dtype=float)

    if n < 4:
        return np.full(d, np.nan)
    for p in range(d):
        y = draws[:, :, p]
        variances = y.var(axis=1, ddof=1)
        if not np.all(np.isfinite(y)) or np.any(variances <= 0):
            continue
        within = variances.mean()
        between = n * y.mean(axis=1).var(ddof=1) if c > 1 else 0.0
        var_plus = (n - 1) / n * within + between / n
        n_fft = 2 ** int(np.ceil(np.log2(2 * n)))
        f = np.fft.rfft(y - y.mean(axis=1, keepdims=True), n=n_fft, axis=1)
        acov = np.fft.irfft(f * f.conj(), n=n_fft, axis=1)[:, :n] / n
        rho = 1 - (within - acov.mean(axis=0)) / var_plus
        rho[0] = 1.0
        pairs = rho[:2 * (n // 2)].reshape(-1, 2).sum(axis=1)
        nonpositive = np.flatnonzero(pairs <= 0)
        if len(nonpositive):
            pairs = pairs[:nonpositive[0]]
        tau = -1 + 2 * np.minimum.accumulate(pairs).sum() if len(pairs) else 1.0
        ess[p] = total_samples / max(tau, 1.0)
    return np.clip(ess, 0.0, float(total_samples))


def summarize_posterior(
    draws: np.ndarray,
    param_names: Sequence[str],
    acceptance_rate: float = np.nan,
    rhat_thresh: float = 1.05,
) -> List[Dict[str, Any]]:
    """Compute detailed summary table for parameter posterior distributions.

    Includes mean, std, 2.5%, 50%, 97.5% quantiles, R-hat, ESS, and convergence warnings.
    """
    draws = np.asarray(draws, dtype=float)
    if draws.ndim == 2:
        draws = draws[None, ...]
    c, n, d = draws.shape
    flat = draws.reshape(-1, d)
    rhat = split_rhat(draws)
    ess = effective_sample_size(draws)

    summaries = []
    for j, name in enumerate(param_names):
        col = flat[:, j]
        q025, q50, q975 = np.percentile(col, [2.5, 50.0, 97.5])
        is_converged = bool(rhat[j] < rhat_thresh)
        warning = "" if is_converged else f"R-hat={rhat[j]:.3f} >= {rhat_thresh}"

        summaries.append(
            {
                "param": name,
                "mean": float(np.mean(col)),
                "sd": float(np.std(col, ddof=1)),
                "median": float(q50),
                "q025": float(q025),
                "q975": float(q975),
                "rhat": float(rhat[j]),
                "ess": float(ess[j]),
                "acceptance_rate": float(acceptance_rate),
                "converged": is_converged,
                "warning": warning,
            }
        )
    return summaries
