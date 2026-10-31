"""Adaptive random-walk Metropolis-Hastings sampler (paper section 3.2).

Paper settings:
- 2 independent chains
- 10,000 iterations per chain
- 5,000 warmup iterations
- 5,000 retained draws per chain (10,000 pooled draws)
- Parameter estimates = posterior mean of retained draws
- Trace plots over all iterations (Figure 4b)
- Split R-hat and acceptance rate monitoring
"""
from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np


@dataclass
class ChainOutput:
    draws: np.ndarray        # Retained post-warmup draws (n_draws, d)
    warmup: np.ndarray       # Warmup draws (n_warmup, d)
    full_draws: np.ndarray   # Full history (n_iter, d) for trace plots
    acceptance_rate: float   # Post-warmup acceptance rate
    warmup_acceptance: float # Warmup acceptance rate


def metropolis(
    logp: Callable[[np.ndarray], float],
    x0: Sequence[float],
    n_iter: int = 10000,
    n_warmup: int = 5000,
    rng: np.random.Generator = None,
) -> ChainOutput:
    """Run one MCMC chain using adaptive random-walk Metropolis-Hastings.

    Args:
        logp: Log-posterior function theta -> log p(theta | data).
        x0: Initial parameter vector.
        n_iter: Total iterations (warmup + retained).
        n_warmup: Number of warmup/burn-in iterations.
        rng: NumPy random Generator.

    Returns:
        ChainOutput with draws, warmup draws, full history, and acceptance rates.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.array(x0, dtype=float)
    d = len(x)
    lp = logp(x)

    # If initial logp is non-finite, attempt small random perturbations to find valid support
    if not np.isfinite(lp):
        for _ in range(50):
            cand = x + 0.05 * rng.standard_normal(d)
            cand_lp = logp(cand)
            if np.isfinite(cand_lp):
                x, lp = cand, cand_lp
                break
        if not np.isfinite(lp):
            raise ValueError(f"Starting point {x0} has non-finite log-posterior {lp}")

    # Initial proposal covariance scaled by Gelman-Roberts-Gilks factor
    cov = np.diag((0.1 * np.maximum(np.abs(x), 0.5)) ** 2)
    L = np.linalg.cholesky(cov)
    scale = 2.38 / np.sqrt(d)

    full_draws = np.empty((n_iter, d), dtype=float)
    acc_window = 0
    acc_warmup_total = 0
    acc_post_total = 0

    window_size = max(50, min(100, n_warmup // 4)) if n_warmup >= 100 else max(10, n_warmup // 2)
    cov_interval = max(100, min(500, n_warmup // 2)) if n_warmup >= 200 else max(20, n_warmup)

    for i in range(n_iter):
        step = scale * (L @ rng.standard_normal(d))
        y = x + step
        lq = logp(y)

        # Metropolis acceptance probability alpha = min(1, exp(lq - lp))
        if np.isfinite(lq) and np.log(rng.random()) < (lq - lp):
            x, lp = y, lq
            acc_window += 1
            if i < n_warmup:
                acc_warmup_total += 1
            else:
                acc_post_total += 1

        full_draws[i] = x

        # Adaptation during warmup only (preserves ergodicity post-warmup)
        if i < n_warmup:
            if (i + 1) % window_size == 0:
                rate = acc_window / float(window_size)
                scale *= float(np.clip(np.exp((rate - 0.234) * 0.8), 0.5, 2.0))
                acc_window = 0

            # Re-estimate proposal covariance structure from empirical warmup trajectory
            if (i + 1) % cov_interval == 0 and i >= window_size:
                history = full_draws[max(0, i - 2 * cov_interval) : i + 1]
                if np.all(history.std(axis=0) > 1e-8):
                    emp_cov = np.cov(history.T)
                    reg_cov = np.atleast_2d(emp_cov) + 1e-8 * np.eye(d)
                    try:
                        L = np.linalg.cholesky(reg_cov)
                    except np.linalg.LinAlgError:
                        pass

    n_post = max(1, n_iter - n_warmup)
    n_warm = max(1, n_warmup)
    post_acc_rate = float(acc_post_total / n_post)
    warm_acc_rate = float(acc_warmup_total / n_warm)

    return ChainOutput(
        draws=full_draws[n_warmup:],
        warmup=full_draws[:n_warmup],
        full_draws=full_draws,
        acceptance_rate=post_acc_rate,
        warmup_acceptance=warm_acc_rate,
    )
