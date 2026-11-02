"""Maximum Likelihood Estimation (MLE) as a research extension to the paper's MCMC approach.

Estimates parameters for all 7 candidate distributions via direct numerical optimization
of the log-likelihood function and compares results against MCMC.
"""
from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .distributions import CENTERS, DISTS, Dist
from .metrics import evaluate
from .parallel import pmap


@dataclass
class MLEResult:
    name: str
    param_names: Tuple[str, ...]
    theta: np.ndarray
    log_likelihood: float
    runtime_seconds: float
    converged: bool
    message: str
    n_iterations: int
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "distribution": self.name,
            "converged": self.converged,
            "runtime_seconds": self.runtime_seconds,
            "log_likelihood": self.log_likelihood,
            "n_iterations": self.n_iterations,
        }
        for p, val in zip(self.param_names, self.theta):
            out[f"param_{p}"] = float(val)
        out.update(self.metrics)
        return out


def fit_mle(
    dist: Dist,
    counts: np.ndarray,
    data: Optional[np.ndarray] = None,
    tmin: Optional[float] = None,
) -> MLEResult:
    """Perform Maximum Likelihood Estimation for a single distribution.

    Args:
        dist: Dist instance.
        counts: Binned headway counts (49 bins).
        data: Optional raw observations array (for KS evaluation).
        tmin: Minimum observed headway.

    Returns:
        MLEResult with parameters, runtime, convergence status, log-likelihood, and metrics.
    """
    total_counts = float(counts.sum())
    cdf = np.cumsum(counts) / total_counts
    median = float(CENTERS[np.searchsorted(cdf, 0.5)])
    mean = float(np.dot(CENTERS, counts) / total_counts)
    init_guess = dist.init(median, mean)

    # Negative log-likelihood over 49 bins
    def nll(theta):
        if not dist.check_params(theta):
            return 1e300
        p = dist.bin_probs(theta)
        if not np.all(np.isfinite(p)) or np.any(p < 0.0) or p.sum() <= 0.0:
            return 1e300
        safe_p = np.maximum(p, 1e-300)
        return -float(np.dot(counts, np.log(safe_p)))

    t0 = time.perf_counter()
    res = minimize(
        nll,
        init_guess,
        method="Nelder-Mead",
        bounds=dist.bounds,
        options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-6},
    )

    # If Nelder-Mead didn't converge or reached boundary, try L-BFGS-B with adjusted step
    if not res.success or not dist.check_params(res.x):
        res_alt = minimize(
            nll,
            init_guess,
            method="L-BFGS-B",
            bounds=dist.bounds,
            options={"maxiter": 2000, "eps": 1e-4},
        )
        if res_alt.fun < res.fun and dist.check_params(res_alt.x):
            res = res_alt

    runtime = time.perf_counter() - t0
    theta = res.x
    converged = bool(res.success and dist.check_params(theta))
    ll = -float(nll(theta)) if converged else -np.inf

    # Compute goodness-of-fit metrics
    if data is not None and len(data) > 0:
        eval_data = data
    else:
        # Reconstruct pseudo-data matching bin counts if raw data is omitted
        eval_data = np.repeat(CENTERS, counts.astype(int))

    metrics = evaluate(dist, theta, eval_data, counts)

    return MLEResult(
        name=dist.name,
        param_names=dist.params,
        theta=theta,
        log_likelihood=ll,
        runtime_seconds=runtime,
        converged=converged,
        message=str(res.message),
        n_iterations=int(getattr(res, "nit", 0)),
        metrics=metrics,
    )


def _mle_task(task: Tuple[str, np.ndarray, Optional[np.ndarray], Optional[float]]) -> MLEResult:
    name, counts, data, tmin = task
    return fit_mle(DISTS[name], counts, data, tmin)


def fit_all_mle(
    counts: np.ndarray,
    data: Optional[np.ndarray] = None,
    tmin: Optional[float] = None,
    names: Optional[Sequence[str]] = None,
    workers: Optional[int] = None,
) -> Dict[str, MLEResult]:
    """Fit all specified distributions via Maximum Likelihood Estimation in parallel."""
    names = list(names or DISTS.keys())
    tasks = [(n, counts, data, tmin) for n in names]
    results: List[MLEResult] = pmap(_mle_task, tasks, workers)
    return {r.name: r for r in results}


def compare_mcmc_vs_mle(
    mcmc_params: Dict[str, Sequence[float]],
    mcmc_metrics: Dict[str, Dict[str, Any]],
    mcmc_runtimes: Dict[str, float],
    mle_results: Dict[str, MLEResult],
) -> pd.DataFrame:
    """Generate comparative summary table between MCMC and MLE."""
    rows = []
    for name in mcmc_params.keys():
        mle = mle_results.get(name)
        if mle is None:
            continue
        dist = DISTS[name]
        p_names = dist.params
        mcmc_th = mcmc_params[name]
        mle_th = mle.theta
        mcmc_m = mcmc_metrics[name]
        mle_m = mle.metrics

        param_str_mcmc = ", ".join(f"{p}={v:.3f}" for p, v in zip(p_names, mcmc_th))
        param_str_mle = ", ".join(f"{p}={v:.3f}" for p, v in zip(p_names, mle_th))

        rows.append(
            {
                "distribution": name,
                "mcmc_params": param_str_mcmc,
                "mle_params": param_str_mle,
                "mcmc_runtime_s": mcmc_runtimes.get(name, np.nan),
                "mle_runtime_s": mle.runtime_seconds,
                "runtime_speedup": (
                    mcmc_runtimes.get(name, 1.0) / max(mle.runtime_seconds, 1e-6)
                ),
                "mcmc_kl": mcmc_m["kl"],
                "mle_kl": mle_m["kl"],
                "mcmc_wasserstein": mcmc_m["wasserstein"],
                "mle_wasserstein": mle_m["wasserstein"],
                "mcmc_ks_D": mcmc_m["ks_D"],
                "mle_ks_D": mle_m["ks_D"],
                "mcmc_chi2": mcmc_m["chi2"],
                "mle_chi2": mle_m["chi2"],
                "mle_converged": mle.converged,
            }
        )
    return pd.DataFrame(rows)
