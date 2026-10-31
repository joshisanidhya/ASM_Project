"""Stochastic simulation module for synthetic headway generation and validation.

Generates synthetic headway observations from fitted distributions on the observation
interval [ALPHA, T_MAX] = [0.5, 25.0] seconds, evaluates summary statistics, and validates
simulated samples against both empirical data and theoretical continuous distributions.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from .distributions import ALPHA, CENTERS, EDGES, T_MAX, Dist

_FINE_GRID = np.linspace(ALPHA, T_MAX, 50_001)


@dataclass
class SimulationSummary:
    distribution: str
    sample_size: int
    mean: float
    std: float
    median: float
    iqr: float
    q05: float
    q25: float
    q75: float
    q95: float
    min_val: float
    max_val: float
    skewness: float
    kurtosis: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "distribution": self.distribution,
            "sample_size": self.sample_size,
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "iqr": self.iqr,
            "q05": self.q05,
            "q25": self.q25,
            "q75": self.q75,
            "q95": self.q95,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
        }


def sample_distribution(
    dist: Dist,
    theta: Sequence[float],
    size: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate synthetic headway observations from a fitted distribution on [ALPHA, T_MAX].

    Uses high-resolution inverse CDF transform sampling with strict bounds enforcement.
    """
    if size <= 0:
        return np.array([], dtype=float)
    rng = np.random.default_rng(seed)

    # High-resolution monotonic inverse-CDF lookup
    cdf_grid = dist.trunc_cdf(_FINE_GRID, theta)
    if not np.all(np.isfinite(cdf_grid)) or cdf_grid[-1] <= 0.0:
        raise ValueError(f"Invalid CDF for {dist.name} with theta={theta}")

    # Monotonic accumulation ensures numerical stability
    u_grid = np.maximum.accumulate(cdf_grid)
    u_grid[-1] = 1.0
    u_grid[0] = 0.0

    uniforms = rng.random(size)
    samples = np.interp(uniforms, u_grid, _FINE_GRID)
    return np.clip(samples, ALPHA, T_MAX)


def compute_summary_statistics(samples: np.ndarray, name: str = "Distribution") -> SimulationSummary:
    """Compute comprehensive descriptive statistics for a headway sample."""
    arr = np.asarray(samples, dtype=float)
    if len(arr) == 0:
        return SimulationSummary(
            distribution=name,
            sample_size=0,
            mean=0.0,
            std=0.0,
            median=0.0,
            iqr=0.0,
            q05=0.0,
            q25=0.0,
            q75=0.0,
            q95=0.0,
            min_val=0.0,
            max_val=0.0,
            skewness=0.0,
            kurtosis=0.0,
        )

    q05, q25, q50, q75, q95 = np.percentile(arr, [5.0, 25.0, 50.0, 75.0, 95.0])
    return SimulationSummary(
        distribution=name,
        sample_size=len(arr),
        mean=float(np.mean(arr)),
        std=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        median=float(q50),
        iqr=float(q75 - q25),
        q05=float(q05),
        q25=float(q25),
        q75=float(q75),
        q95=float(q95),
        min_val=float(np.min(arr)),
        max_val=float(np.max(arr)),
        skewness=float(stats.skew(arr)),
        kurtosis=float(stats.kurtosis(arr)),
    )


def compare_empirical_and_simulated(
    empirical: np.ndarray,
    simulated: np.ndarray,
) -> Dict[str, Any]:
    """Compare empirical observations against a simulated sample."""
    emp = np.asarray(empirical, dtype=float)
    sim = np.asarray(simulated, dtype=float)

    ks_res = stats.ks_2samp(emp, sim)
    w_dist = float(stats.wasserstein_distance(emp, sim))

    emp_stats = compute_summary_statistics(emp, name="Empirical")
    sim_stats = compute_summary_statistics(sim, name="Simulated")

    return {
        "ks_statistic": float(ks_res.statistic),
        "ks_pvalue": float(ks_res.pvalue),
        "wasserstein_distance": w_dist,
        "mean_diff": abs(emp_stats.mean - sim_stats.mean),
        "median_diff": abs(emp_stats.median - sim_stats.median),
        "std_diff": abs(emp_stats.std - sim_stats.std),
        "empirical_stats": emp_stats.to_dict(),
        "simulated_stats": sim_stats.to_dict(),
    }


def validate_simulated_against_theoretical(
    dist: Dist,
    theta: Sequence[float],
    simulated: np.ndarray,
) -> Dict[str, Any]:
    """Validate that simulated observations statistically conform to the theoretical distribution."""
    sim = np.asarray(simulated, dtype=float)
    n = len(sim)

    # 1-sample KS test against exact continuous CDF
    ks_res = stats.kstest(sim, lambda t: dist.trunc_cdf(t, theta))

    # Binned Chi-Square test
    sim_counts, _ = np.histogram(sim, bins=EDGES)
    expected_p = dist.bin_probs(theta)
    exp_counts = n * expected_p

    # Pool low counts
    valid = exp_counts >= 5.0
    if valid.sum() > len(theta) + 1:
        o = sim_counts[valid]
        e = exp_counts[valid]
        chi2_stat = float(np.sum((o - e) ** 2 / e))
        chi2_dof = int(len(o) - 1 - len(theta))
        chi2_p = float(stats.chi2.sf(chi2_stat, chi2_dof))
    else:
        chi2_stat, chi2_dof, chi2_p = np.nan, 0, np.nan

    # Binned Wasserstein against model probabilities
    sim_p = sim_counts / max(1, n)
    w_dist = float(stats.wasserstein_distance(CENTERS, CENTERS, sim_p, expected_p))

    return {
        "theory_ks_D": float(ks_res.statistic),
        "theory_ks_p": float(ks_res.pvalue),
        "theory_chi2": chi2_stat,
        "theory_chi2_p": chi2_p,
        "theory_wasserstein": w_dist,
        "valid": bool(ks_res.pvalue > 0.01),
    }
