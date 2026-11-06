"""Goodness-of-fit evaluation metrics as specified in paper sections 3.4-3.5:
- Kolmogorov-Smirnov test (Eq. 14)
- Chi-square goodness-of-fit test with low-frequency bin merging (Eq. 15)
- Kullback-Leibler divergence (Eq. 16)
- Wasserstein distance (Eq. 17)
"""
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import chi2, ks_2samp, kstest, wasserstein_distance

from .distributions import ALPHA, CENTERS, T_MAX, Dist

_GRID = np.linspace(ALPHA, T_MAX, 50_001)


def histogram_distances(counts, model_probs):
    """KL and two distinct distances on the paper's 49 bins.

    ``wasserstein`` retains the historical sorted-probability proxy for comparison
    with earlier runs. The paper does not specify its Eq. 17 inputs, so this proxy
    is an interpretation, not a verified reproduction of Table 5.
    ``wasserstein_seconds`` transports mass between headway bin centers.
    """
    p = np.asarray(counts, dtype=float) / np.sum(counts)
    q = np.asarray(model_probs, dtype=float)
    m = p > 0
    return {
        "kl": float(np.sum(p[m] * np.log(p[m] / np.maximum(q[m], 1e-300)))),
        "wasserstein": float(np.mean(np.abs(np.sort(p) - np.sort(q)))),
        "wasserstein_seconds": float(wasserstein_distance(CENTERS, CENTERS, p, q)),
    }


def merge_bins_for_chi2(obs: np.ndarray, exp: np.ndarray, min_count: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """Paper Section 3.4: Group adjacent bins until observed frequency Oi >= 5.

    This reproduces the observed-count rule; it does not guarantee sufficiently
    large expected counts for the asymptotic chi-square approximation.
    """
    o, e = [], []
    ao, ae = 0.0, 0.0
    for oi, ei in zip(obs, exp):
        ao += oi
        ae += ei
        if ao >= min_count:
            o.append(ao)
            e.append(ae)
            ao, ae = 0.0, 0.0
    if ao > 0.0 or ae > 0.0:  # Pool remaining tail mass into final bin
        if o:
            o[-1] += ao
            e[-1] += ae
        else:
            o.append(ao)
            e.append(ae)
    return np.array(o, dtype=float), np.array(e, dtype=float)


def sample_model(dist: Dist, theta: Sequence[float], n: int, seed: int = 0) -> Optional[np.ndarray]:
    """Draw n synthetic headway observations from the fitted distribution on [ALPHA, T_MAX].

    Uses inverse CDF transform over a high-resolution grid.
    """
    cdf_vals = dist.trunc_cdf(_GRID, theta)
    if not np.all(np.isfinite(cdf_vals)) or cdf_vals[-1] <= 0.0:
        return None
    # Ensure strict monotonicity for reliable interpolation
    u_vals = np.maximum.accumulate(cdf_vals)
    rng = np.random.default_rng(seed)
    uniforms = rng.random(n)
    return np.interp(uniforms, u_vals, _GRID)


def evaluate(
    dist: Dist,
    theta: Sequence[float],
    data: np.ndarray,
    counts: np.ndarray,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute all evaluation metrics between empirical data and fitted distribution.

    Returns:
        ks_D: 1-sample KS test statistic supremum |F_obs(t) - F_model(t)|
        ks_p: 1-sample KS test p-value
        ks_2samp_D: 2-sample KS test statistic against synthetic sample
        ks_2samp_p: 2-sample KS test p-value
        chi2: Chi-square statistic on merged bins
        chi2_dof: Chi-square degrees of freedom (bins - 1 - n_params)
        chi2_p: Chi-square p-value
        kl: KL divergence D_KL(P_obs || P_model) over the 49 bins
        wasserstein: legacy sorted-bin probability proxy (dimensionless; unverified Eq. 17 interpretation)
        wasserstein_seconds: Wasserstein-1 between weighted headway bin centers [seconds]
    """
    n = float(counts.sum())
    p_model = dist.bin_probs(theta)

    # The statistic is against the continuous CDF. The p-value is nominal: it does
    # not adjust for estimating parameters on these data or for serial dependence.
    try:
        ks_exact = kstest(data, lambda t: dist.trunc_cdf(t, theta))
        ks_D = float(ks_exact.statistic)
        ks_p = float(ks_exact.pvalue)
    except Exception:
        ks_D, ks_p = np.nan, np.nan

    # 2. Simulated 2-sample KS test (for empirical comparison)
    sim = sample_model(dist, theta, min(len(data), 50_000), seed=seed)
    if sim is not None and len(sim) > 0:
        ks_2 = ks_2samp(data, sim)
        ks_2samp_D = float(ks_2.statistic)
        ks_2samp_p = float(ks_2.pvalue)
    else:
        ks_2samp_D, ks_2samp_p = np.nan, np.nan

    # 3. Chi-square test (Eq. 15) with bin merging for Oi < 5
    o, e = merge_bins_for_chi2(counts, n * p_model, min_count=5.0)
    dof = len(o) - 1 - len(theta)
    # Clip expected counts from below to prevent division by zero
    safe_e = np.maximum(e, 1e-12)
    stat = float(np.sum((o - e) ** 2 / safe_e))
    chi_p = float(chi2.sf(stat, dof)) if dof > 0 else np.nan

    return {
        "ks_D": ks_D,
        "ks_p": ks_p,
        "ks_2samp_D": ks_2samp_D,
        "ks_2samp_p": ks_2samp_p,
        "chi2": stat,
        "chi2_dof": dof,
        "chi2_p": chi_p,
        **histogram_distances(counts, p_model),
    }
