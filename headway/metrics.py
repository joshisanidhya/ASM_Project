"""Goodness-of-fit as specified in paper sections 3.4-3.5: KS (eq. 14), chi-square (eq. 15),
KL divergence (eq. 16), Wasserstein distance (eq. 17)."""
import numpy as np
from scipy.stats import chi2, ks_2samp

from .distributions import ALPHA, T_MAX, Dist

_GRID = np.linspace(ALPHA, T_MAX, 50_001)


def _merge_bins(obs, exp, min_count=5):
    """Paper 3.4: if a bin's observed frequency is < 5, group it with nearby bins until >= 5."""
    o, e, ao, ae = [], [], 0.0, 0.0
    for oi, ei in zip(obs, exp):
        ao, ae = ao + oi, ae + ei
        if ao >= min_count:
            o.append(ao); e.append(ae); ao = ae = 0.0
    if ao or ae:  # leftover tail joins the last group
        if o:
            o[-1] += ao; e[-1] += ae
        else:
            o.append(ao); e.append(ae)
    return np.array(o), np.array(e)


def sample_model(dist: Dist, theta, n, seed=0):
    """Draw n headways from the fitted distribution (inverse CDF on the observed window)."""
    cdf = dist.trunc_cdf(_GRID, theta)
    if not np.all(np.isfinite(cdf)):
        return None
    return np.interp(np.random.default_rng(seed).random(n), cdf, _GRID)


def evaluate(dist: Dist, theta, data, counts):
    n = counts.sum()
    p_model = dist.bin_probs(theta)
    p_obs = counts / n

    # KS (eq. 14): field sample Q vs a sample P drawn from the fitted distribution
    sim = sample_model(dist, theta, len(data))
    ks = ks_2samp(data, sim) if sim is not None else None

    # Chi-square (eq. 15) on the binned data
    o, e = _merge_bins(counts, n * p_model)
    dof = len(o) - 1 - len(theta)
    stat = ((o - e) ** 2 / e).sum()
    chi_p = chi2.sf(stat, dof) if dof > 0 else np.nan

    # KL divergence (eq. 16), observed P against model Q over the bins
    m = p_obs > 0
    kl = float((p_obs[m] * np.log(p_obs[m] / np.maximum(p_model[m], 1e-300))).sum())

    # Wasserstein (eq. 17), p=1: mean |X_(i) - Y_(i)| over the sorted observed and model
    # bin-probability vectors. The paper does not say what X and Y are; this reading is the
    # one whose magnitudes (~1e-3, Table 5) fit. It ignores where each bin sits on the time axis.
    w = float(np.mean(np.abs(np.sort(p_obs) - np.sort(p_model))))

    return {"ks_D": ks.statistic if ks else np.nan, "ks_p": ks.pvalue if ks else np.nan,
            "chi2": stat, "chi2_dof": dof, "chi2_p": chi_p, "kl": kl, "wasserstein": w}
