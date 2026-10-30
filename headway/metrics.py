"""Goodness-of-fit: KS, chi-square (eq. 14-15), KL divergence, Wasserstein (eq. 16-17)."""
import numpy as np
from scipy.stats import chi2, kstest, wasserstein_distance

from .distributions import CENTERS, Dist


def _merge_bins(obs, exp, min_count=5):
    """Merge adjacent bins until observed and expected are both >= min_count."""
    o, e, ao, ae = [], [], 0.0, 0.0
    for oi, ei in zip(obs, exp):
        ao, ae = ao + oi, ae + ei
        if ao >= min_count and ae >= min_count:
            o.append(ao); e.append(ae); ao = ae = 0.0
    if ao or ae:  # leftover tail joins the last group
        if o:
            o[-1] += ao; e[-1] += ae
        else:
            o.append(ao); e.append(ae)
    return np.array(o), np.array(e)


def evaluate(dist: Dist, theta, data, counts):
    n = counts.sum()
    p_model = dist.bin_probs(theta)
    p_obs = counts / n

    ks = kstest(data, lambda t: dist.trunc_cdf(t, theta))

    o, e = _merge_bins(counts, n * p_model)
    dof = len(o) - 1 - len(theta)
    stat = ((o - e) ** 2 / e).sum()
    chi_p = chi2.sf(stat, dof) if dof > 0 else np.nan

    m = p_obs > 0
    kl = float((p_obs[m] * np.log(p_obs[m] / np.maximum(p_model[m], 1e-300))).sum())
    w = wasserstein_distance(CENTERS, CENTERS, p_obs, p_model)

    return {"ks_D": ks.statistic, "ks_p": ks.pvalue, "chi2": stat, "chi2_dof": dof,
            "chi2_p": chi_p, "kl": kl, "wasserstein": w}
