"""The proposed headway distribution f(t)=b^|t-a| and the six baselines (paper Table 1).

Every distribution is a CDF plus a log-prior. Likelihood, KS, chi-square, KL and
Wasserstein all derive from the CDF, so nothing else is needed per distribution.
All CDFs are renormalised on the observed window [ALPHA, T_MAX].
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.special import gammainc, ndtr

ALPHA = 0.5   # minimum headway [s], paper section 3.1
T_MAX = 25.0  # maximum headway [s], paper section 4
EDGES = np.arange(ALPHA, T_MAX + 0.25, 0.5)  # 49 bins of 0.5 s
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2

NEG_INF = -np.inf


# --- priors (log-density up to a constant) -------------------------------
def _normal(x, sd=10.0):
    return -0.5 * (x / sd) ** 2


def _gamma(x, shape=0.5, rate=0.5):
    return (shape - 1) * np.log(x) - rate * x if x > 0 else NEG_INF


def _uniform(x, lo, hi):
    return 0.0 if lo < x < hi else NEG_INF


# --- unnormalised CDFs ---------------------------------------------------
def _proposed(t, a, b):
    """Antiderivative of b^|s-a| (continuous at s=a); differences give eq. (2)."""
    L = np.log(b)
    t = np.asarray(t, dtype=float)

    def A(s):
        with np.errstate(over="ignore"):  # np.where evaluates both branches
            return np.where(s <= a, -b ** (a - s) / L, (b ** (s - a) - 2) / L)
    # normalise on [ALPHA, inf): total area is A(inf) - A(ALPHA), A(inf) = -2/L  (eq. 12-13)
    lo = A(np.array(ALPHA))
    return (A(t) - lo) / (-2 / L - lo)


def _slognorm(t, mu, sigma, gamma):
    x = np.maximum(t - gamma, 1e-300)
    return np.where(t > gamma, ndtr((np.log(x) - mu) / sigma), 0.0)


def _weibull(t, alpha, beta):
    return 1 - np.exp(-((t / beta) ** alpha))


def _loglogistic(t, alpha, beta):
    return 1 / (1 + (t / beta) ** -alpha)


def _gamma_cdf(t, alpha, beta):  # beta is a rate
    return gammainc(alpha, beta * t)


def _burr(t, alpha, beta, lam):
    return 1 - (1 + (t / lam) ** alpha) ** -beta


def _shexp(t, lam, gamma):
    return np.where(t >= gamma, 1 - np.exp(-lam * np.maximum(t - gamma, 0)), 0.0)


@dataclass(frozen=True)
class Dist:
    name: str
    params: tuple
    cdf: Callable          # (t, *theta) -> F(t), unnormalised
    logprior: Callable     # (theta, tmin) -> float
    init: Callable         # (median, mean) -> starting theta

    def raw_bin_probs(self, theta):
        """Model mass per bin under the paper's pdf (baselines: natural, un-truncated).

        Used for the likelihood. Truncating here would make the exponential's shift
        unidentifiable, since a shifted exponential is memoryless."""
        return np.diff(self.cdf(EDGES, *theta))

    def trunc_cdf(self, t, theta):
        """CDF conditional on the observed window [ALPHA, T_MAX] (for KS, chi-square, KL, W)."""
        lo, hi = self.cdf(np.array([ALPHA, T_MAX]), *theta)
        t = np.clip(t, ALPHA, T_MAX)
        with np.errstate(all="ignore"):
            return (self.cdf(t, *theta) - lo) / (hi - lo)

    def bin_probs(self, theta):
        return np.diff(self.trunc_cdf(EDGES, theta))


def _gam3(*xs):
    return sum(_gamma(x) for x in xs)


DISTS = {d.name: d for d in [
    Dist("Proposed", ("a", "b"), _proposed,
         lambda th, tm: _normal(th[0]) + _uniform(th[1], 0, 1),
         lambda med, mean: (med, 0.5)),
    Dist("Shifted Log-normal", ("mu", "sigma", "gamma"), _slognorm,
         lambda th, tm: _normal(th[0]) + _gamma(th[1]) + _uniform(th[2], -10, tm),
         lambda med, mean: (np.log(max(med - 0.3, 0.1)), 0.7, 0.3)),
    Dist("Weibull", ("alpha", "beta"), _weibull,
         lambda th, tm: _gam3(*th),
         lambda med, mean: (1.5, mean)),
    Dist("Log-logistic", ("alpha", "beta"), _loglogistic,
         lambda th, tm: _gam3(*th),
         lambda med, mean: (2.5, med)),
    Dist("Gamma", ("alpha", "beta"), _gamma_cdf,
         lambda th, tm: _gam3(*th),
         lambda med, mean: (2.0, 2.0 / mean)),
    Dist("Burr", ("alpha", "beta", "lambda"), _burr,
         lambda th, tm: _gam3(*th),
         lambda med, mean: (3.0, 1.0, med)),
    Dist("Shifted Exponential", ("lambda", "gamma"), _shexp,
         lambda th, tm: _gamma(th[0]) + _uniform(th[1], -10, tm),
         lambda med, mean: (1 / max(mean - 0.4, 0.1), 0.4)),
]}
