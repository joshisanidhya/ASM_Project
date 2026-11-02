"""The proposed headway distribution f(t) = b^|t-a| and six baselines (paper Table 1).

Implements:
1. Proposed General Exponential: g(t) = b^|t-a|, normalized on [ALPHA, inf)
2. Shifted Log-normal: (t - gamma) ~ LogNormal(mu, sigma)
3. Weibull: standard 2-parameter Weibull(alpha, beta)
4. Log-logistic (Fisk): 2-parameter Log-logistic(alpha, beta)
5. Gamma: 2-parameter Gamma(alpha, beta) with shape alpha, rate beta
6. Burr Type XII: 3-parameter Burr(alpha, beta, lambda)
7. Shifted Exponential: (t - gamma) ~ Exp(lambda)

All distributions provide:
- Exact PDF f(t | theta) and CDF F(t | theta) on natural support
- Truncated PDF and CDF conditioned on the empirical window [ALPHA, T_MAX] = [0.5, 25.0] s
- Exact analytical or numerically verified bin probabilities over 49 intervals of 0.5 s width
- Prior log-densities matching paper Table 1
- Parameter bounds and validation
"""
from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np
from scipy.special import gamma as gamma_fn, gammainc, ndtr

ALPHA = 0.5   # Minimum headway threshold [s], paper section 3.1
T_MAX = 25.0  # Maximum headway threshold [s], paper section 4
BIN_WIDTH = 0.5
EDGES = np.arange(ALPHA, T_MAX + 0.25, BIN_WIDTH)  # 49 bins: 0.5 to 25.0 s
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2.0
NUM_BINS = len(CENTERS)  # 49

NEG_INF = -np.inf


# --- Prior log-densities (paper Table 1) -----------------------------------
def _log_normal_prior(x: float, sd: float = 10.0) -> float:
    """Normal prior N(0, sd^2) for central tendency parameters (e.g. a, mu)."""
    return -0.5 * (x / sd) ** 2


def _log_gamma_prior(x: float, shape: float = 0.5, rate: float = 0.5) -> float:
    """Gamma prior Gamma(shape, rate) with density ~ x^(shape-1) * exp(-rate * x)."""
    return (shape - 1.0) * np.log(x) - rate * x if x > 0.0 else NEG_INF


def _log_uniform_prior(x: float, lo: float, hi: float) -> float:
    """Uniform prior Uniform(lo, hi)."""
    return 0.0 if lo < x < hi else NEG_INF


# --- 1. Proposed General Exponential Distribution ---------------------------
def _proposed_pdf(t, a: float, b: float):
    """Normalized probability density function for g(t) = b^|t-a| on [ALPHA, inf).

    Paper Eq. (1), (12), (13):
        int_ALPHA^inf b^|t-a| dt = (b^(a-ALPHA) - 2) / ln(b)    if a > ALPHA
                                  -b^(ALPHA-a) / ln(b)          if a <= ALPHA
    """
    if b <= 0.0 or b >= 1.0:
        return np.zeros_like(t, dtype=float)
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    m = t >= ALPHA
    if not np.any(m):
        return out
    log_b = np.log(b)
    if a > ALPHA:
        norm = (b ** (a - ALPHA) - 2.0) / log_b
    else:
        norm = -(b ** (ALPHA - a)) / log_b
    out[m] = (b ** np.abs(t[m] - a)) / norm
    return out


def _proposed_cdf(t, a: float, b: float):
    """Piecewise closed-form CDF on [ALPHA, inf), paper Eq. (2) and Eq. (9)-(13)."""
    if b <= 0.0 or b >= 1.0:
        return np.zeros_like(t, dtype=float)
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    m = t >= ALPHA
    if not np.any(m):
        return out

    tm = t[m]
    res = np.zeros_like(tm)
    if a > ALPHA:
        denom = b ** (a - ALPHA) - 2.0
        m1 = tm <= a
        res[m1] = (b ** (a - ALPHA) - b ** (a - tm[m1])) / denom
        m2 = ~m1
        res[m2] = (b ** (a - ALPHA) + b ** (tm[m2] - a) - 2.0) / denom
    else:
        res = 1.0 - b ** (tm - ALPHA)

    out[m] = np.clip(res, 0.0, 1.0)
    return out


# --- 2. Shifted Log-normal Distribution ------------------------------------
def _slognorm_pdf(t, mu: float, sigma: float, gamma: float):
    """PDF: 1 / ((t-gamma)*sigma*sqrt(2*pi)) * exp(-(ln(t-gamma)-mu)^2 / (2*sigma^2))."""
    if sigma <= 0.0:
        return np.zeros_like(t, dtype=float)
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    m = (t > gamma) & (t > 0.0)
    if not np.any(m):
        return out
    x = t[m] - gamma
    out[m] = (1.0 / (x * sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((np.log(x) - mu) / sigma) ** 2)
    return out


def _slognorm_cdf(t, mu: float, sigma: float, gamma: float):
    """CDF: Phi((ln(t-gamma) - mu) / sigma) for t > gamma, t > 0."""
    if sigma <= 0.0:
        return np.zeros_like(t, dtype=float)
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    m = (t > gamma) & (t > 0.0)
    if not np.any(m):
        return out
    x = t[m] - gamma
    out[m] = ndtr((np.log(x) - mu) / sigma)
    return out


# --- 3. Weibull Distribution ------------------------------------------------
def _weibull_pdf(t, alpha: float, beta: float):
    """PDF: (alpha / beta) * (t / beta)^(alpha-1) * exp(-(t / beta)^alpha)."""
    if alpha <= 0.0 or beta <= 0.0:
        return np.zeros_like(t, dtype=float)
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    m = t >= 0.0
    if not np.any(m):
        return out
    tb = t[m] / beta
    with np.errstate(over="ignore", invalid="ignore"):
        val = (alpha / beta) * (tb ** (alpha - 1.0)) * np.exp(-(tb ** alpha))
    out[m] = np.nan_to_num(val, nan=0.0, posinf=0.0)
    return out


def _weibull_cdf(t, alpha: float, beta: float):
    """CDF: 1 - exp(-(t / beta)^alpha)."""
    if alpha <= 0.0 or beta <= 0.0:
        return np.zeros_like(t, dtype=float)
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    m = t >= 0.0
    if not np.any(m):
        return out
    with np.errstate(over="ignore"):
        val = 1.0 - np.exp(-((t[m] / beta) ** alpha))
    out[m] = np.clip(val, 0.0, 1.0)
    return out


# --- 4. Log-logistic Distribution (Fisk) -----------------------------------
def _loglogistic_pdf(t, alpha: float, beta: float):
    """PDF: (alpha / beta) * (t / beta)^(alpha-1) / [1 + (t / beta)^alpha]^2."""
    if alpha <= 0.0 or beta <= 0.0:
        return np.zeros_like(t, dtype=float)
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    m = t > 0.0
    if not np.any(m):
        return out
    tb = t[m] / beta
    with np.errstate(over="ignore", invalid="ignore"):
        tba = tb ** alpha
        val = (alpha / beta) * (tb ** (alpha - 1.0)) / ((1.0 + tba) ** 2)
    out[m] = np.nan_to_num(val, nan=0.0, posinf=0.0)
    return out


def _loglogistic_cdf(t, alpha: float, beta: float):
    """CDF: 1 / [1 + (t / beta)^(-alpha)]."""
    if alpha <= 0.0 or beta <= 0.0:
        return np.zeros_like(t, dtype=float)
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    m = t > 0.0
    if not np.any(m):
        return out
    with np.errstate(over="ignore", invalid="ignore"):
        val = 1.0 / (1.0 + (t[m] / beta) ** (-alpha))
    out[m] = np.nan_to_num(val, nan=0.0, posinf=0.0)
    return np.clip(out, 0.0, 1.0)


# --- 5. Gamma Distribution --------------------------------------------------
def _gamma_pdf(t, alpha: float, beta: float):
    """PDF: (beta^alpha / Gamma(alpha)) * t^(alpha-1) * exp(-beta * t). beta is rate."""
    if alpha <= 0.0 or beta <= 0.0:
        return np.zeros_like(t, dtype=float)
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    m = t > 0.0
    if not np.any(m):
        return out
    with np.errstate(over="ignore", invalid="ignore"):
        log_pdf = (
            alpha * np.log(beta)
            - np.log(gamma_fn(alpha))
            + (alpha - 1.0) * np.log(t[m])
            - beta * t[m]
        )
        val = np.exp(log_pdf)
    out[m] = np.nan_to_num(val, nan=0.0, posinf=0.0)
    return out


def _gamma_cdf(t, alpha: float, beta: float):
    """CDF: P(alpha, beta * t) using regularized lower incomplete gamma."""
    if alpha <= 0.0 or beta <= 0.0:
        return np.zeros_like(t, dtype=float)
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    m = t > 0.0
    if not np.any(m):
        return out
    out[m] = gammainc(alpha, beta * t[m])
    return np.clip(out, 0.0, 1.0)


# --- 6. Burr Type XII Distribution -----------------------------------------
def _burr_pdf(t, alpha: float, beta: float, lam: float):
    """PDF: (alpha * beta / lam) * (t / lam)^(alpha-1) * [1 + (t / lam)^alpha]^(-beta-1)."""
    if alpha <= 0.0 or beta <= 0.0 or lam <= 0.0:
        return np.zeros_like(t, dtype=float)
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    m = t > 0.0
    if not np.any(m):
        return out
    tl = t[m] / lam
    with np.errstate(over="ignore", invalid="ignore"):
        tla = tl ** alpha
        val = (alpha * beta / lam) * (tl ** (alpha - 1.0)) * ((1.0 + tla) ** (-beta - 1.0))
    out[m] = np.nan_to_num(val, nan=0.0, posinf=0.0)
    return out


def _burr_cdf(t, alpha: float, beta: float, lam: float):
    """CDF: 1 - [1 + (t / lam)^alpha]^(-beta)."""
    if alpha <= 0.0 or beta <= 0.0 or lam <= 0.0:
        return np.zeros_like(t, dtype=float)
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    m = t > 0.0
    if not np.any(m):
        return out
    with np.errstate(over="ignore", invalid="ignore"):
        val = 1.0 - (1.0 + (t[m] / lam) ** alpha) ** (-beta)
    out[m] = np.nan_to_num(val, nan=0.0, posinf=0.0)
    return np.clip(out, 0.0, 1.0)


# --- 7. Shifted Exponential Distribution ------------------------------------
def _shexp_pdf(t, lam: float, gamma: float):
    """PDF: lam * exp(-lam * (t - gamma)) for t >= gamma, t >= 0."""
    if lam <= 0.0:
        return np.zeros_like(t, dtype=float)
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    m = (t >= gamma) & (t >= 0.0)
    if not np.any(m):
        return out
    out[m] = lam * np.exp(-lam * (t[m] - gamma))
    return out


def _shexp_cdf(t, lam: float, gamma: float):
    """CDF: 1 - exp(-lam * (t - gamma)) for t >= gamma, t >= 0."""
    if lam <= 0.0:
        return np.zeros_like(t, dtype=float)
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    m = (t >= gamma) & (t >= 0.0)
    if not np.any(m):
        return out
    out[m] = 1.0 - np.exp(-lam * (t[m] - gamma))
    return np.clip(out, 0.0, 1.0)


# --- Distribution Abstraction -----------------------------------------------
@dataclass(frozen=True)
class Dist:
    name: str
    params: Tuple[str, ...]
    pdf: Callable               # (t, *theta) -> pdf
    cdf: Callable               # (t, *theta) -> cdf
    logprior: Callable          # (theta, tmin) -> log prior density
    init: Callable              # (median, mean) -> initial theta
    bounds: Sequence[Tuple[float, float]]  # Parameter bounds for MLE

    def check_params(self, theta) -> bool:
        """Validate parameter constraints."""
        if len(theta) != len(self.params):
            return False
        for (val, (lo, hi)) in zip(theta, self.bounds):
            if val is None or not np.isfinite(val) or val <= lo or val >= hi:
                return False
        return True

    def raw_bin_probs(self, theta) -> np.ndarray:
        """Model mass per bin on un-truncated CDF."""
        if not self.check_params(theta):
            return np.zeros(NUM_BINS, dtype=float)
        p = np.diff(self.cdf(EDGES, *theta))
        return np.nan_to_num(p, nan=0.0, posinf=0.0)

    def trunc_cdf(self, t, theta) -> np.ndarray:
        """CDF conditional on the observed window [ALPHA, T_MAX].

        F_trunc(t) = (F(t) - F(ALPHA)) / (F(T_MAX) - F(ALPHA))
        """
        if not self.check_params(theta):
            return np.zeros_like(t, dtype=float)
        lo, hi = self.cdf(np.array([ALPHA, T_MAX]), *theta)
        denom = hi - lo
        if denom <= 1e-12:
            return np.zeros_like(t, dtype=float)
        t_arr = np.clip(np.asarray(t, dtype=float), ALPHA, T_MAX)
        val = (self.cdf(t_arr, *theta) - lo) / denom
        return np.clip(val, 0.0, 1.0)

    def trunc_pdf(self, t, theta) -> np.ndarray:
        """PDF conditional on the observed window [ALPHA, T_MAX].

        f_trunc(t) = f(t) / (F(T_MAX) - F(ALPHA))
        """
        if not self.check_params(theta):
            return np.zeros_like(t, dtype=float)
        lo, hi = self.cdf(np.array([ALPHA, T_MAX]), *theta)
        denom = hi - lo
        if denom <= 1e-12:
            return np.zeros_like(t, dtype=float)
        t_arr = np.asarray(t, dtype=float)
        out = np.zeros_like(t_arr)
        m = (t_arr >= ALPHA) & (t_arr <= T_MAX)
        out[m] = self.pdf(t_arr[m], *theta) / denom
        return out

    def bin_probs(self, theta) -> np.ndarray:
        """Normalized probability mass for the 49 intervals in [ALPHA, T_MAX]."""
        p = np.diff(self.trunc_cdf(EDGES, theta))
        total = p.sum()
        if total > 0.0:
            p /= total
        return p


def _gam_sum(*xs):
    return sum(_log_gamma_prior(x) for x in xs)


DISTS = {
    d.name: d
    for d in [
        Dist(
            name="Proposed",
            params=("a", "b"),
            pdf=_proposed_pdf,
            cdf=_proposed_cdf,
            logprior=lambda th, tm: _log_normal_prior(th[0]) + _log_uniform_prior(th[1], 0.0, 1.0),
            init=lambda med, mean: (float(med), 0.5),
            bounds=[(-5.0, 25.0), (1e-6, 0.9999)],
        ),
        Dist(
            name="Shifted Log-normal",
            params=("mu", "sigma", "gamma"),
            pdf=_slognorm_pdf,
            cdf=_slognorm_cdf,
            logprior=lambda th, tm: _log_normal_prior(th[0])
            + _log_gamma_prior(th[1])
            + _log_uniform_prior(th[2], -10.0, tm),
            init=lambda med, mean: (float(np.log(max(med - 0.3, 0.1))), 0.7, 0.3),
            bounds=[(-5.0, 5.0), (1e-4, 10.0), (-10.0, ALPHA - 1e-4)],
        ),
        Dist(
            name="Weibull",
            params=("alpha", "beta"),
            pdf=_weibull_pdf,
            cdf=_weibull_cdf,
            logprior=lambda th, tm: _gam_sum(*th),
            init=lambda med, mean: (1.5, float(mean)),
            bounds=[(1e-4, 20.0), (1e-4, 50.0)],
        ),
        Dist(
            name="Log-logistic",
            params=("alpha", "beta"),
            pdf=_loglogistic_pdf,
            cdf=_loglogistic_cdf,
            logprior=lambda th, tm: _gam_sum(*th),
            init=lambda med, mean: (2.5, float(med)),
            bounds=[(1e-4, 20.0), (1e-4, 50.0)],
        ),
        Dist(
            name="Gamma",
            params=("alpha", "beta"),
            pdf=_gamma_pdf,
            cdf=_gamma_cdf,
            logprior=lambda th, tm: _gam_sum(*th),
            init=lambda med, mean: (2.0, float(2.0 / mean)),
            bounds=[(1e-4, 50.0), (1e-4, 50.0)],
        ),
        Dist(
            name="Burr",
            params=("alpha", "beta", "lambda"),
            pdf=_burr_pdf,
            cdf=_burr_cdf,
            logprior=lambda th, tm: _gam_sum(*th),
            init=lambda med, mean: (3.0, 1.0, float(med)),
            bounds=[(1e-4, 30.0), (1e-4, 20.0), (1e-4, 50.0)],
        ),
        Dist(
            name="Shifted Exponential",
            params=("lambda", "gamma"),
            pdf=_shexp_pdf,
            cdf=_shexp_cdf,
            logprior=lambda th, tm: _log_gamma_prior(th[0]) + _log_uniform_prior(th[1], -10.0, tm),
            init=lambda med, mean: (float(1.0 / max(mean - 0.4, 0.1)), 0.4),
            bounds=[(1e-4, 20.0), (-10.0, ALPHA - 1e-4)],
        ),
    ]
}
