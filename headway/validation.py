"""Quantitative validation of the stochastic models (course brief, section 6).

Three questions, each answered with numbers and not by eye:

1. Does the simulated renewal process agree with renewal theory?       -> renewal_validation
2. Is the renewal assumption (independent headways) reasonable?       -> independence_check
3. Is the regime Markov chain a good model of the regime sequence?    -> markov_validation

All of them take the headways in the order the loader returns them. For highD and NGSIM that is
file order, i.e. consecutive 1 Hz samples of one follower, with an occasional jump to the next
vehicle at a vehicle boundary (a few percent of the transitions, not removed).
"""
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, poisson

REGIME_CUTS = [1.5, 4.0]  # Close-Following < 1.5 s <= Stable Platoon < 4.0 s <= Free-Flow


def regime_states(headways: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(np.asarray(headways, float), REGIME_CUTS), 0, 2)


# --- 1. renewal process: simulation vs theory -----------------------------------------------
def simulate_counts(headways: np.ndarray, times: Sequence[float], n_paths: int = 2000,
                    seed: int = 0) -> np.ndarray:
    """N(t) for n_paths independent renewal paths: headways bootstrapped i.i.d. from the data.

    N(t) = max{n : S_n <= t} with S_n = H_1 + ... + H_n. Returns (n_paths, len(times)).
    """
    h = np.asarray(headways, float)
    t_max = max(times)
    k = int(np.ceil(2 * t_max / h.mean())) + 50  # about twice the arrivals needed, plus a buffer
    arrivals = np.cumsum(np.random.default_rng(seed).choice(h, size=(n_paths, k)), axis=1)
    if (arrivals[:, -1] <= t_max).any():
        raise RuntimeError("not enough simulated arrivals to cover t_max")
    return np.stack([(arrivals <= t).sum(axis=1) for t in times], axis=1)


def renewal_validation(headways: np.ndarray, times: Sequence[float] = (30, 60, 120, 300),
                       n_paths: int = 2000, seed: int = 0) -> Tuple[pd.DataFrame, np.ndarray]:
    """Simulated moments and distribution of N(t) against renewal theory.

    With mu = E[H], sigma^2 = Var(H) the standard results are
        E[N(t)]   ~ t/mu + (sigma^2 - mu^2) / (2 mu^2)
        Var[N(t)] ~ sigma^2 t / mu^3,           N(t) approximately Normal.
    A Poisson process of the same rate has Var = E; the ratio Var/E tends to sigma^2/mu^2.
    ks_normal / ks_poisson are max |F_sim - F_model| for the count distribution.
    """
    h = np.asarray(headways, float)
    mu, s2 = h.mean(), h.var(ddof=1)
    counts = simulate_counts(h, times, n_paths, seed)
    rows = []
    for j, t in enumerate(times):
        c = counts[:, j]
        m_th, v_th = t / mu + (s2 - mu ** 2) / (2 * mu ** 2), s2 * t / mu ** 3
        k = np.arange(c.min(), c.max() + 1)
        f_sim = np.array([(c <= x).mean() for x in k])
        rows.append({
            "t_s": t,
            "mean_sim": c.mean(), "mean_theory": m_th, "mean_rel_err": abs(c.mean() - m_th) / m_th,
            "var_sim": c.var(ddof=1), "var_theory": v_th, "var_rel_err": abs(c.var(ddof=1) - v_th) / v_th,
            "var_poisson": t / mu,
            "fano_sim": c.var(ddof=1) / c.mean(), "fano_theory": s2 / mu ** 2,
            "ks_normal": float(np.abs(f_sim - norm.cdf((k + 0.5 - m_th) / np.sqrt(v_th))).max()),
            "ks_poisson": float(np.abs(f_sim - poisson.cdf(k, t / mu)).max()),
        })
    return pd.DataFrame(rows), counts


# --- 2. is the renewal (independent headways) assumption reasonable? --------------------------
def independence_check(headways: np.ndarray, max_lag: int = 5) -> pd.DataFrame:
    """Autocorrelation of the headway sequence at lags 1..max_lag against the white-noise band.

    For independent headways r_k ~ N(0, 1/n), so |r_k| > 1.96 / sqrt(n) is significant.
    """
    x = np.asarray(headways, float)
    x = x - x.mean()
    denom = (x ** 2).sum()
    band = 1.96 / np.sqrt(len(x))
    return pd.DataFrame([{"lag": k, "autocorr": float((x[:-k] * x[k:]).sum() / denom),
                          "band95": band} for k in range(1, max_lag + 1)])


# --- 3. Markov chain validation --------------------------------------------------------------
def _transition_counts(s: np.ndarray, lag: int = 1) -> np.ndarray:
    return np.bincount(s[:-lag] * 3 + s[lag:], minlength=9).reshape(3, 3).astype(float)


def _row_normalise(c: np.ndarray) -> np.ndarray:
    r = c.sum(axis=1, keepdims=True)
    return np.where(r > 0, c / np.maximum(r, 1), np.eye(3))


def markov_validation(headways: np.ndarray, n_sim: int = 200_000, seed: int = 0) -> Dict[str, float]:
    """Does a first-order, time-homogeneous 3-state chain describe the regime sequence?

    * chapman_kolmogorov_n: max |empirical n-step matrix - P^n|   (exact for a Markov chain)
    * stationary_*: stationary pi against the observed regime fractions
    * sim_*: a simulated long path against pi, and P re-estimated from it against P
    * second_order_*: likelihood-ratio test of first order against second order, plus its effect
      size (average total-variation distance between P(.|i,j) and P(.|j)). With very large n the
      test rejects for tiny effects, so read the effect size too.
    """
    s = regime_states(headways)
    P = _row_normalise(_transition_counts(s))
    out: Dict[str, float] = {"n_transitions": float(len(s) - 1)}

    for n in (2, 3, 5):
        diff = np.abs(_row_normalise(_transition_counts(s, n)) - np.linalg.matrix_power(P, n))
        out[f"chapman_kolmogorov_{n}_max_err"] = float(diff.max())

    A = np.vstack([P.T - np.eye(3), np.ones(3)])
    pi = np.clip(np.linalg.lstsq(A, np.append(np.zeros(3), 1.0), rcond=None)[0], 0, 1)
    pi /= pi.sum()
    frac = np.bincount(s, minlength=3) / len(s)
    out["stationary_max_abs_diff"] = float(np.abs(pi - frac).max())
    out["stationary_tv_distance"] = float(0.5 * np.abs(pi - frac).sum())

    rng = np.random.default_rng(seed)
    cum, u = np.cumsum(P, axis=1), rng.random(n_sim)
    sim = np.empty(n_sim, dtype=int)
    sim[0] = 0
    for i in range(1, n_sim):
        sim[i] = min(int(np.searchsorted(cum[sim[i - 1]], u[i], side="right")), 2)
    out["sim_occupation_tv_distance"] = float(0.5 * np.abs(np.bincount(sim, minlength=3) / n_sim - pi).sum())
    out["sim_P_reestimate_max_err"] = float(np.abs(_row_normalise(_transition_counts(sim)) - P).max())

    n3 = np.bincount(s[:-2] * 9 + s[1:-1] * 3 + s[2:], minlength=27).reshape(3, 3, 3).astype(float)
    n_ij, n_jk, n_j = n3.sum(2), n3.sum(0), n3.sum((0, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        g2 = 2 * np.nansum(np.where(n3 > 0, n3 * np.log(n3 * n_j[None, :, None]
                                                         / (n_ij[:, :, None] * n_jk[None, :, :])), 0.0))
        cond_ij = n3 / n_ij[:, :, None]
        cond_j = n_jk / n_j[:, None]
        tv = 0.5 * np.nansum(np.abs(cond_ij - cond_j[None, :, :]), axis=2)
    out["second_order_g2"] = float(g2)
    out["second_order_dof"] = 12.0
    out["second_order_p"] = float(chi2.sf(g2, 12))
    out["second_order_effect_tv"] = float((np.nan_to_num(tv) * n_ij).sum() / n_ij.sum())
    return out
