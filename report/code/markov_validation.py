def markov_validation(headways: np.ndarray, n_sim: int = 200_000, seed: int = 0, groups=None) -> Dict[str, float]:
    """Does a first-order, time-homogeneous 3-state chain describe the regime sequence?

    * chapman_kolmogorov_n: max |empirical n-step matrix - P^n|   (exact for a Markov chain)
    * stationary_*: stationary pi against the observed regime fractions
    * sim_*: a simulated long path against pi, and P re-estimated from it against P
    * second_order_*: likelihood-ratio test of first order against second order, plus its effect
      size (average total-variation distance between P(.|i,j) and P(.|j)). With very large n the
      test rejects for tiny effects, so read the effect size too.
    """
    s = regime_states(headways)
    P = _row_normalise(_transition_counts(s, 1, groups))
    out: Dict[str, float] = {"n_transitions": float(_transition_counts(s, 1, groups).sum())}

    for n in (2, 3, 5):
        diff = np.abs(_row_normalise(_transition_counts(s, n, groups)) - np.linalg.matrix_power(P, n))
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

    ok3 = _same_run(groups, len(s), 2)  # (i, i+2) in one run implies i+1 is too: runs are consecutive seconds
    n3 = np.bincount((s[:-2] * 9 + s[1:-1] * 3 + s[2:])[ok3], minlength=27).reshape(3, 3, 3).astype(float)
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
