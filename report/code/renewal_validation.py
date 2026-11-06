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
