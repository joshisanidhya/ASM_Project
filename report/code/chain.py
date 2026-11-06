def _chain(task: Tuple[str, np.ndarray, float, np.ndarray, int, int, int, int, str]) -> ChainOutput:
    """Worker task executing one independent MCMC chain."""
    name, counts, tmin, x0, chain_idx, n_iter, n_warmup, seed, likelihood = task
    dist = DISTS[name]
    logpost = make_logpost(dist, counts, tmin, likelihood)
    rng = np.random.default_rng([seed, chain_idx, 100])

    # Chain 0 starts at MAP; Chain 1+ starts slightly perturbed to verify convergence
    start = np.array(x0, dtype=float)
    if chain_idx > 0:
        perturb = 1.0 + 0.05 * (1.0 if chain_idx % 2 == 1 else -1.0)
        perturbed_start = start * perturb
        if np.isfinite(logpost(perturbed_start)):
            start = perturbed_start

    i = GAP_PARAM.get(name)
    if i is None:
        return metropolis(logpost, start, n_iter=n_iter, n_warmup=n_warmup, rng=rng)

    # Sample v = log(min t - gamma); log p(v) = log p(gamma) + v is the exact change of variables.
    logpost_z = lambda z: logpost(_from_z(z, i, tmin)) + z[i]
    out = metropolis(logpost_z, _to_z(start, i, tmin), n_iter=n_iter, n_warmup=n_warmup, rng=rng)
    return replace(out, draws=_from_z(out.draws, i, tmin), warmup=_from_z(out.warmup, i, tmin),
                   full_draws=_from_z(out.full_draws, i, tmin))
