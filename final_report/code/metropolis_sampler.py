def metropolis(
    logp: Callable[[np.ndarray], float],
    x0: Sequence[float],
    n_iter: int = 10000,
    n_warmup: int = 5000,
    rng: np.random.Generator = None,
) -> ChainOutput:
    ...
    x = np.array(x0, dtype=float)
    d = len(x)
    lp = logp(x)
    ...
    # Initial proposal covariance scaled by Gelman-Roberts-Gilks factor
    cov = np.diag((0.1 * np.maximum(np.abs(x), 0.5)) ** 2)
    L = np.linalg.cholesky(cov)
    scale = 2.38 / np.sqrt(d)

    full_draws = np.empty((n_iter, d), dtype=float)
    acc_window = 0
    acc_warmup_total = 0
    acc_post_total = 0

    window_size = max(50, min(100, n_warmup // 4)) if n_warmup >= 100 else max(10, n_warmup // 2)
    cov_interval = max(100, min(500, n_warmup // 2)) if n_warmup >= 200 else max(20, n_warmup)

    for i in range(n_iter):
        step = scale * (L @ rng.standard_normal(d))
        y = x + step
        lq = logp(y)

        # Metropolis acceptance probability alpha = min(1, exp(lq - lp))
        if np.isfinite(lq) and np.log(rng.random()) < (lq - lp):
            x, lp = y, lq
            acc_window += 1
            if i < n_warmup:
                acc_warmup_total += 1
            else:
                acc_post_total += 1

        full_draws[i] = x

        # Adaptation during warmup only (preserves ergodicity post-warmup)
        if i < n_warmup:
            if (i + 1) % window_size == 0:
                rate = acc_window / float(window_size)
                scale *= float(np.clip(np.exp((rate - 0.234) * 0.8), 0.5, 2.0))
                acc_window = 0

            ...

    ...
