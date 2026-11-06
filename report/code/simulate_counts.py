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
