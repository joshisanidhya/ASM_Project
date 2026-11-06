def window_count_summary(gaps: pd.DataFrame, window: float = 60.0, min_windows: int = 5) -> Dict[str, float]:
    """Observed variability of vehicle counts in consecutive windows against renewal and Poisson theory.

    For every (unit, lane) the arrivals are counted in consecutive windows of ``window`` seconds, from the
    first arrival to the last. The index of dispersion (variance/mean) of the counts is compared with
    the renewal prediction sigma^2/mu^2 computed from the *same series'* inter-arrival times, and with 1
    (Poisson). Series are averaged with weights equal to their number of windows. Flow that drifts during a
    recording inflates the observed dispersion, which is why the result is a bound and not an exact test.
    """
    rows = []
    for (unit, lane), s in gaps.groupby(["unit", "lane"]):
        t = s["t"].to_numpy()
        t = np.concatenate([[t[0] - s["gap"].iloc[0]], t])  # first arrival, recovered from the first gap
        n_win = int((t[-1] - t[0]) // window)
        if n_win < min_windows or len(s) < 10:
            continue
        c = np.histogram(t, bins=t[0] + window * np.arange(n_win + 1))[0]
        gp = s["gap"].to_numpy()
        if c.mean() > 0:
            rows.append((n_win, c.mean(), c.var(ddof=1), gp.var(ddof=1) / gp.mean() ** 2, window / gp.mean()))
    r = np.array(rows)
    w = r[:, 0]
    fano_obs = float(np.average(r[:, 2] / r[:, 1], weights=w))
    return {
        "window_s": window, "n_series": len(r), "n_windows": int(w.sum()),
        "mean_count": float(np.average(r[:, 1], weights=w)),
        "fano_observed": fano_obs,
        "fano_renewal_theory": float(np.average(r[:, 3], weights=w)),
        "fano_poisson": 1.0,
        "mean_count_renewal": float(np.average(r[:, 4], weights=w)),
    }
