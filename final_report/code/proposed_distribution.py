def _proposed_pdf(t, a: float, b: float):
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
