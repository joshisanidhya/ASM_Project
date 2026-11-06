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
