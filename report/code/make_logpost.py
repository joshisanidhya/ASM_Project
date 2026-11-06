def make_logpost(dist: Dist, counts: np.ndarray, tmin: float, likelihood: str = "paper") -> Callable[[Sequence[float]], float]:
    """Construct the binned multinomial log-posterior function.

    ln p(theta | counts) = ln p(theta) + sum_{k=1}^49 O_k ln(p_k(theta))
    where p_k(theta) is the model probability in bin k.

    likelihood="paper" uses the distribution's own (un-truncated) density, as the paper does, so
    the bin probabilities sum to less than one. likelihood="truncated" conditions on the observed
    window [0.5, 25] s (they sum to one), which is the exact likelihood for a sample that was
    truncated to that window. Under it the shifted exponential's threshold is not identifiable
    (the conditional law does not depend on it) and its posterior is just the prior.
    """
    if likelihood not in LIKELIHOODS:
        raise ValueError(f"likelihood must be one of {LIKELIHOODS}")
    bin_probs = dist.raw_bin_probs if likelihood == "paper" else dist.bin_probs

    def logpost(theta: Sequence[float]) -> float:
        lp = dist.logprior(theta, tmin)
        if not np.isfinite(lp):
            return -np.inf
        if not dist.check_params(theta):
            return -np.inf
        with np.errstate(all="ignore"):
            p = bin_probs(theta)
        if not np.all(np.isfinite(p)) or np.any(p < 0.0) or p.sum() <= 0.0:
            return -np.inf
        # Multinomial log-likelihood over the 49 bins
        safe_p = np.maximum(p, 1e-300)
        return lp + float(np.dot(counts, np.log(safe_p)))

    return logpost
