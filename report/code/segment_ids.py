def segment_ids(vehicle: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Label each retained 1 Hz sample with the run of consecutive seconds of one vehicle it belongs to.

    A new run starts whenever the vehicle changes or a second is missing (the follower had no
    valid leader, or its headway left [0.5, 25] s, in between). Markov transitions and
    autocorrelations are computed only between samples of the same run, so no transition spans
    two vehicles or a gap in time.
    """
    v, s = np.asarray(vehicle), np.asarray(second)
    new = np.ones(len(v), dtype=bool)
    new[1:] = (v[1:] != v[:-1]) | (s[1:] != s[:-1] + 1)
    return np.cumsum(new) - 1
