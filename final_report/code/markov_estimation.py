@classmethod
def from_headways(cls, headways: np.ndarray, groups: Optional[np.ndarray] = None) -> "TrafficRegimeMarkovChain":
    h = np.asarray(headways, dtype=float)
    # Discretize into 3 physical car-following regimes
    # 0: Close-Following [0.5, 1.5), 1: Stable Platoon [1.5, 4.0), 2: Free-Flow [4.0, inf)
    states = np.digitize(h, [1.5, 4.0])  # returns 0, 1, 2
    states = np.clip(states, 0, 2)

    # Count 1-step transitions
    n_states = 3
    ok = np.ones(len(states) - 1, dtype=bool) if groups is None else (np.asarray(groups)[:-1] == np.asarray(groups)[1:])
    counts = np.bincount((states[:-1] * 3 + states[1:])[ok], minlength=9).reshape(3, 3).astype(float)

    # Regularize zero rows with uniform transitions
    row_sums = counts.sum(axis=1, keepdims=True)
    P = np.where(row_sums > 0, counts / row_sums, np.eye(n_states))
    return cls(P)
