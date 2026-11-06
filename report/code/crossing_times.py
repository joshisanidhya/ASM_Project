def crossing_times(vehicle: np.ndarray, t: np.ndarray, pos: np.ndarray, lane: np.ndarray, x0: float):
    """Time at which each vehicle first crosses the cross-section ``pos == x0``.

    Rows must be sorted by (vehicle, t). A crossing is a pair of consecutive samples of the same
    vehicle with ``pos`` on opposite sides of ``x0``; the crossing time is interpolated linearly
    between them and the lane is the lane of the later sample. A vehicle is counted once, at its
    first crossing. Returns (vehicle id, crossing time, lane) arrays.
    """
    same = vehicle[1:] == vehicle[:-1]
    a, b = pos[:-1] - x0, pos[1:] - x0
    cross = same & (((a < 0) & (b >= 0)) | ((a > 0) & (b <= 0)))
    i = np.flatnonzero(cross)
    frac = (-a[i]) / (b[i] - a[i])
    tc = t[i] + frac * (t[i + 1] - t[i])
    v, ln = vehicle[i + 1], lane[i + 1]
    first = np.ones(len(i), dtype=bool)
    first[1:] = v[1:] != v[:-1]  # i is increasing, so a vehicle's crossings are adjacent
    return v[first], tc[first], ln[first]
