"""Virtual-detector reconstruction of inter-arrival times (headway.detector)."""
import numpy as np
import pandas as pd

from headway import detector


def _trajectories(arrival, speed=20.0, hz=5, length=400.0, lane=1):
    """Vehicles driving at constant speed that reach x = 200 at the given times; rows sorted by (vehicle, t)."""
    rows = []
    for vid, ta in enumerate(arrival):
        t = np.arange(ta - 10.0, ta + 10.0, 1.0 / hz)
        rows.append(pd.DataFrame({"v": vid, "t": t, "x": 200.0 + speed * (t - ta), "lane": lane}))
    return pd.concat(rows, ignore_index=True)


def test_crossing_time_is_interpolated_exactly():
    arrival = np.array([13.37, 20.011, 33.9])
    d = _trajectories(arrival)
    v, tc, ln = detector.crossing_times(d.v.to_numpy(), d.t.to_numpy(), d.x.to_numpy(), d.lane.to_numpy(), 200.0)
    assert list(v) == [0, 1, 2]
    np.testing.assert_allclose(tc, arrival, atol=1e-9)  # exact for linear motion


def test_crossing_works_for_both_directions_and_counts_once():
    d = _trajectories(np.array([5.0]))
    rev = d.assign(x=400.0 - d.x)  # same crossing, now moving in the -x direction
    assert len(detector.crossing_times(rev.v.to_numpy(), rev.t.to_numpy(), rev.x.to_numpy(), rev.lane.to_numpy(), 200.0)[0]) == 1
    wobble = pd.concat([d, d.assign(t=d.t + 20.0, x=d.x[::-1].to_numpy())], ignore_index=True)  # same vehicle crosses back
    out = detector.crossing_times(wobble.v.to_numpy(), wobble.t.to_numpy(), wobble.x.to_numpy(), wobble.lane.to_numpy(), 200.0)
    assert len(out[0]) == 1


def test_vehicle_that_never_reaches_the_detector_is_not_counted():
    d = _trajectories(np.array([5.0, 9.0]))
    d.loc[d.v == 1, "x"] -= 1000.0
    v, _, _ = detector.crossing_times(d.v.to_numpy(), d.t.to_numpy(), d.x.to_numpy(), d.lane.to_numpy(), 200.0)
    assert list(v) == [0]


def test_gaps_are_taken_within_a_lane_only():
    g = detector.lane_gaps(np.zeros(5, int), np.array([1, 2, 1, 2, 1]), np.array([0.0, 1.0, 3.0, 8.0, 10.0]))
    got = g.sort_values(["lane", "t"])
    assert got.gap.tolist() == [3.0, 7.0, 7.0]  # lane 1: 3-0, 10-3; lane 2: 8-1


def test_poisson_arrivals_give_exponential_gaps_and_unit_dispersion():
    rng = np.random.default_rng(1)
    t = np.cumsum(rng.exponential(2.0, 60_000))
    g = detector.lane_gaps(np.zeros(len(t), int), np.ones(len(t), int), t)
    s = detector.gap_summary(g)
    assert abs(s["mean_s"] - 2.0) < 0.03 and abs(s["cv2"] - 1.0) < 0.03
    assert all(abs(detector.gap_autocorrelation(g)["autocorr"]) < 3 * detector.gap_autocorrelation(g)["band95"])
    w = detector.window_count_summary(g, window=60.0)
    assert abs(w["fano_observed"] - 1.0) < 0.1


def test_regular_arrivals_are_under_dispersed_and_serial_dependence_is_detected():
    rng = np.random.default_rng(2)
    t = np.cumsum(2.0 + 0.2 * rng.standard_normal(30_000))  # almost periodic
    g = detector.lane_gaps(np.zeros(len(t), int), np.ones(len(t), int), t)
    assert detector.gap_summary(g)["cv2"] < 0.05
    assert detector.window_count_summary(g, window=60.0)["fano_observed"] < 0.5
    # a gap series with positive memory (AR(1)) must show lag-1 autocorrelation well outside the band
    e, x = rng.standard_normal(30_000), np.zeros(30_000)
    for i in range(1, len(x)):
        x[i] = 0.6 * x[i - 1] + e[i]
    t = np.cumsum(10.0 + x)
    g = detector.lane_gaps(np.zeros(len(t), int), np.ones(len(t), int), t)
    ac = detector.gap_autocorrelation(g)
    assert ac.autocorr[0] > 0.5 and ac.autocorr[0] > 5 * ac.band95[0]
