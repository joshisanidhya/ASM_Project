"""Detector inter-arrival times reconstructed from trajectories, and a test of the renewal model on them.

    python -m headway.detector --highd data/highd --ngsim data/ngsim/export.csv --out outputs/detector

The time headway of the datasets (highD ``thw``, NGSIM ``Time_Headway``) is measured between a
follower and its leader at one instant. A *detector inter-arrival time* is the time between two
successive vehicles of a lane crossing a fixed cross-section. The two are different random
variables, and the renewal counting process N(t) of the paper describes the second. A virtual loop
detector is placed at a fixed longitudinal coordinate, the time at which each vehicle's reference
point crosses it is obtained by linear interpolation between the two samples around the crossing,
and the crossing times of one lane in one recording are differenced.

This module then asks the questions the headway-driven renewal model cannot answer:

* how do detector inter-arrival times differ from the one-second time headways? (``gap_summary``)
* are successive inter-arrival times of a lane uncorrelated, as a renewal process requires?
  (``gap_autocorrelation``)
* does the variance of the vehicle count in a window match renewal theory and not Poisson?
  (``window_count_summary``)
"""
import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .parallel import pmap


# --- crossing times ---------------------------------------------------------------------------
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


def lane_gaps(unit: np.ndarray, lane: np.ndarray, tc: np.ndarray) -> pd.DataFrame:
    """Inter-arrival times of successive crossings within one (unit, lane); unit = recording or period."""
    df = pd.DataFrame({"unit": unit, "lane": lane, "t": tc}).sort_values(["unit", "lane", "t"], kind="stable")
    df["gap"] = df.groupby(["unit", "lane"])["t"].diff()
    return df.dropna(subset=["gap"]).reset_index(drop=True)


# --- datasets -----------------------------------------------------------------------------------
def _highd_recording(tracks_csv) -> pd.DataFrame:
    tracks_csv = Path(tracks_csv)
    meta = pd.read_csv(tracks_csv.with_name(tracks_csv.name.replace("_tracks.csv", "_recordingMeta.csv")))
    hz = float(meta["frameRate"].iloc[0])
    df = pd.read_csv(tracks_csv, usecols=["frame", "id", "x", "width", "laneId"])
    df = df.sort_values(["id", "frame"], kind="stable")
    pos = (df["x"] + df["width"] / 2.0).to_numpy()  # vehicle centre; x is the corner of the bounding box
    x0 = 0.5 * (pos.min() + pos.max())
    v, tc, ln = crossing_times(df["id"].to_numpy(), df["frame"].to_numpy() / hz, pos, df["laneId"].to_numpy(), x0)
    out = lane_gaps(np.full(len(v), int(tracks_csv.name[:2])), ln, tc)
    return out.assign(x0=x0)


def highd_gaps(path, workers: Optional[int] = None) -> pd.DataFrame:
    files = sorted(Path(path).glob("*_tracks.csv"))
    return pd.concat(pmap(_highd_recording, files, workers), ignore_index=True)


NGSIM_LANES = (1, 2, 3, 4, 5, 6)   # Lane_ID 7 is the on-ramp
NGSIM_Y0_FT = 800.0                 # detector position along the section, feet (the section is ~1650 ft)
PERIOD_GAP_MS = 10 * 60 * 1000


def ngsim_gaps(path, location="i-80", period=1, y0=NGSIM_Y0_FT, lanes=NGSIM_LANES) -> pd.DataFrame:
    cols = ["Vehicle_ID", "Frame_ID", "Local_Y", "Lane_ID", "Global_Time", "Location"]
    df = pd.read_csv(path, usecols=cols, thousands=",", on_bad_lines="skip")
    df = df[df.Location.str.lower() == location.lower()]
    if period:  # recording periods are runs of Global_Time separated by gaps over 10 minutes
        u = np.unique(df.Global_Time.to_numpy())
        cuts = u[1:][np.diff(u) > PERIOD_GAP_MS]
        df = df[np.searchsorted(cuts, df.Global_Time.to_numpy(), side="right") + 1 == period]
    df = df.sort_values(["Vehicle_ID", "Global_Time"], kind="stable")
    v, tc, ln = crossing_times(df.Vehicle_ID.to_numpy(), df.Global_Time.to_numpy() / 1000.0,
                               df.Local_Y.to_numpy(float), df.Lane_ID.to_numpy(), y0)
    keep = np.isin(ln, lanes)
    out = lane_gaps(np.full(keep.sum(), int(period or 0)), ln[keep], tc[keep])
    return out.assign(x0=y0)


# --- statistics -----------------------------------------------------------------------------------
def gap_summary(gaps: pd.DataFrame, lo: float = 0.5, hi: float = 25.0) -> Dict[str, float]:
    """Moments of the inter-arrival times (all, and restricted to the headway window [lo, hi])."""
    g = gaps["gap"].to_numpy()
    w = g[(g >= lo) & (g <= hi)]
    return {
        "n_gaps": len(g), "n_units": gaps.groupby(["unit", "lane"]).ngroups,
        "mean_s": g.mean(), "median_s": float(np.median(g)), "std_s": g.std(ddof=1),
        "cv2": g.var(ddof=1) / g.mean() ** 2, "flow_veh_h_lane": 3600.0 / g.mean(),
        "frac_below_lo": float((g < lo).mean()), "frac_above_hi": float((g > hi).mean()),
        "n_window": len(w), "mean_window_s": w.mean(), "std_window_s": w.std(ddof=1),
    }


def gap_autocorrelation(gaps: pd.DataFrame, max_lag: int = 5) -> pd.DataFrame:
    """Lag-k autocorrelation of successive inter-arrival times, pooled over lanes and recordings.

    Each (unit, lane) series is centred on its own mean, so differences in flow between lanes or
    recordings do not masquerade as dependence. r_k = sum of lag-k products / sum of squares over all
    series; for independent gaps r_k ~ N(0, 1/m) with m the number of pairs.
    """
    num, den, pairs = np.zeros(max_lag), 0.0, np.zeros(max_lag, dtype=int)
    for _, s in gaps.groupby(["unit", "lane"])["gap"]:
        x = s.to_numpy() - s.mean()
        den += float((x ** 2).sum())
        for k in range(1, max_lag + 1):
            if len(x) > k:
                num[k - 1] += float((x[:-k] * x[k:]).sum())
                pairs[k - 1] += len(x) - k
    return pd.DataFrame({"lag": np.arange(1, max_lag + 1), "autocorr": num / den,
                         "band95": 1.96 / np.sqrt(pairs), "n_pairs": pairs})


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


def plot_gaps(title: str, gaps: pd.DataFrame, acorr: pd.DataFrame, path, headway_acorr: Optional[pd.DataFrame] = None) -> None:
    """(a) density of detector inter-arrival times against the exponential law of the same mean;
    (b) lag-k autocorrelation of the gaps (and of the one-second time headways, if given) with the white-noise band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = gaps["gap"].to_numpy()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    edges = np.arange(0.0, 25.0 + 0.5, 0.5)
    a.hist(g[g <= 25.0], bins=edges, density=True, color="#BBBBBB", edgecolor="#777777", label="detector inter-arrival times")
    x = np.linspace(0.0, 25.0, 500)
    a.plot(x, np.exp(-x / g.mean()) / g.mean(), color="#1F77B4", lw=2, label=f"exponential, mean {g.mean():.2f} s")
    a.set(xlabel="Inter-arrival time [s]", ylabel="Density [1/s]", xlim=(0, 25), title="(a) distribution of gaps")
    a.legend(frameon=False, fontsize=9)
    a.grid(True, linestyle=":", alpha=0.6)
    w = 0.35 if headway_acorr is not None else 0.6
    b.bar(acorr["lag"] - (w / 2 if headway_acorr is not None else 0), acorr["autocorr"], width=w, color="#1F77B4",
          label="detector inter-arrival times")
    if headway_acorr is not None:
        b.bar(headway_acorr["lag"] + w / 2, headway_acorr["autocorr"], width=w, color="#D62728", label="1 Hz time headways")
    b.axhspan(-acorr["band95"].iloc[0], acorr["band95"].iloc[0], color="#999999", alpha=0.35, label="95% white-noise band")
    b.set(xlabel="Lag k (successive vehicles / seconds)", ylabel="Autocorrelation", title="(b) serial dependence")
    b.legend(frameon=False, fontsize=9)
    b.grid(True, linestyle=":", alpha=0.6)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def analyse(name: str, gaps: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    gaps.to_csv(out / f"{name}_gaps.csv", index=False)
    pd.DataFrame([gap_summary(gaps)]).to_csv(out / f"{name}_gap_summary.csv", index=False)
    gap_autocorrelation(gaps).to_csv(out / f"{name}_gap_autocorr.csv", index=False)
    pd.DataFrame([window_count_summary(gaps, w) for w in (30.0, 60.0, 120.0)]).to_csv(
        out / f"{name}_window_counts.csv", index=False)
    # distribution of the gaps against the exponential (Poisson) distribution of the same mean
    g = gaps["gap"].to_numpy()
    ks = stats.kstest(g, "expon", args=(0, g.mean()))
    pd.DataFrame([{"ks_D_vs_exponential": ks.statistic, "ks_p": ks.pvalue}]).to_csv(
        out / f"{name}_gap_vs_exponential.csv", index=False)


def main(argv: Optional[Sequence[str]] = None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--highd", help="folder with the highD *_tracks.csv files")
    ap.add_argument("--ngsim", help="NGSIM export CSV")
    ap.add_argument("--location", default="i-80")
    ap.add_argument("--period", type=int, default=1)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default="outputs/detector")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.highd:
        analyse("highd", highd_gaps(a.highd, a.workers), out)
    if a.ngsim:
        analyse("ngsim", ngsim_gaps(a.ngsim, a.location, a.period), out)
    print(f"wrote detector analysis to {out}/")


if __name__ == "__main__":
    main()
