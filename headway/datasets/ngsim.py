"""Loader for the NGSIM vehicle trajectory dataset (FHWA).

Paper settings:
- Original frame rate: 10 Hz
- Extracted sampling rate: 1 Hz (subsampled every 10 frames)
- Traffic regime: Highway and urban road (human-driven vehicles, interrupted/uninterrupted)
- Dataset focus: I-80 reconstructed freeway data (or user-specified location)
- Headway variable: Time_Headway (temporal headway in seconds)
- Preceding vehicle requirement: Preceding > 0 (identifies car-following interaction)
- Filtering window: [0.5, 25.0] seconds
"""
import numpy as np
import pandas as pd

from .common import clip_headways

NGSIM_DEFAULT_HZ = 10
PERIOD_GAP_MS = 10 * 60 * 1000  # recording periods are separated by gaps far longer than this


def load_ngsim(path, workers=None, location=None, lanes=None, period=None, hz=NGSIM_DEFAULT_HZ):
    """NGSIM export (Data.gov). One sample per second per follower, leader must exist.

    NGSIM stores Time_Headway=0 / Preceding=0 when there is no leader, so
    Preceding > 0 is the validity test (the 0.5 s floor would also catch it).
    The paper states neither a lane nor a time-period filter, so both are optional:
    `lanes` keeps only these Lane_IDs; `period` keeps one recording period (1-based, in time
    order), where periods are the runs of Global_Time separated by gaps over 10 minutes.
    `workers` is unused: a single CSV parse is serial (a few minutes at most).
    """
    cols = ["Frame_ID", "Preceding", "Time_Headway", "Location"]
    cols += ["Lane_ID"] * bool(lanes) + ["Global_Time"] * bool(period)
    df = pd.read_csv(path, usecols=cols, thousands=",", on_bad_lines="skip")  # export writes "1,309"
    if location:
        df = df[df.Location.str.lower() == location.lower()]
    if lanes:
        df = df[df.Lane_ID.isin(lanes)]
    df = df[(df.Frame_ID % hz == 0) & (df.Preceding > 0)]
    if period:
        t = df.Global_Time.to_numpy()
        u = np.unique(t)
        cuts = u[1:][np.diff(u) > PERIOD_GAP_MS]  # first timestamp of every period after the first
        df = df[np.searchsorted(cuts, t, side="right") + 1 == period]
    return clip_headways(df.Time_Headway.to_numpy())
