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

from ..distributions import ALPHA, T_MAX
from .common import segment_ids

NGSIM_DEFAULT_HZ = 10
PERIOD_GAP_MS = 10 * 60 * 1000  # recording periods are separated by gaps far longer than this


def load_ngsim(path, workers=None, location=None, lanes=None, period=None, hz=NGSIM_DEFAULT_HZ,
               return_groups=False):
    """NGSIM export (Data.gov). One sample per second per follower, leader must exist.

    NGSIM stores Time_Headway=0 / Preceding=0 when there is no leader, so
    Preceding > 0 is the validity test (the 0.5 s floor would also catch it).
    The paper states neither a lane nor a time-period filter, so both are optional:
    `lanes` keeps only these Lane_IDs; `period` keeps one recording period (1-based, in time
    order), where periods are the runs of Global_Time separated by gaps over 10 minutes.

    The returned headways are ordered by vehicle, then time. The raw export is not in that order,
    and the Markov-chain and autocorrelation checks of headway.validation depend on it (the fits
    and histograms do not). With return_groups=True the trajectory-run id of every headway is
    returned too (see common.segment_ids).
    `workers` is unused: a single CSV parse is serial (a few minutes at most).
    """
    want = {"Frame_ID", "Preceding", "Time_Headway", "Location", "Vehicle_ID"}
    want |= {"Lane_ID"} if lanes else set()
    want |= {"Global_Time"} if period else set()
    df = pd.read_csv(path, usecols=lambda c: c in want, thousands=",", on_bad_lines="skip")  # export writes "1,309"
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
    df = df[(df.Time_Headway >= ALPHA) & (df.Time_Headway <= T_MAX)]
    if "Vehicle_ID" in df:
        df = df.sort_values(["Vehicle_ID", "Frame_ID"], kind="stable")
    h = df.Time_Headway.to_numpy(float)
    if not return_groups:
        return h
    if "Vehicle_ID" not in df:
        return h, np.arange(len(h))  # no vehicle ids: every sample is its own run, so no transitions are counted
    return h, segment_ids(df.Vehicle_ID.to_numpy(), (df.Frame_ID // hz).to_numpy())
