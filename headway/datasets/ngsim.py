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
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from .common import clip_headways

NGSIM_DEFAULT_HZ = 10

def load_ngsim(path, workers=None, location=None, lanes=None, hz=10):
    """NGSIM export (Data.gov). One sample per second per follower, leader must exist.

    NGSIM stores Time_Headway=0 / Preceding=0 when there is no leader, so
    Preceding > 0 is the validity test (the 0.5 s floor would also catch it).
    `lanes` optionally keeps only these Lane_IDs; the paper does not state a lane filter.
    `workers` is unused: a single CSV parse is serial (a few minutes at most).
    """
    df = pd.read_csv(path, usecols=["Frame_ID", "Preceding", "Time_Headway", "Location", "Lane_ID"],
                     thousands=",", on_bad_lines="skip")  # export writes "1,309"
    if location:
        df = df[df.Location.str.lower() == location.lower()]
    if lanes:
        df = df[df.Lane_ID.isin(lanes)]
    df = df[(df.Frame_ID % hz == 0) & (df.Preceding > 0)]
    return clip_headways(df.Time_Headway.to_numpy())
