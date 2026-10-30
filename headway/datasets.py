"""Dataset loaders. A loader returns 1 Hz car-following headways in seconds.

Add highD / exiD / Waymo / Lyft by writing another function with this contract
and registering it in LOADERS.
"""
import numpy as np
import pandas as pd

from .distributions import ALPHA, T_MAX


def clip_headways(h):
    """Paper section 4: drop headways < 0.5 s (unsafe) and > 25 s (not following)."""
    h = np.asarray(h, float)
    return h[(h >= ALPHA) & (h <= T_MAX)]


def load_ngsim(csv, location=None, hz=10):
    """NGSIM export (Data.gov). One sample per second per follower, leader must exist.

    NGSIM stores Time_Headway=0 / Preceding=0 when there is no leader, so
    Preceding > 0 is the validity test (the 0.5 s floor would also catch it).
    """
    df = pd.read_csv(csv, usecols=["Frame_ID", "Preceding", "Time_Headway", "Location"],
                     dtype={"Frame_ID": "int64", "Preceding": "int64", "Time_Headway": "float64"},
                     thousands=",", on_bad_lines="skip")  # export writes "1,309"
    if location:
        df = df[df.Location.str.lower() == location.lower()]
    df = df[(df.Frame_ID % hz == 0) & (df.Preceding > 0)]
    return clip_headways(df.Time_Headway.to_numpy())


LOADERS = {"ngsim": load_ngsim}
