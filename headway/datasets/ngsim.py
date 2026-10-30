import pandas as pd

from .common import clip_headways


def load_ngsim(path, workers=None, location=None, hz=10):
    """NGSIM export (Data.gov). One sample per second per follower, leader must exist.

    NGSIM stores Time_Headway=0 / Preceding=0 when there is no leader, so
    Preceding > 0 is the validity test (the 0.5 s floor would also catch it).
    `workers` is unused: a single CSV parse is serial (a few minutes at most).
    """
    df = pd.read_csv(path, usecols=["Frame_ID", "Preceding", "Time_Headway", "Location"],
                     thousands=",", on_bad_lines="skip")  # export writes "1,309"
    if location:
        df = df[df.Location.str.lower() == location.lower()]
    df = df[(df.Frame_ID % hz == 0) & (df.Preceding > 0)]
    return clip_headways(df.Time_Headway.to_numpy())
