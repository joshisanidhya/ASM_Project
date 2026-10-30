from pathlib import Path

import numpy as np
import pandas as pd

from ..parallel import pmap
from .common import clip_headways


def _recording(tracks_csv):
    """Headways from one highD recording (NN_tracks.csv + NN_recordingMeta.csv)."""
    meta = tracks_csv.with_name(tracks_csv.name.replace("_tracks", "_recordingMeta"))
    hz = int(round(pd.read_csv(meta, usecols=["frameRate"]).frameRate.iloc[0]))
    df = pd.read_csv(tracks_csv, usecols=["frame", "precedingId", "thw"])
    # highD writes precedingId=0 / thw=0 when the vehicle has no leader
    df = df[(df.frame % hz == 0) & (df.precedingId > 0)]
    return clip_headways(df.thw.to_numpy())


def load_highd(path, workers=None):
    """`path` is the highD folder holding 01_tracks.csv ... 60_tracks.csv; one task per recording."""
    files = sorted(Path(path).glob("*_tracks.csv"))
    if not files:
        raise FileNotFoundError(f"no *_tracks.csv in {path}")
    return np.concatenate(pmap(_recording, files, workers))
