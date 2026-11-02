"""Loader for the highD dataset (German Highway drone trajectory dataset).

Paper settings:
- Original frame rate: 25 Hz
- Extracted sampling rate: 1 Hz (subsampled every 25 frames)
- Traffic regime: Uninterrupted highway flow (human-driven vehicles)
- Headway variable: Time Headway (thw) in seconds, for vehicles with a preceding leader (precedingId > 0)
- Filtering window: [0.5, 25.0] seconds
"""
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from ..parallel import pmap
from .common import clip_headways

HIGHD_DEFAULT_HZ = 25


def _recording(tracks_csv: Path) -> np.ndarray:
    """Extract 1 Hz time headways from a single highD recording."""
    tracks_csv = Path(tracks_csv)
    # Search for matching recordingMeta.csv
    meta_name = tracks_csv.name.replace("_tracks.csv", "_recordingMeta.csv")
    meta_path = tracks_csv.with_name(meta_name)

    hz = HIGHD_DEFAULT_HZ
    if meta_path.exists():
        try:
            meta_df = pd.read_csv(meta_path, nrows=5)
            # Find column named frameRate (case-insensitive)
            cols = {c.lower(): c for c in meta_df.columns}
            if "framerate" in cols:
                hz = int(round(float(meta_df[cols["framerate"]].iloc[0])))
        except Exception:
            hz = HIGHD_DEFAULT_HZ

    # Load tracks CSV with only required columns
    df = pd.read_csv(tracks_csv)
    cols = {c.lower(): c for c in df.columns}
    frame_col = cols.get("frame", "frame")
    preceding_col = cols.get("precedingid", "precedingId")
    thw_col = cols.get("thw", "thw")

    if thw_col not in df.columns or frame_col not in df.columns:
        return np.array([], dtype=float)

    # Filter: 1 Hz extraction and leader vehicle exists (precedingId > 0)
    has_leader = df[preceding_col] > 0 if preceding_col in df.columns else (df[thw_col] > 0)
    sampled_1hz = (df[frame_col] % hz == 0)
    valid_mask = sampled_1hz & has_leader

    thw_vals = pd.to_numeric(df.loc[valid_mask, thw_col], errors="coerce").dropna().to_numpy()
    return clip_headways(thw_vals)


def load_highd(path: Union[str, Path], workers: Optional[int] = None) -> np.ndarray:
    """Load highD dataset from folder containing recordings.

    Args:
        path: Directory containing *_tracks.csv files.
        workers: Optional worker count for parallel loading.

    Returns:
        1D NumPy array of validated 1 Hz time headways in [0.5, 25.0] s.
    """
    p = Path(path)
    if p.is_file():
        files = [p]
    else:
        # Use a set to prevent duplicate globbing on case-insensitive filesystems (Windows)
        files = sorted(set(list(p.glob("*_tracks.csv")) + list(p.glob("*_tracks.CSV"))))
    if not files:
        raise FileNotFoundError(f"No *_tracks.csv files found in highD directory: {path}")

    records = pmap(_recording, files, workers)
    if not records:
        return np.array([], dtype=float)
    return np.concatenate(records)
