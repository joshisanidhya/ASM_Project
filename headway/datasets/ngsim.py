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


def load_ngsim(
    path: Union[str, Path],
    workers: Optional[int] = None,
    location: Optional[str] = "i-80",
    hz: int = NGSIM_DEFAULT_HZ,
) -> np.ndarray:
    """Load NGSIM trajectory CSV export.

    Args:
        path: Path to the NGSIM CSV file or directory containing it.
        workers: Unused (CSV parsing is streaming/serial).
        location: Specific location to filter (default 'i-80' matching paper section 4).
        hz: Original frame rate (default 10 Hz).

    Returns:
        1D NumPy array of validated 1 Hz time headways in [0.5, 25.0] s.
    """
    p = Path(path)
    if p.is_dir():
        csv_files = sorted(p.glob("*.csv"), key=lambda f: -f.stat().st_size)
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in NGSIM directory: {path}")
        file_path = csv_files[0]
    else:
        file_path = p

    if not file_path.exists():
        raise FileNotFoundError(f"NGSIM file does not exist: {file_path}")

    # Inspect CSV headers for case-insensitive matching
    sample_df = pd.read_csv(file_path, nrows=5)
    cols_map = {c.lower(): c for c in sample_df.columns}

    frame_col = cols_map.get("frame_id", "Frame_ID")
    preceding_col = cols_map.get("preceding", "Preceding")
    thw_col = cols_map.get("time_headway", "Time_Headway")
    loc_col = cols_map.get("location", "Location")

    cols_to_use = [frame_col, preceding_col, thw_col]
    if loc_col in sample_df.columns and location:
        cols_to_use.append(loc_col)

    df = pd.read_csv(
        file_path,
        usecols=cols_to_use,
        thousands=",",
        on_bad_lines="skip",
        low_memory=False,
    )

    # Optional location filter (e.g. I-80 reconstructed car-following)
    if location and loc_col in df.columns:
        df = df[df[loc_col].astype(str).str.lower().str.contains(location.lower())]

    # Preceding vehicle must exist (> 0) and subsample at 1 Hz
    has_leader = df[preceding_col] > 0
    sampled_1hz = (df[frame_col] % hz == 0)
    valid_mask = has_leader & sampled_1hz

    thw_series = pd.to_numeric(df.loc[valid_mask, thw_col], errors="coerce").dropna()
    return clip_headways(thw_series.to_numpy())
