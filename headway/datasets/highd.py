"""Loader for the highD dataset (German Highway drone trajectory dataset).

Paper settings:
- Original frame rate: 25 Hz
- Extracted sampling rate: 1 Hz (subsampled every 25 frames)
- Traffic regime: Uninterrupted highway flow (human-driven vehicles)
- Headway variable: Time Headway (thw) in seconds, for vehicles with a preceding leader (precedingId > 0)
- Filtering window: [0.5, 25.0] seconds
"""
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..distributions import ALPHA, T_MAX
from ..parallel import pmap
from .common import segment_ids

HIGHD_DEFAULT_HZ = 25


def _recording(tracks_csv: Path) -> Tuple[np.ndarray, np.ndarray]:
    """1 Hz time headways of one highD recording and their trajectory-run ids.

    Rows come out ordered by vehicle, then time, so consecutive entries inside a run are
    consecutive seconds of one follower.
    """
    tracks_csv = Path(tracks_csv)
    meta_path = tracks_csv.with_name(tracks_csv.name.replace("_tracks.csv", "_recordingMeta.csv"))

    hz = HIGHD_DEFAULT_HZ
    if meta_path.exists():
        try:
            meta_df = pd.read_csv(meta_path, nrows=5)
            cols = {c.lower(): c for c in meta_df.columns}
            if "framerate" in cols:
                hz = int(round(float(meta_df[cols["framerate"]].iloc[0])))
        except Exception:
            hz = HIGHD_DEFAULT_HZ

    df = pd.read_csv(tracks_csv)
    cols = {c.lower(): c for c in df.columns}
    frame_col = cols.get("frame", "frame")
    id_col = cols.get("id", "id")
    preceding_col = cols.get("precedingid", "precedingId")
    thw_col = cols.get("thw", "thw")
    if thw_col not in df.columns or frame_col not in df.columns:
        return np.array([], dtype=float), np.array([], dtype=np.int64)

    # 1 Hz extraction, a leader exists (precedingId > 0), and the paper's headway window
    has_leader = df[preceding_col] > 0 if preceding_col in df.columns else (df[thw_col] > 0)
    thw = pd.to_numeric(df[thw_col], errors="coerce")
    keep = (df[frame_col] % hz == 0) & has_leader & (thw >= ALPHA) & (thw <= T_MAX)
    sel = df.loc[keep, [id_col, frame_col]].assign(thw=thw[keep])
    sel = sel.sort_values([id_col, frame_col], kind="stable")
    return sel["thw"].to_numpy(float), segment_ids(sel[id_col].to_numpy(), (sel[frame_col] // hz).to_numpy())


def load_highd(path: Union[str, Path], workers: Optional[int] = None, return_groups: bool = False):
    """Load highD from a folder of recordings.

    Args:
        path: Directory containing *_tracks.csv files (or one such file).
        workers: Optional worker count for parallel loading.
        return_groups: Also return the trajectory-run id of every headway (unique across recordings).

    Returns:
        1D array of 1 Hz time headways in [0.5, 25.0] s, ordered by recording, vehicle, time;
        with return_groups=True a tuple (headways, run ids).
    """
    p = Path(path)
    if p.is_file():
        files = [p]
    else:
        # A set prevents duplicate globbing on case-insensitive filesystems (Windows)
        files = sorted(set(list(p.glob("*_tracks.csv")) + list(p.glob("*_tracks.CSV"))))
    if not files:
        raise FileNotFoundError(f"No *_tracks.csv files found in highD directory: {path}")

    parts = pmap(_recording, files, workers)
    h = np.concatenate([x for x, _ in parts])
    offsets = np.cumsum([0] + [int(g.max()) + 1 if len(g) else 0 for _, g in parts[:-1]])
    g = np.concatenate([gi + o for (_, gi), o in zip(parts, offsets)]) if len(h) else np.array([], dtype=np.int64)
    return (h, g) if return_groups else h
