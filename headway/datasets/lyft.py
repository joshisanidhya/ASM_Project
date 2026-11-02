"""Loader for the Lyft Level 5 Autonomous Driving Prediction Dataset (Zarr format).

Paper settings:
- Original frame rate: 10 Hz
- Extracted sampling rate: 1 Hz (subsampled every 10 frames)
- Traffic regime: Urban road, mixed traffic flow (human-driven vehicles + autonomous vehicles)
- Follower: Autonomous ego vehicle (driver viewpoint)

Scientific Methodology & Approximations:
1. Leader Selection:
   - Evaluates tracked vehicles in each frame ahead of the autonomous ego vehicle.
   - Lateral corridor: |lateral_displacement| < LANE_HALF_WIDTH (1.8 m) relative to ego heading.
   - Nearest vehicle in corridor: min(longitudinal_displacement > 0).
2. Headway Calculation:
   - Bumper-to-bumper distance: gap = longitudinal_dist - (EGO_LENGTH + leader_length) / 2.
   - Speed: v = ||d(ego_xy) / dt|| computed via central differences.
   - Headway proxy: h = gap / v (for v > MIN_SPEED = 0.5 m/s).
   - Limitation Note: True microscopic time headway is defined by vehicle passage times past
     a stationary longitudinal cross-section. On straight road segments at near-constant speed,
     gap / v closely approximates passage-time headway. In curves or turning maneuvers,
     HD vector map lane polyline association would provide higher precision.
3. Filtering window: [0.5, 25.0] seconds.
"""
import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from ..parallel import pmap
from .common import clip_headways

# l5kit perception labels: 3=CAR, 4=VAN, 6=BUS, 7=TRUCK, 8=EMERGENCY_VEHICLE, 9=OTHER_VEHICLE
VEHICLE_LABELS = (3, 4, 6, 7, 8, 9)
EGO_LENGTH = 4.87         # [m] Standard l5kit ego extent length
LANE_HALF_WIDTH = 1.8     # [m] Half-width corridor for in-lane car-following
MIN_SPEED = 0.5           # [m/s] Below this threshold, headway = gap / speed is numerically ill-posed
HZ = 10                   # Lyft recording frame rate (10 Hz)


def leader_gap(
    ego_xy: np.ndarray,
    ego_yaw: float,
    centroids: np.ndarray,
    lengths: np.ndarray,
    lane_half_width: float = LANE_HALF_WIDTH,
) -> float:
    """Calculate bumper-to-bumper gap [m] to the nearest vehicle in the ego vehicle's lane corridor.

    Args:
        ego_xy: (2,) array [x, y] of ego vehicle center.
        ego_yaw: Ego orientation angle [rad].
        centroids: (N, 2) array of other vehicle centroids.
        lengths: (N,) array of other vehicle longitudinal extents.
        lane_half_width: Lateral corridor boundary [m].

    Returns:
        Bumper-to-bumper distance [m] if a valid leader exists, otherwise NaN.
    """
    if len(centroids) == 0:
        return np.nan
    h = np.array([np.cos(ego_yaw), np.sin(ego_yaw)])
    d = centroids - ego_xy
    lon = np.dot(d, h)
    lat = np.dot(d, np.array([-h[1], h[0]]))

    ahead = (lon > 0.0) & (np.abs(lat) < lane_half_width)
    if not np.any(ahead):
        return np.nan

    valid_lon = np.where(ahead, lon, np.inf)
    idx = int(np.argmin(valid_lon))
    gap = lon[idx] - (EGO_LENGTH + lengths[idx]) / 2.0
    return gap if gap > 0.0 else np.nan


def scene_headways(
    t: np.ndarray,
    ego_xy: np.ndarray,
    ego_yaw: np.ndarray,
    frame_agents_fn: Callable[[int], Tuple[np.ndarray, np.ndarray]],
) -> List[float]:
    """Sample car-following headways at 1 Hz from a single Lyft scene."""
    if len(t) < 2:
        return []
    # Velocity via finite differences
    dt = np.gradient(t)
    dt_safe = np.where(dt > 0, dt, 0.1)
    vx = np.gradient(ego_xy[:, 0]) / dt_safe
    vy = np.gradient(ego_xy[:, 1]) / dt_safe
    speed = np.hypot(vx, vy)

    out = []
    # Subsample at 1 Hz (every HZ frames)
    for k in range(0, len(t), HZ):
        if speed[k] <= MIN_SPEED:
            continue
        cents, lens = frame_agents_fn(k)
        gap = leader_gap(ego_xy[k], ego_yaw[k], cents, lens)
        if np.isfinite(gap) and gap > 0.0:
            out.append(float(gap / speed[k]))
    return out


def _chunk(task: Tuple[str, int, int]) -> np.ndarray:
    """Worker task processing a slice of scenes from a Level 5 Zarr store."""
    path, lo, hi = task
    try:
        import zarr
        z = zarr.open_group(path, mode="r")
        scenes = z["scenes"][lo:hi]
        frames_arr = z["frames"]
        agents_arr = z["agents"]
    except Exception:
        return np.array([], dtype=float)

    out: List[float] = []
    for scene in scenes:
        f0, f1 = scene["frame_index_interval"]
        frames = frames_arr[f0:f1]
        if len(frames) == 0:
            continue
        iv = frames["agent_index_interval"]
        a0 = iv[0, 0]
        a1 = iv[-1, 1]
        ag = agents_arr[a0:a1]
        vehicle_mask = np.isin(ag["label_probabilities"].argmax(1), VEHICLE_LABELS)
        rot = frames["ego_rotation"]

        def frame_agents(k):
            s = slice(iv[k, 0] - a0, iv[k, 1] - a0)
            m = vehicle_mask[s]
            return ag["centroid"][s][m], ag["extent"][s][m, 0]

        timestamps_s = frames["timestamp"] / 1e9
        ego_xy = frames["ego_translation"][:, :2]
        ego_yaw = np.arctan2(rot[:, 1, 0], rot[:, 0, 0])

        out += scene_headways(timestamps_s, ego_xy, ego_yaw, frame_agents)

    return np.array(out, dtype=float)


def load_lyft(
    path: Union[str, Path],
    workers: Optional[int] = None,
    chunks_per_worker: int = 4,
) -> np.ndarray:
    """Load Lyft Level 5 Zarr dataset (e.g. sample.zarr, train.zarr).

    Args:
        path: Path to the .zarr directory.
        workers: Number of worker processes.
        chunks_per_worker: Number of scene tasks per worker core.

    Returns:
        1D NumPy array of validated 1 Hz time headways in [0.5, 25.0] s.
    """
    try:
        import zarr
    except ImportError:
        raise ImportError("zarr is required to load Lyft dataset. Install with `pip install zarr<3`")

    z = zarr.open_group(str(path), mode="r")
    num_scenes = len(z["scenes"])
    if num_scenes == 0:
        return np.array([], dtype=float)

    total_workers = workers or (os.cpu_count() or 1)
    step = max(1, num_scenes // (total_workers * chunks_per_worker))
    tasks = [(str(path), lo, min(lo + step, num_scenes)) for lo in range(0, num_scenes, step)]

    chunks = pmap(_chunk, tasks, workers)
    if not chunks:
        return np.array([], dtype=float)
    return clip_headways(np.concatenate(chunks))
