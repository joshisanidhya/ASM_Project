"""Lyft Level 5 *Prediction* dataset (zarr: scenes / frames / agents).

Lyft has no headway column and the AV is the follower (driver viewpoint), so the leader
is derived: the nearest tracked vehicle ahead of the AV inside its lane corridor.

    headway = bumper-to-bumper gap / AV speed

ponytail: the lane is approximated by a lateral corridor around the AV heading, not the
semantic map, so leaders around curves/turns can be missed or mis-picked. Swap in
map-based lane membership (l5kit MapAPI) if that matters.
"""
import numpy as np

from ..parallel import pmap
from .common import clip_headways

# l5kit PERCEPTION_LABELS indices: CAR, VAN, BUS, TRUCK, EMERGENCY_VEHICLE, OTHER_VEHICLE.
# Verify against the l5kit version that produced your zarr.
VEHICLE_LABELS = (3, 4, 6, 7, 8, 9)
EGO_LENGTH = 4.87        # [m] l5kit EGO_EXTENT_LENGTH
LANE_HALF_WIDTH = 1.8    # [m] lateral corridor, assumption
MIN_SPEED = 0.5          # [m/s] below this headway = gap/speed is meaningless
HZ = 10                  # Lyft frame rate; one sample per second


def leader_gap(ego_xy, ego_yaw, centroids, lengths):
    """Bumper-to-bumper gap [m] to the nearest vehicle ahead in the ego corridor, else NaN."""
    h = np.array([np.cos(ego_yaw), np.sin(ego_yaw)])
    d = centroids - ego_xy
    lon, lat = d @ h, d @ np.array([-h[1], h[0]])
    ahead = (lon > 0) & (np.abs(lat) < LANE_HALF_WIDTH)
    if not ahead.any():
        return np.nan
    i = np.argmin(np.where(ahead, lon, np.inf))
    return lon[i] - (EGO_LENGTH + lengths[i]) / 2


def scene_headways(t, ego_xy, ego_yaw, frame_agents):
    """Headways [s] sampled once per second in one scene.

    t: (T,) seconds, ego_xy: (T,2), ego_yaw: (T,),
    frame_agents: callable k -> (centroids (N,2), lengths (N,)) of vehicles in frame k.
    """
    speed = np.hypot(*np.gradient(ego_xy, t, axis=0).T)
    out = []
    for k in range(0, len(t), HZ):
        if speed[k] <= MIN_SPEED:
            continue
        gap = leader_gap(ego_xy[k], ego_yaw[k], *frame_agents(k))
        if gap > 0:  # NaN compares False
            out.append(gap / speed[k])
    return out


def _chunk(task):
    path, lo, hi = task
    import zarr
    z = zarr.open_group(path, mode="r")
    out = []
    for scene in z["scenes"][lo:hi]:
        f0, f1 = scene["frame_index_interval"]
        frames = z["frames"][f0:f1]
        iv = frames["agent_index_interval"]
        a0 = iv[0, 0]
        ag = z["agents"][a0:iv[-1, 1]]
        vehicle = np.isin(ag["label_probabilities"].argmax(1), VEHICLE_LABELS)
        rot = frames["ego_rotation"]

        def frame_agents(k):
            s = slice(iv[k, 0] - a0, iv[k, 1] - a0)
            m = vehicle[s]
            return ag["centroid"][s][m], ag["extent"][s][m, 0]

        out += scene_headways(frames["timestamp"] / 1e9, frames["ego_translation"][:, :2],
                              np.arctan2(rot[:, 1, 0], rot[:, 0, 0]), frame_agents)
    return np.array(out)


def load_lyft(path, workers=None, chunks_per_worker=4):
    """`path` is a Level 5 .zarr folder (e.g. sample.zarr, train.zarr). Scenes split across cores."""
    import os
    import zarr
    n = len(zarr.open_group(path, mode="r")["scenes"])
    step = max(1, n // ((workers or os.cpu_count()) * chunks_per_worker))
    tasks = [(str(path), lo, min(lo + step, n)) for lo in range(0, n, step)]
    return clip_headways(np.concatenate(pmap(_chunk, tasks, workers)))
