import numpy as np
import pandas as pd

from headway.datasets.highd import load_highd
from headway.datasets.lyft import EGO_LENGTH, leader_gap, load_lyft


def _write_highd(folder, rec):
    pd.DataFrame({"frameRate": [25]}).to_csv(folder / f"{rec}_recordingMeta.csv", index=False)
    frames = np.arange(1, 101)
    follower = pd.DataFrame({"frame": frames, "precedingId": 2, "thw": 2.0})
    leaderless = pd.DataFrame({"frame": frames, "precedingId": 0, "thw": 0.0})
    tooshort = pd.DataFrame({"frame": [25, 50], "precedingId": 2, "thw": [0.3, 30.0]})  # both out of range
    pd.concat([follower, leaderless, tooshort]).to_csv(folder / f"{rec}_tracks.csv", index=False)


def test_highd_samples_1hz_and_filters(tmp_path):
    for rec in ("01", "02"):
        _write_highd(tmp_path, rec)
    for workers in (1, 2):  # inline and process pool
        h = load_highd(tmp_path, workers=workers)
        assert h.tolist() == [2.0] * 8  # frames 25/50/75/100 x 2 recordings


def test_leader_gap_picks_nearest_in_lane_vehicle():
    cars = np.array([[30.0, 0.5], [20.0, 5.0], [-10.0, 0.0], [50.0, 0.0]])  # lead, adjacent lane, behind, far
    gap = leader_gap(np.zeros(2), 0.0, cars, np.full(4, EGO_LENGTH))
    assert abs(gap - (30 - EGO_LENGTH)) < 1e-9
    assert np.isnan(leader_gap(np.zeros(2), 0.0, cars[1:3], np.full(2, 4.0)))


def test_lyft_zarr_roundtrip(tmp_path):
    zarr = __import__("pytest").importorskip("zarr")
    T = 31  # 3 s at 10 Hz, AV at 10 m/s, one car 30 m ahead at the same speed
    scene = np.zeros(1, [("frame_index_interval", "<i8", (2,)), ("host", "<U16"),
                         ("start_time", "<i8"), ("end_time", "<i8")])
    scene["frame_index_interval"] = [[0, T]]
    fr = np.zeros(T, [("timestamp", "<i8"), ("agent_index_interval", "<i8", (2,)),
                      ("traffic_light_faces_index_interval", "<i8", (2,)),
                      ("ego_translation", "<f8", (3,)), ("ego_rotation", "<f8", (3, 3))])
    k = np.arange(T)
    fr["timestamp"] = k * 100_000_000
    fr["agent_index_interval"] = np.stack([k, k + 1], 1)
    fr["ego_translation"][:, 0] = 10.0 * k / 10
    fr["ego_rotation"] = np.eye(3)
    ag = np.zeros(T, [("centroid", "<f8", (2,)), ("extent", "<f4", (3,)), ("yaw", "<f4"),
                      ("velocity", "<f4", (2,)), ("track_id", "<u8"), ("label_probabilities", "<f4", (17,))])
    ag["centroid"][:, 0] = 10.0 * k / 10 + 30
    ag["centroid"][:, 1] = 0.5
    ag["extent"][:, 0] = EGO_LENGTH
    ag["label_probabilities"][:, 3] = 1.0  # CAR
    g = zarr.open_group(str(tmp_path / "s.zarr"), mode="w")
    for name, arr in [("scenes", scene), ("frames", fr), ("agents", ag)]:
        g.array(name, arr)

    want = (30 - EGO_LENGTH) / 10
    for workers in (1, 2):
        h = load_lyft(tmp_path / "s.zarr", workers=workers)
        assert len(h) == 4 and np.allclose(h, want), h
