import numpy as np
import pandas as pd

from headway.datasets.highd import load_highd


def _write_highd(folder, rec):
    pd.DataFrame({"frameRate": [25]}).to_csv(folder / f"{rec}_recordingMeta.csv", index=False)
    frames = np.arange(1, 101)
    follower = pd.DataFrame({"frame": frames, "id": 1, "precedingId": 2, "thw": 2.0})
    leaderless = pd.DataFrame({"frame": frames, "id": 2, "precedingId": 0, "thw": 0.0})
    tooshort = pd.DataFrame({"frame": [25, 50], "id": 3, "precedingId": 2, "thw": [0.3, 30.0]})  # both out of range
    pd.concat([follower, leaderless, tooshort]).to_csv(folder / f"{rec}_tracks.csv", index=False)


def test_highd_samples_1hz_and_filters(tmp_path):
    for rec in ("01", "02"):
        _write_highd(tmp_path, rec)
    for workers in (1, 2):  # inline and process pool
        h = load_highd(tmp_path, workers=workers)
        assert h.tolist() == [2.0] * 8  # frames 25/50/75/100 x 2 recordings


def test_ngsim_samples_1hz_and_filters(tmp_path):
    from headway.datasets.ngsim import load_ngsim

    csv_path = tmp_path / "trajectories.csv"
    frames = np.arange(1, 51)
    df = pd.DataFrame({
        "Frame_ID": frames,
        "Preceding": [2] * 50,
        "Time_Headway": [2.5] * 50,
        "Location": ["i-80"] * 50,
    })
    # Add leaderless and invalid rows
    df.loc[df["Frame_ID"] == 20, "Preceding"] = 0
    df.loc[df["Frame_ID"] == 30, "Time_Headway"] = 0.2  # below 0.5s floor
    df.to_csv(csv_path, index=False)

    h = load_ngsim(csv_path, location="i-80", hz=10)
    # Frames 10, 40, 50 should be valid (20 has Preceding=0, 30 has thw=0.2)
    assert len(h) == 3
    assert np.allclose(h, 2.5)
