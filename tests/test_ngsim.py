import pandas as pd

from headway.datasets.ngsim import load_ngsim


def test_ngsim_filters(tmp_path):
    rows = [  # Frame_ID, Preceding, Time_Headway, Location, Lane_ID
        ('"1,310"', 5, 2.0, "i-80", 1),     # kept (frame 1310 % 10 == 0, comma thousands)
        (1320, 5, 3.0, "i-80", 2),          # kept
        (1325, 5, 4.0, "i-80", 2),          # dropped: not a 1 Hz sample
        (1330, 0, 0.0, "i-80", 2),          # dropped: no leader
        (1340, 5, 0.3, "i-80", 2),          # dropped: < 0.5 s
        (1350, 5, 2.5, "us-101", 2),        # dropped: other location
    ]
    p = tmp_path / "e.csv"
    p.write_text("Frame_ID,Preceding,Time_Headway,Location,Lane_ID\n"
                 + "\n".join(",".join(map(str, r)) for r in rows))
    assert sorted(load_ngsim(p, location="I-80")) == [2.0, 3.0]
    assert load_ngsim(p, location="i-80", lanes=[2]).tolist() == [3.0]


def test_ngsim_period_splits_on_time_gaps(tmp_path):
    hour = 3_600_000  # recording periods are separated by gaps far above the 10 min threshold
    rows = [(10, 2.0, 0), (20, 2.5, 1_000), (30, 3.0, hour), (40, 3.5, hour + 1_000), (50, 4.0, 2 * hour)]
    p = tmp_path / "e.csv"
    p.write_text("Frame_ID,Preceding,Time_Headway,Location,Global_Time\n"
                 + "\n".join(f"{f},5,{h},i-80,{t}" for f, h, t in rows))
    assert load_ngsim(p, period=1).tolist() == [2.0, 2.5]
    assert load_ngsim(p, period=2).tolist() == [3.0, 3.5]
    assert load_ngsim(p, period=3).tolist() == [4.0]
    assert len(load_ngsim(p)) == 5
