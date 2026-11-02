"""Where does an NGSIM sample's mean headway come from?

    uv run python -m headway.ngsim_breakdown data/ngsim/export.csv [location]

Applies the same filters as the loader (1 Hz, leader present, 0.5-25 s), then prints
count / mean / median headway per lane, per 15-minute window and per follower speed.
Use it to compare against the paper's NGSIM sample (mean ~2.9 s implied by Table 3).
"""
import sys

import pandas as pd


def main(path, location="i-80"):
    df = pd.read_csv(path, thousands=",", on_bad_lines="skip",
                     usecols=["Frame_ID", "Preceding", "Time_Headway", "Location", "Lane_ID",
                              "Global_Time", "v_Vel"])
    df = df[df.Location.str.lower() == location.lower()]
    df = df[(df.Frame_ID % 10 == 0) & (df.Preceding > 0) & df.Time_Headway.between(0.5, 25)]
    df["window"] = pd.to_datetime(df.Global_Time, unit="ms").dt.floor("15min").dt.strftime("%H:%M")
    df["speed ft/s"] = pd.cut(df.v_Vel, [-1, 10, 20, 30, 40, 1000])
    print(f"{location}: {len(df):,} headways, mean {df.Time_Headway.mean():.2f}s, "
          f"median {df.Time_Headway.median():.2f}s")
    for key in ("Lane_ID", "window", "speed ft/s"):
        print(f"\nby {key}")
        print(df.groupby(key, observed=True).Time_Headway.agg(["count", "mean", "median"]).round(2))


if __name__ == "__main__":
    main(*sys.argv[1:])
