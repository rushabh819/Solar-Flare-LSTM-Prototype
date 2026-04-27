from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from datetime import datetime, timedelta
import time

import drms
import pandas as pd

OUT_PATH = Path("data/raw/sharp_keywords.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

SERIES = "hmi.sharp_cea_720s_nrt"
KEYS = [
    "HARPNUM", "T_REC", "NOAA_ARS", "USFLUX", "TOTUSJH", "TOTUSJZ", "ABSNJZH",
    "SAVNCPP", "TOTPOT", "R_VALUE", "MEANGAM", "MEANGBT", "MEANGBZ", "MEANGBH",
    "MEANJZH", "MEANJZD", "MEANALP", "MEANPOT", "MEANSHR", "AREA", "LAT_FWT", "LON_FWT",
]

START_DATE = "2026-04-16"
END_DATE = "2026-04-22"   # inclusive


def daterange(start_date: datetime, end_date: datetime):
    cur = start_date
    while cur <= end_date:
        yield cur
        cur += timedelta(days=1)


def build_query(day: datetime) -> str:
    day_str = day.strftime("%Y.%m.%d_00:00:00_TAI")
    return f"{SERIES}[][{day_str}/1d@12m]"


def main():
    client = drms.Client()
    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_dt = datetime.strptime(END_DATE, "%Y-%m-%d")

    all_parts = []

    for day in daterange(start_dt, end_dt):
        query = build_query(day)
        print(f"\nQuerying: {query}")
        try:
            df = client.query(query, key=KEYS)
            if df is None or len(df) == 0:
                print(f"No rows returned for {day.date()}")
                continue

            print(f"Rows for {day.date()}: {len(df)}")
            all_parts.append(df.copy())
            time.sleep(1)
        except Exception as e:
            print(f"Failed for {day.date()}: {e}")

    if not all_parts:
        raise RuntimeError("No SHARP rows were downloaded for the requested range.")

    sharp_df = pd.concat(all_parts, ignore_index=True)
    sharp_df = sharp_df.drop_duplicates(subset=["HARPNUM", "T_REC"]).reset_index(drop=True)
    sharp_df.to_csv(OUT_PATH, index=False)

    print("\nSaved combined SHARP file to:", OUT_PATH)
    print("Total rows:", len(sharp_df))
    print("Columns:", sharp_df.columns.tolist())


if __name__ == "__main__":
    main()
