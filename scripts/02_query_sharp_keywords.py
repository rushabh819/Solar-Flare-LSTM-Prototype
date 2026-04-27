from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

"""
Small SHARP test query.
Use this only for 1-day smoke tests. For multi-day acquisition use
02e_fetch_sharp_range_chunked.py instead.
"""

import drms
import pandas as pd

OUT_PATH = Path("data/raw/sharp_keywords.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

QUERY = "hmi.sharp_cea_720s_nrt[][2026.04.16_00:00:00_TAI/1d@12m]"
KEYS = [
    "HARPNUM", "T_REC", "NOAA_ARS", "USFLUX", "TOTUSJH", "TOTUSJZ", "ABSNJZH",
    "SAVNCPP", "TOTPOT", "R_VALUE", "MEANGAM", "MEANGBT", "MEANGBZ", "MEANGBH",
    "MEANJZH", "MEANJZD", "MEANALP", "MEANPOT", "MEANSHR", "AREA", "LAT_FWT", "LON_FWT",
]


def main():
    client = drms.Client()
    df = client.query(QUERY, key=KEYS)
    if isinstance(df, pd.DataFrame) and not df.empty:
        df.to_csv(OUT_PATH, index=False)
        print(f"Saved {len(df)} SHARP rows to {OUT_PATH}")
    else:
        raise RuntimeError("JSOC returned no SHARP rows. Adjust the query string.")


if __name__ == "__main__":
    main()
