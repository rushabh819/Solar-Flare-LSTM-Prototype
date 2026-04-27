from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import json

import pandas as pd
import requests

OUT_DIR = Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GOES_XRS_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"
GOES_FLARES_URL = "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json"


def fetch_json(url: str):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def main():
    goes_raw = fetch_json(GOES_XRS_URL)
    flare_raw = fetch_json(GOES_FLARES_URL)

    with open(OUT_DIR / "goes_xrs_raw.json", "w", encoding="utf-8") as f:
        json.dump(goes_raw, f, indent=2)
    with open(OUT_DIR / "goes_flares_raw.json", "w", encoding="utf-8") as f:
        json.dump(flare_raw, f, indent=2)

    goes_df = pd.DataFrame(goes_raw)
    if goes_df.empty:
        raise RuntimeError("GOES XRS file returned no rows.")

    goes_df = goes_df.pivot(index="time_tag", columns="energy", values="observed_flux").reset_index()
    goes_df = goes_df.rename(columns={"0.05-0.4nm": "xrsa", "0.1-0.8nm": "xrsb"})
    goes_df.to_csv(OUT_DIR / "goes_xrs.csv", index=False)

    flare_df = pd.DataFrame(flare_raw)
    if not flare_df.empty:
        flare_df.to_csv(OUT_DIR / "goes_recent_flares.csv", index=False)

    print("Saved:")
    print(OUT_DIR / "goes_xrs.csv")
    print(OUT_DIR / "goes_recent_flares.csv")


if __name__ == "__main__":
    main()
