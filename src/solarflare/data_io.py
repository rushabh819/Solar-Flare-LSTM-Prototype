from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import parse_datetime_col

REQUIRED_EVENT_COLUMNS = {"peak_time", "noaa_ar", "goes_class"}
REQUIRED_GOES_COLUMNS = {"time_tag", "xrsa", "xrsb"}
REQUIRED_SHARP_COLUMNS_MIN = {"HARPNUM", "T_REC", "NOAA_ARS"}


class DataSchemaError(ValueError):
    pass


def load_sharp_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_SHARP_COLUMNS_MIN - set(df.columns)
    if missing:
        raise DataSchemaError(f"SHARP CSV missing required columns: {sorted(missing)}")
    df = parse_datetime_col(df, "T_REC")
    df = df.sort_values(["HARPNUM", "T_REC"]).reset_index(drop=True)
    return df


def load_goes_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_GOES_COLUMNS - set(df.columns)
    if missing:
        raise DataSchemaError(f"GOES CSV missing required columns: {sorted(missing)}")
    df = parse_datetime_col(df, "time_tag")
    df = df.sort_values("time_tag").drop_duplicates("time_tag").reset_index(drop=True)
    return df


def load_event_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_EVENT_COLUMNS - set(df.columns)
    if missing:
        raise DataSchemaError(f"Event CSV missing required columns: {sorted(missing)}")
    df = parse_datetime_col(df, "peak_time")
    df["noaa_ar"] = pd.to_numeric(df["noaa_ar"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["noaa_ar", "goes_class"])
    df = df.sort_values("peak_time").reset_index(drop=True)
    return df
