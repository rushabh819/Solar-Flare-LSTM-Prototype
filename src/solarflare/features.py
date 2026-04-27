from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT_GOES_FEATURES
from .utils import safe_log10

GOES_ROLLING_WINDOWS = {
    "1h": "1h",
    "6h": "6h",
    "24h": "24h",
}


def normalize_noaa_ars(value) -> list[int]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.upper() == "MISSING":
        return []
    text = text.replace(";", ",").replace(" ", ",")
    out = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(int(token))
        except ValueError:
            continue
    return sorted(set(out))


def add_goes_rolling_features(goes_df: pd.DataFrame) -> pd.DataFrame:
    df = goes_df.copy().sort_values("time_tag")
    df = df.set_index("time_tag")

    df["xrsa_log10"] = safe_log10(df["xrsa"])
    df["xrsb_log10"] = safe_log10(df["xrsb"])
    df["goes_xrsa_xrsb_ratio"] = df["xrsa"] / np.clip(df["xrsb"], 1e-12, None)
    df["goes_xrsb_last"] = df["xrsb_log10"]
    df["goes_xrsa_last"] = df["xrsa_log10"]

    for label, win in GOES_ROLLING_WINDOWS.items():
        df[f"goes_xrsb_{label}_mean"] = df["xrsb_log10"].rolling(win, min_periods=1).mean()
        df[f"goes_xrsb_{label}_max"] = df["xrsb_log10"].rolling(win, min_periods=1).max()

    df["goes_xrsb_trend_1h"] = (
        df["xrsb_log10"] - df["xrsb_log10"].shift(60)
        if len(df) > 60
        else df["xrsb_log10"].diff().fillna(0.0)
    )
    df["goes_xrsb_trend_1h"] = df["goes_xrsb_trend_1h"].fillna(0.0)

    keep_cols = [c for c in DEFAULT_GOES_FEATURES if c in df.columns]
    out = df[keep_cols].reset_index()
    return out


def merge_sharp_with_goes(sharp_df: pd.DataFrame, goes_features_df: pd.DataFrame) -> pd.DataFrame:
    left = sharp_df.copy().sort_values("T_REC")
    right = goes_features_df.copy().sort_values("time_tag")

    left["T_REC"] = pd.to_datetime(left["T_REC"], errors="coerce").astype("datetime64[ns]")
    right["time_tag"] = pd.to_datetime(right["time_tag"], errors="coerce").astype("datetime64[ns]")

    left = left.dropna(subset=["T_REC"])
    right = right.dropna(subset=["time_tag"])

    merged = pd.merge_asof(
        left,
        right,
        left_on="T_REC",
        right_on="time_tag",
        direction="backward",
        tolerance=pd.Timedelta("24h"),
    )
    return merged
