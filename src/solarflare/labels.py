from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import normalize_noaa_ars

CLASS_BASE = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}
CLASS_ORDER = {"A": 0, "B": 1, "C": 2, "M": 3, "X": 4}


@dataclass
class LabelResult:
    table: pd.DataFrame
    multiclass_col: str
    binary_col: str


def goes_class_to_flux(goes_class: str) -> float:
    text = str(goes_class).strip().upper()
    match = re.match(r"([ABCMX])(\d+(?:\.\d+)?)", text)
    if not match:
        return np.nan
    letter, magnitude = match.groups()
    return CLASS_BASE[letter] * float(magnitude)


def goes_class_rank(goes_class: str) -> int:
    text = str(goes_class).strip().upper()
    match = re.match(r"([ABCMX])", text)
    if not match:
        return -1
    return CLASS_ORDER[match.group(1)]


def threshold_to_rank(min_class: str) -> int:
    rank = goes_class_rank(min_class)
    if rank < 0:
        raise ValueError(f"Invalid min_class: {min_class}")
    return rank


def build_future_labels(
    sharp_df: pd.DataFrame,
    events_df: pd.DataFrame,
    horizon_hours: int = 24,
    min_class: str = "M1.0",
) -> LabelResult:
    horizon = pd.Timedelta(hours=horizon_hours)
    min_rank = threshold_to_rank(min_class)

    events = events_df.copy()
    events["class_rank"] = events["goes_class"].map(goes_class_rank)
    events["class_flux"] = events["goes_class"].map(goes_class_to_flux)

    rows = []
    for row in sharp_df.itertuples(index=False):
        row_dict = row._asdict()
        noaa_candidates = normalize_noaa_ars(row_dict.get("NOAA_ARS"))
        t0 = row_dict["T_REC"]
        t1 = t0 + horizon

        if noaa_candidates:
            evt = events[
                (events["noaa_ar"].isin(noaa_candidates))
                & (events["peak_time"] > t0)
                & (events["peak_time"] <= t1)
            ]
        else:
            evt = events.iloc[0:0]

        if len(evt) == 0:
            max_rank = -1
            max_class = "quiet"
            binary = 0
        else:
            idx = evt["class_rank"].idxmax()
            max_rank = int(evt.loc[idx, "class_rank"])
            max_class = str(evt.loc[idx, "goes_class"])
            binary = int(max_rank >= min_rank)

        row_dict["max_future_class"] = max_class
        row_dict["max_future_rank"] = max_rank
        row_dict["target"] = binary
        row_dict["target_multiclass"] = max(max_rank, 0)
        rows.append(row_dict)

    out = pd.DataFrame(rows)
    return LabelResult(table=out, multiclass_col="target_multiclass", binary_col="target")
