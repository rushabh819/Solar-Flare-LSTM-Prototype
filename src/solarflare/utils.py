from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def parse_datetime_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    s = out[col].astype(str).str.strip()

    parsed = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")

    # JSOC/DRMS timestamps, e.g. 2026.04.16_00:00:00_TAI
    mask_jsoc = s.str.endswith("_TAI", na=False)
    if mask_jsoc.any():
        jsoc_vals = s[mask_jsoc].str.replace("_TAI", "", regex=False)
        parsed.loc[mask_jsoc] = pd.to_datetime(
            jsoc_vals,
            format="%Y.%m.%d_%H:%M:%S",
            errors="coerce",
        )

    # Standard ISO timestamps, e.g. 2026-04-16T02:43:00Z
    mask_other = ~mask_jsoc
    if mask_other.any():
        parsed_other = pd.to_datetime(
            s[mask_other],
            errors="coerce",
            utc=True,
            format="mixed",
        ).dt.tz_convert("UTC").dt.tz_localize(None)
        parsed.loc[mask_other] = parsed_other

    out[col] = parsed
    out = out.dropna(subset=[col])
    return out


def safe_log10(x: np.ndarray | pd.Series, floor: float = 1e-12):
    return np.log10(np.clip(np.asarray(x, dtype=float), floor, None))
