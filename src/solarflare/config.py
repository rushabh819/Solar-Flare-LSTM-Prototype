from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

DEFAULT_SHARP_FEATURES = [
    "USFLUX",
    "TOTUSJH",
    "TOTUSJZ",
    "ABSNJZH",
    "SAVNCPP",
    "TOTPOT",
    "R_VALUE",
    "MEANGAM",
    "MEANGBT",
    "MEANGBZ",
    "MEANGBH",
    "MEANJZH",
    "MEANJZD",
    "MEANALP",
    "MEANPOT",
    "MEANSHR",
    "AREA",
]

DEFAULT_GOES_FEATURES = [
    "goes_xrsa_last",
    "goes_xrsb_last",
    "goes_xrsb_1h_mean",
    "goes_xrsb_6h_mean",
    "goes_xrsb_24h_mean",
    "goes_xrsb_1h_max",
    "goes_xrsb_6h_max",
    "goes_xrsb_24h_max",
    "goes_xrsa_xrsb_ratio",
    "goes_xrsb_trend_1h",
]

@dataclass
class DatasetConfig:
    horizon_hours: int = 24
    min_class: str = "M1.0"
    cadence_minutes: int = 12
    sequence_length: int = 60
    train_frac: float = 0.70
    val_frac: float = 0.15
    random_seed: int = 42
    sharp_features: List[str] = field(default_factory=lambda: DEFAULT_SHARP_FEATURES.copy())
    goes_features: List[str] = field(default_factory=lambda: DEFAULT_GOES_FEATURES.copy())
    harpn_col: str = "HARPNUM"
    time_col: str = "T_REC"
    noaa_col: str = "NOAA_ARS"
    label_col: str = "target"
