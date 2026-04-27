from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class SequenceBundle:
    X: np.ndarray
    y: np.ndarray
    sample_times: pd.Series
    harpnumbers: pd.Series
    feature_columns: List[str]


def build_lstm_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    label_col: str = "target",
    sequence_length: int = 60,
) -> SequenceBundle:
    X_list = []
    y_list = []
    t_list = []
    h_list = []

    sorted_df = df.sort_values(["HARPNUM", "T_REC"]).reset_index(drop=True)

    for harpnum, g in sorted_df.groupby("HARPNUM"):
        g = g.dropna(subset=feature_columns + [label_col]).reset_index(drop=True)
        if len(g) < sequence_length:
            continue
        values = g[feature_columns].to_numpy(dtype=float)
        labels = g[label_col].to_numpy(dtype=int)
        times = g["T_REC"].to_numpy()

        for end_idx in range(sequence_length - 1, len(g)):
            start_idx = end_idx - sequence_length + 1
            X_list.append(values[start_idx : end_idx + 1])
            y_list.append(labels[end_idx])
            t_list.append(times[end_idx])
            h_list.append(harpnum)

    if not X_list:
        raise ValueError("No sequences were created. Check sequence_length and missing data.")

    return SequenceBundle(
        X=np.stack(X_list),
        y=np.asarray(y_list, dtype=int),
        sample_times=pd.to_datetime(pd.Series(t_list), utc=True),
        harpnumbers=pd.Series(h_list),
        feature_columns=feature_columns,
    )


def chronological_split(
    bundle: SequenceBundle,
    horizon_hours: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
):
    times = bundle.sample_times.sort_values().reset_index(drop=True)
    train_cut = times.iloc[int(len(times) * train_frac)]
    val_cut = times.iloc[int(len(times) * (train_frac + val_frac))]
    purge = pd.Timedelta(hours=horizon_hours)

    def make_masks(use_purge: bool):
        if use_purge:
            train_mask = bundle.sample_times <= (train_cut - purge)
            val_mask = (bundle.sample_times > train_cut) & (bundle.sample_times <= (val_cut - purge))
            test_mask = bundle.sample_times > val_cut
        else:
            train_mask = bundle.sample_times <= train_cut
            val_mask = (bundle.sample_times > train_cut) & (bundle.sample_times <= val_cut)
            test_mask = bundle.sample_times > val_cut
        return train_mask.to_numpy(), val_mask.to_numpy(), test_mask.to_numpy()

    def has_both_classes(mask):
        y = bundle.y[mask]
        return len(y) > 0 and len(np.unique(y)) == 2

    train_mask, val_mask, test_mask = make_masks(use_purge=True)

    if (
        val_mask.sum() == 0
        or test_mask.sum() == 0
        or not has_both_classes(val_mask)
        or not has_both_classes(test_mask)
    ):
        train_mask, val_mask, test_mask = make_masks(use_purge=False)

    return train_mask, val_mask, test_mask


def fit_and_apply_scaler(X_train: np.ndarray, X_other: list[np.ndarray]):
    n_train, seq_len, n_feat = X_train.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_feat))

    def transform(arr: np.ndarray) -> np.ndarray:
        n, s, f = arr.shape
        if n == 0:
            return arr
        return scaler.transform(arr.reshape(-1, f)).reshape(n, s, f)

    transformed = [transform(X_train)]
    for arr in X_other:
        transformed.append(transform(arr))
    return scaler, transformed
