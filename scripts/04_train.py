from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import joblib
import pandas as pd
from tensorflow import keras

from solarflare.config import DatasetConfig
from solarflare.metrics import evaluate_binary, optimize_threshold_for_tss
from solarflare.model import build_lstm_classifier
from solarflare.sequences import build_lstm_sequences, chronological_split, fit_and_apply_scaler
from solarflare.utils import save_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_csv", required=True)
    p.add_argument("--sequence_length", type=int, default=60)
    p.add_argument("--cadence_minutes", type=int, default=12)
    p.add_argument("--train_frac", type=float, default=0.50)
    p.add_argument("--val_frac", type=float, default=0.25)
    p.add_argument("--epochs", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = DatasetConfig(
        sequence_length=args.sequence_length,
        cadence_minutes=args.cadence_minutes,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )

    df = pd.read_csv(args.dataset_csv)
    df["T_REC"] = pd.to_datetime(df["T_REC"], utc=True, errors="coerce")
    df = df.dropna(subset=["T_REC"]).sort_values(["HARPNUM", "T_REC"]).reset_index(drop=True)

    feature_columns = [c for c in (cfg.sharp_features + cfg.goes_features) if c in df.columns]
    bundle = build_lstm_sequences(df, feature_columns=feature_columns, label_col=cfg.label_col, sequence_length=cfg.sequence_length)
    train_mask, val_mask, test_mask = chronological_split(
        bundle,
        horizon_hours=cfg.horizon_hours,
        train_frac=cfg.train_frac,
        val_frac=cfg.val_frac,
    )

    X_train, y_train = bundle.X[train_mask], bundle.y[train_mask]
    X_val, y_val = bundle.X[val_mask], bundle.y[val_mask]
    X_test, y_test = bundle.X[test_mask], bundle.y[test_mask]

    scaler, transformed = fit_and_apply_scaler(X_train, [X_val, X_test])
    X_train_s, X_val_s, X_test_s = transformed

    model = build_lstm_classifier(sequence_length=cfg.sequence_length, n_features=X_train_s.shape[-1])
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_pr_auc", mode="max", factor=0.5, patience=2),
    ]

    class_counts = pd.Series(y_train).value_counts().to_dict()
    pos = class_counts.get(1, 1)
    neg = class_counts.get(0, 1)
    class_weight = {0: 1.0, 1: max(1.0, neg / max(pos, 1))}

    history = model.fit(
        X_train_s,
        y_train,
        validation_data=(X_val_s, y_val),
        epochs=args.epochs,
        batch_size=64,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    val_prob = model.predict(X_val_s, verbose=0).reshape(-1)
    best = optimize_threshold_for_tss(y_val, val_prob)
    threshold = float(best["threshold"])

    test_prob = model.predict(X_test_s, verbose=0).reshape(-1)
    test_metrics = evaluate_binary(y_test, test_prob, threshold=threshold)

    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"lstm_h24.keras"
    scaler_path = out_dir / f"lstm_h24_scaler.joblib"
    meta_path = out_dir / f"lstm_h24_meta.json"

    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    save_json(
        {
            "sequence_length": cfg.sequence_length,
            "horizon_hours": cfg.horizon_hours,
            "feature_columns": feature_columns,
            "threshold": threshold,
            "validation_best": best,
            "test_metrics": test_metrics,
            "history": history.history,
        },
        meta_path,
    )

    print("Saved model artifacts:")
    print(model_path)
    print(scaler_path)
    print(meta_path)
    print("Test metrics:")
    print(test_metrics)


if __name__ == "__main__":
    main()
