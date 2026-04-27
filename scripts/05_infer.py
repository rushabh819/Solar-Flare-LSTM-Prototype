from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import json
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

from solarflare.sequences import build_lstm_sequences


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_csv", required=True)
    p.add_argument("--model_path", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    meta_path = model_path.with_name(model_path.stem + "_meta.json")
    scaler_path = model_path.with_name(model_path.stem + "_scaler.joblib")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    sequence_length = int(meta["sequence_length"])
    feature_columns = list(meta["feature_columns"])
    threshold = float(meta["threshold"])

    df = pd.read_csv(args.dataset_csv)
    df["T_REC"] = pd.to_datetime(df["T_REC"], utc=True, errors="coerce")
    df = df.dropna(subset=["T_REC"]).sort_values(["HARPNUM", "T_REC"]).reset_index(drop=True)

    bundle = build_lstm_sequences(df, feature_columns=feature_columns, label_col="target", sequence_length=sequence_length)
    latest_idx = bundle.sample_times.groupby(bundle.harpnumbers).idxmax().values
    X_latest = bundle.X[latest_idx]
    harp_latest = bundle.harpnumbers.iloc[latest_idx].to_numpy()
    time_latest = bundle.sample_times.iloc[latest_idx].to_numpy()

    scaler = joblib.load(scaler_path)
    n, s, f = X_latest.shape
    X_latest_s = scaler.transform(X_latest.reshape(-1, f)).reshape(n, s, f)

    model = keras.models.load_model(model_path, compile=False)
    prob = model.predict(X_latest_s, verbose=0).reshape(-1)
    pred = (prob >= threshold).astype(int)

    out = pd.DataFrame(
        {
            "HARPNUM": harp_latest,
            "T_REC": time_latest,
            "flare_risk": prob,
            "alert": pred,
            "confidence": np.where(pred == 1, prob, 1 - prob),
        }
    ).sort_values("flare_risk", ascending=False)

    out_path = Path("reports/inference_output.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(out.head(20))
    print(f"Saved inference output to {out_path}")


if __name__ == "__main__":
    main()
