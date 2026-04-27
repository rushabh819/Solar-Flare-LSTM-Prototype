from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

from solarflare.config import DatasetConfig
from solarflare.data_io import load_event_csv, load_goes_csv, load_sharp_csv
from solarflare.features import add_goes_rolling_features, merge_sharp_with_goes
from solarflare.labels import build_future_labels


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sharp_csv", required=True)
    p.add_argument("--goes_csv", required=True)
    p.add_argument("--events_csv", required=True)
    p.add_argument("--horizon_hours", type=int, default=24)
    p.add_argument("--min_class", type=str, default="M1.0")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = DatasetConfig(horizon_hours=args.horizon_hours, min_class=args.min_class)

    sharp_df = load_sharp_csv(args.sharp_csv)
    goes_df = load_goes_csv(args.goes_csv)
    events_df = load_event_csv(args.events_csv)

    goes_features = add_goes_rolling_features(goes_df)
    merged = merge_sharp_with_goes(sharp_df, goes_features)
    labeled = build_future_labels(
        merged,
        events_df,
        horizon_hours=cfg.horizon_hours,
        min_class=cfg.min_class,
    ).table

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = cfg.min_class.replace(".", "")
    out_path = out_dir / f"model_table_h{cfg.horizon_hours}_{suffix}.csv"
    labeled.to_csv(out_path, index=False)
    print(f"Saved dataset to {out_path} ({len(labeled)} rows)")


if __name__ == "__main__":
    main()
