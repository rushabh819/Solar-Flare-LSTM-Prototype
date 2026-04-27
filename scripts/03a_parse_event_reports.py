from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pathlib import Path
import re
import pandas as pd

INPUT_DIR = Path("data/raw/event_reports")
OUTPUT_PATH = Path("data/raw/flare_events.csv")

XRA_PATTERN = re.compile(
    r"(?P<event_id>\d{4})\s+"
    r"(?:\+\s+)?"
    r"(?P<begin>\d{4}|////)\s+"
    r"(?P<max>\d{4}|////)\s+"
    r"(?P<end>\d{4}|////)\s+"
    r"(?P<obs>\S+)\s+"
    r"(?P<q>\S+)\s+"
    r"XRA\s+"
    r"1-8A\s+"
    r"(?P<goes_class>[ABCMX]\d+\.\d)\s+"
    r"(?P<intensity>\S+)\s+"
    r"(?P<noaa_ar>\d{4,5})"
)


def normalize_noaa_ar(noaa_ar: int) -> int:
    if 0 < noaa_ar < 10000:
        return noaa_ar + 10000
    return noaa_ar


def hhmm_to_timestamp(date_str: str, hhmm: str):
    if hhmm == "////":
        return pd.NaT
    hh = hhmm[:2]
    mm = hhmm[2:]
    return pd.Timestamp(f"{date_str} {hh}:{mm}:00", tz="UTC")


def extract_date_from_filename(path: Path) -> str:
    stem = path.name[:8]
    return f"{stem[:4]}-{stem[4:6]}-{stem[6:8]}"


def parse_event_report(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="ignore")
    flat = " ".join(text.split())
    date_str = extract_date_from_filename(path)
    rows = []

    for m in XRA_PATTERN.finditer(flat):
        goes_class = m.group("goes_class")
        noaa_ar = normalize_noaa_ar(int(m.group("noaa_ar")))

        peak_hhmm = m.group("max")
        if peak_hhmm == "////":
            peak_hhmm = m.group("begin")

        peak_time = hhmm_to_timestamp(date_str, peak_hhmm)
        if pd.isna(peak_time):
            continue

        rows.append({
            "peak_time": peak_time.isoformat().replace("+00:00", "Z"),
            "noaa_ar": noaa_ar,
            "goes_class": goes_class,
            "source_file": path.name,
        })

    return pd.DataFrame(rows)


def main():
    files = sorted(INPUT_DIR.glob("*events.txt"))
    if not files:
        raise FileNotFoundError(f"No event report files found in {INPUT_DIR}")

    dfs = [parse_event_report(p) for p in files]
    flare_df = pd.concat(dfs, ignore_index=True)

    if flare_df.empty:
        raise RuntimeError("No XRA flare rows were parsed from the event reports.")

    flare_df = flare_df.drop_duplicates(subset=["peak_time", "noaa_ar", "goes_class"])
    flare_df = flare_df.sort_values("peak_time").reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    flare_df[["peak_time", "noaa_ar", "goes_class"]].to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {OUTPUT_PATH}")
    print(flare_df.head(20).to_string(index=False))
    print("\nShape:", flare_df.shape)
    print("\nClass counts:")
    print(flare_df["goes_class"].str[0].value_counts().sort_index())


if __name__ == "__main__":
    main()
