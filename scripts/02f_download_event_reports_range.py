from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from datetime import datetime, timedelta
import requests

BASE = "https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/solar_event_reports/2026/04"
OUT_DIR = Path("data/raw/event_reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2026-04-16"
END_DATE = "2026-04-23"   # include spillover day for 24h horizon
CURRENT_REPORT_URL = "https://services.swpc.noaa.gov/text/solar-geophysical-event-reports.txt"


def daterange(start_dt, end_dt):
    cur = start_dt
    while cur <= end_dt:
        yield cur
        cur += timedelta(days=1)


def main():
    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_dt = datetime.strptime(END_DATE, "%Y-%m-%d")

    for day in daterange(start_dt, end_dt):
        fname = day.strftime("%Y%m%d") + "events.txt"
        url = f"{BASE}/{fname}"
        path = OUT_DIR / fname

        r = requests.get(url, timeout=60)
        if r.status_code == 404 and fname == end_dt.strftime("%Y%m%d") + "events.txt":
            current = requests.get(CURRENT_REPORT_URL, timeout=60)
            current.raise_for_status()
            path.write_text(current.text, encoding="utf-8")
            print(f"Saved {path} from current SWPC report")
            continue

        r.raise_for_status()
        path.write_text(r.text, encoding="utf-8")
        print(f"Saved {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
