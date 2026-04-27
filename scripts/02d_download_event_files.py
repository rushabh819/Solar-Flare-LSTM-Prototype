from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pathlib import Path
import requests

BASE = "https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/solar_event_reports/2026/04"
FILES = ["20260416events.txt", "20260417events.txt"]

out_dir = Path("data/raw/event_reports")
out_dir.mkdir(parents=True, exist_ok=True)

for name in FILES:
    url = f"{BASE}/{name}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    path = out_dir / name
    path.write_text(r.text, encoding="utf-8")
    print(f"Saved {path} from {url}")
