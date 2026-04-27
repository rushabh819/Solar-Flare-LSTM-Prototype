# Solar Flare LSTM Prototype (Finalized)

This repository is the **final corrected prototype package** for region-level solar flare forecasting using:

- **SDO/HMI SHARP** active-region magnetic parameters
- **GOES XRS** soft X-ray flux data
- **SWPC daily event reports** for flare labeling
- **LSTM** sequence modeling for **24-hour flare prediction**

It consolidates all major corrections made during development, including:

- chunked SHARP download to avoid JSOC memory failures
- datetime parsing fixes for JSOC `*_TAI` and ISO UTC timestamps
- safe SHARP ↔ GOES merge key normalization
- SWPC region-number normalization (`4419 -> 14419` style mapping)
- event-window dataset filtering for meaningful chronological evaluation
- configurable train/validation fractions
- robust split fallback when purged validation/test windows lose positive examples

---

## 1. Final objective

The final working task is:

- **horizon:** 24 hours
- **label threshold:** `C1.0+`
- **prediction type:** binary sequence classification

For each SHARP region-time record:

- `target = 1` if the active region produces a flare of class **C1.0 or higher** within the next **24 hours**
- `target = 0` otherwise

---

## 2. Recommended project structure

```text
solar_flare_lstm_prototype_final/
├── data/
│   ├── raw/
│   │   ├── sharp_keywords.csv
│   │   ├── goes_xrs.csv
│   │   ├── flare_events.csv
│   │   └── event_reports/
│   └── processed/
│       ├── model_table_h24_M10.csv
│       ├── model_table_h24_C10.csv
│       └── model_table_h24_C10_event_window.csv
├── models/
│   ├── lstm_h24.keras
│   ├── lstm_h24_scaler.joblib
│   └── lstm_h24_meta.json
├── reports/
│   ├── inference_output.csv
│   └── prototype_results_summary.json
├── scripts/
│   ├── 01_fetch_goes_recent.py
│   ├── 02_query_sharp_keywords.py
│   ├── 02d_download_event_files.py
│   ├── 02e_fetch_sharp_range_chunked.py
│   ├── 02f_download_event_reports_range.py
│   ├── 03a_parse_event_reports.py
│   ├── 03_build_dataset.py
│   ├── 04_train.py
│   └── 05_infer.py
├── src/
│   └── solarflare/
│       ├── config.py
│       ├── data_io.py
│       ├── features.py
│       ├── labels.py
│       ├── metrics.py
│       ├── model.py
│       ├── sequences.py
│       └── utils.py
├── requirements.txt
└── README.md
```

---

## 3. Setup

### Windows PowerShell

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Optional syntax check

```bash
python -m py_compile src/solarflare/config.py
python -m py_compile src/solarflare/data_io.py
python -m py_compile src/solarflare/features.py
python -m py_compile src/solarflare/labels.py
python -m py_compile src/solarflare/sequences.py
python -m py_compile src/solarflare/model.py
python -m py_compile src/solarflare/metrics.py
python -m py_compile scripts/01_fetch_goes_recent.py
python -m py_compile scripts/02_query_sharp_keywords.py
python -m py_compile scripts/02e_fetch_sharp_range_chunked.py
python -m py_compile scripts/02f_download_event_reports_range.py
python -m py_compile scripts/03a_parse_event_reports.py
python -m py_compile scripts/03_build_dataset.py
python -m py_compile scripts/04_train.py
python -m py_compile scripts/05_infer.py
```

---

## 4. Data sources and expected schemas

### 4.1 SHARP keyword table

Expected core columns:

- `HARPNUM`
- `T_REC`
- `NOAA_ARS`
- `USFLUX`
- `TOTUSJH`
- `TOTUSJZ`
- `ABSNJZH`
- `SAVNCPP`
- `TOTPOT`
- `R_VALUE`
- `MEANGAM`
- `MEANGBT`
- `MEANGBZ`
- `MEANGBH`
- `MEANJZH`
- `MEANJZD`
- `MEANALP`
- `MEANPOT`
- `MEANSHR`
- `AREA`
- `LAT_FWT`
- `LON_FWT`

### 4.2 GOES XRS CSV

Expected schema:

- `time_tag`
- `xrsa`
- `xrsb`

### 4.3 Parsed flare events CSV

Expected schema:

- `peak_time`
- `noaa_ar`
- `goes_class`

---

## 5. End-to-end run order

## Step 1 — Fetch recent GOES XRS data

```bash
python scripts/01_fetch_goes_recent.py
```

Outputs include:

- `data/raw/goes_xrs.csv`
- `data/raw/goes_xrs_raw.json`
- `data/raw/goes_flares_raw.json`
- `data/raw/goes_recent_flares.csv`

---

## Step 2 — Fetch SHARP safely in daily chunks

Use the chunked downloader instead of one huge JSOC query.

```bash
python scripts/02e_fetch_sharp_range_chunked.py
```

Final working development run used:

- **date range:** `2026-04-16` to `2026-04-22`
- **total SHARP rows:** `9051`

### Why this matters

A large direct query to JSOC can fail with an out-of-memory error. During development, this happened when querying too much SHARP data at once, which is why this repo now uses daily chunking instead. That earlier failure is documented in the development logs. fileciteturn0file0

---

## Step 3 — Download event reports for the same range

```bash
python scripts/02f_download_event_reports_range.py
```

This downloads daily SWPC event reports to:

- `data/raw/event_reports/`

The finalized script includes a fallback for the final spillover day by using the current SWPC report if the archive day is not yet published.

---

## Step 4 — Parse event reports into structured flare labels

```bash
python scripts/03a_parse_event_reports.py
```

This creates:

- `data/raw/flare_events.csv`

### Final parser corrections included

- extracts `XRA 1-8A` rows from SWPC text reports
- uses the event `max` time as `peak_time`
- normalizes SWPC region numbers like `4419` to `14419`

Final working event parse in the development window produced:

- **57 flare-event rows**
- observed classes: **B** and **C**
- no **M** flares in that small window

---

## Step 5 — Build the merged dataset

### Example: C1.0+ labels

```bash
python scripts/03_build_dataset.py --sharp_csv data/raw/sharp_keywords.csv --goes_csv data/raw/goes_xrs.csv --events_csv data/raw/flare_events.csv --horizon_hours 24 --min_class C1.0
```

This creates:

- `data/processed/model_table_h24_C10.csv`

In the working run, this produced:

- **9051 rows**
- **28 HARPs**
- **381 positives**
- **8670 negatives**

### Note on M1.0

The M1.0 version can still be generated, but in the small development window it produced all-zero labels because no M-class events were present.

---

## Step 6 — Create an event-rich filtered dataset window

Because positives were concentrated on a subset of dates, the final meaningful experiment used a filtered window:

```bash
python -c "import pandas as pd; df=pd.read_csv('data/processed/model_table_h24_C10.csv'); df['T_REC']=pd.to_datetime(df['T_REC'], errors='coerce'); out=df[df['T_REC'] < '2026-04-21'].copy(); out.to_csv('data/processed/model_table_h24_C10_event_window.csv', index=False); print('ROWS:', len(out))"
```

This creates:

- `data/processed/model_table_h24_C10_event_window.csv`

---

## Step 7 — Train the final event-window model

```bash
python scripts/04_train.py --dataset_csv data/processed/model_table_h24_C10_event_window.csv --sequence_length 60 --cadence_minutes 12 --train_frac 0.50 --val_frac 0.25 --epochs 8
```

### Final training configuration

- **sequence length:** `60`
- **cadence:** `12 minutes`
- **effective history:** ~`12 hours`
- **train fraction:** `0.50`
- **validation fraction:** `0.25`
- **test fraction:** remaining `0.25`

### Final sequence split used in the successful run

- **Train:** 2607 sequences (`2417` negative, `190` positive)
- **Validation:** 1300 sequences (`1229` negative, `71` positive)
- **Test:** 1295 sequences (`1246` negative, `49` positive)

### Why the split logic was corrected

This repo includes fallback logic in `chronological_split()` to handle cases where:

- purge windows remove the whole validation block
- validation/test windows contain only one class

If that happens, the code falls back to a no-purge chronological split for short smoke-test windows.

### Outputs

Training produces:

- `models/lstm_h24.keras`
- `models/lstm_h24_scaler.joblib`
- `models/lstm_h24_meta.json`

---

## Step 8 — Run inference

```bash
python scripts/05_infer.py --dataset_csv data/processed/model_table_h24_C10_event_window.csv --model_path models/lstm_h24.keras
```

This creates:

- `reports/inference_output.csv`

---

## 6. Final working results

## 6.1 Validation-best threshold

The selected threshold from validation was:

- **threshold:** `0.05`

## 6.2 Validation-best metrics

- **Precision:** `0.5820`
- **Recall:** `1.0000`
- **F1:** `0.7358`
- **PR-AUC:** `1.0000`
- **ROC-AUC:** `1.0000`
- **TSS:** `0.9585`
- **HSS:** `0.7162`

## 6.3 Test metrics

- **Precision:** `0.3920`
- **Recall:** `1.0000`
- **F1:** `0.5632`
- **PR-AUC:** `0.9816`
- **ROC-AUC:** `0.9992`
- **TSS:** `0.9390`
- **HSS:** `0.5381`

Test confusion counts:

- **TP:** `49`
- **TN:** `1170`
- **FP:** `76`
- **FN:** `0`

### Interpretation

This is a **high-recall alerting prototype**:

- it missed **no positive events** in test
- it produced some false alarms
- it is more suitable as an **early warning / triage** system than a highly conservative operational alert generator

---

## 7. Final inference behavior

In the successful inference run, the model produced **21 latest active-region forecasts**.

The top-ranked region was:

- **HARPNUM:** `13451`
- **timestamp:** `2026-04-20 23:48:00+00:00`
- **flare_risk:** `0.5567485`
- **alert:** `1`

All remaining regions were much lower, mostly around `0.02–0.03`.

### Threshold comparison from the final run

At thresholds:

- `0.05`
- `0.10`
- `0.20`
- `0.30`
- `0.40`
- `0.50`

The number of alerts remained:

- **1 alert**

This shows the model produced **one clearly separated top candidate** in the final inference snapshot.

---

## 8. Important limitations

### 8.1 Short development window

The final meaningful run was built on a compact event-rich window, so results should be treated as prototype validation, not production-ready benchmarking.

### 8.2 Many rows with `NOAA_ARS = MISSING`

Some SHARP rows do not map cleanly to NOAA region identifiers, which limits perfect region-aware labeling.

### 8.3 High-recall thresholding

The selected threshold (`0.05`) is intentionally aggressive and favors sensitivity over precision.

### 8.4 Top alert can still be a false positive

In the final sanity check, the top inferred row was still labeled `target = 0` at the latest available labeled timestamp. That means the model learned useful ranking behavior, but the final alert calibration still needs improvement.

---

## 9. All major corrections applied in this finalized version

This final package includes the fixes for every major issue encountered during development:

1. **JSOC large-query memory failure**
   - fixed by daily chunked SHARP queries

2. **Datetime parsing failures**
   - fixed by supporting both JSOC `_TAI` strings and ISO UTC strings in `utils.py`

3. **`merge_asof` dtype mismatch**
   - fixed by forcing both merge keys to the same datetime dtype in `features.py`

4. **Empty merged datasets / empty CSV output**
   - fixed after correcting SHARP datetime parsing

5. **Region-number mismatch (`4419` vs `14419`)**
   - fixed in `03a_parse_event_reports.py`

6. **All-zero M1.0 labels in the small date window**
   - resolved by switching the finalized prototype run to **C1.0+**

7. **Validation split disappearing due to purge**
   - fixed via fallback logic in `sequences.py`

8. **Validation/test one-class issue in short event windows**
   - fixed via split fallback and event-window filtering

9. **Need for configurable split ratios**
   - fixed by adding `--train_frac` and `--val_frac` to `04_train.py`

---

## 10. Suggested next improvements

1. extend the date range substantially
2. evaluate `C1.0`, `M1.0`, and stronger thresholds
3. compare LSTM with GRU, Bi-LSTM, and Transformer baselines
4. calibrate probabilities for better alert thresholds
5. improve NOAA/SHARP mapping quality
6. build a dashboard for ranked active-region monitoring
7. add explainability and feature-importance analysis

---

## 11. Final prototype summary

This finalized prototype successfully demonstrates the full workflow:

- download solar telemetry
- parse event logs
- generate region-aware labels
- assemble a supervised time-series dataset
- train an LSTM classifier
- evaluate on a chronological split
- produce ranked flare-risk inference outputs

It is a working **research-grade prototype** for region-level solar flare forecasting and a strong foundation for future model and data improvements.
