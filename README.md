# SWIFT_Prediction

Daily forecasts, 1–30 days ahead, of the two drivers a satellite-drag
(re-entry) calculation needs: the F10.7 solar radio flux and the geomagnetic
Ap index. Built for the CCMC Swift re-entry challenge.

Both use the same recipe: one LightGBM per lead day on the last 60 days of
raw daily series, one fixed parameter set, trained through 2021, early-
stopped on 2022–23, scored on 2024 onward (cycle-25 maximum), with an
uncertainty band taken from past forecast errors.

## F10.7

Inputs: F10.7, sunspot number, SWPC active-region summaries and GONG/JSOC
far-side detections; the target is the ratio of future flux to the current
81-day mean. Trained 1947–2021. Test: pooled RMSE 29.1 sfu; against SWPC's
operational 27-day outlook on 125 matched issues, 28.6 vs 32.1 sfu (−11%),
32.2 vs 37.3 on days ≥150 sfu (−14%).

The band is multiplicative and depends on how active the Sun is: it comes
from the model's earlier out-of-sample errors at a similar 81-day flux level.
On the test years, each forecast's 90 / 80 / 50% bands, built from earlier
forecasts only, hold 91.1 / 81.9 / 49.5% of days (details and limits in
`realtime/README.md`).

- `scripts/` — rebuild `data/` from the raw archives (`fetch_raw.sh` →
  `parse_srs.py` → `build_dataset.py`; `fetch_farside.py` →
  `build_farside_daily.py`). Endpoints and archives: `data/SOURCES.md`.
- `model/` — the model: `train_e24.py` (one LightGBM per lead via
  `single_model.py`), `common.py` (data loading, windows, splits),
  `score_cell.py` (the scorer); test predictions and scorecard in `out/`.
  Reproduce: `env/bin/python model/train_e24.py && env/bin/python model/score_cell.py model`.
- `benchmarks/` — SWPC 27-day outlook comparison (`swpc_27day.py data/swpc_prf model`)
  and its per-lead plot.
- `realtime/` — the frozen model run daily: saved boosters, calibrated band,
  data refresh, predictor, verifier; issued forecasts archived in
  `realtime/forecasts/`. See `realtime/README.md`.
- `deploy/` — the daily cron wrapper and the website views built from it.

## Ap

Inputs: the eight 3-hourly ap values of each of the last 60 days, sunspot
number and F10.7 — all from GFZ Potsdam's one Kp/ap file, which starts in
1932 and updates daily. Trained 1932–2021. Test (978 forecast days,
2024-01 → 2026-09):

| Lead | RMSE | r | Climatology RMSE | Better than climatology |
|---|---|---|---|---|
| 1 day | 13.7 | 0.58 | 16.7 | 18% |
| 3 days | 16.1 | 0.28 | 16.7 | 4% |
| 7 days | 16.3 | 0.23 | 16.7 | 3% |
| 27 days | 16.3 | 0.22 | 16.7 | 2% |

What that means: tomorrow's Ap is forecast with real skill; from day 2 the
forecast is close to the long-term average, because nothing in these inputs
says when the next storm arrives. Storms are not forecast past day 1 (the
largest day-2 forecast is 32; on days with Ap ≥ 30 the forecast runs 36 too
low). Persistence and 27-day recurrence are both far worse than climatology
past day 1, so climatology is the reference to read.

The point forecast is the expected Ap, which is what accumulated drag needs.
Ap is skewed, so the typical day is lower: the `q50` column is the median.

The band is the part to rely on. It is multiplicative, cannot go below zero,
and is recalibrated for every forecast on the out-of-sample errors of the
previous 11 years (one solar cycle). On the test years the 90 / 80 / 50%
bands contain 89.3 / 80.2 / 50.5% of days; the 90% band holds 88–90% at
every lead and 89–90% in each of 2024, 2025 and 2026. Applied the same way
back to 1943 it averages 90.1%, with every year between 85% and 96%
(`ap/out/band_history.png`). Two limits: on the test years 3.6% of days fall
below the 90% band and 7.1% above it (5% each would be even), and storm days
are the upper tail the band is built to miss — 30% of Ap ≥ 30 days fall
inside it.

- `scripts/build_ap.py` — download the GFZ file → `data/ap_daily.csv`.
- `ap/` — `shared.py` (windows, splits, the band), `train.py` (30 boosters +
  the error record behind the band, ~15 min), `score.py`, `plots.py`,
  `predict.py` (issue a forecast → `ap/forecasts/`); saved boosters in
  `ap/models/`, test predictions, scorecard and figures in `ap/out/`.
  Reproduce: `env/bin/python scripts/build_ap.py && cd ap && ../env/bin/python train.py && ../env/bin/python score.py && ../env/bin/python plots.py`.
  Daily: `env/bin/python scripts/build_ap.py && (cd ap && ../env/bin/python predict.py)`,
  any time after 00 UT.

Tried and left out, each within noise of the inputs above on the test era:
Dst, measured solar-wind speed/Bz/density, active-region and far-side
aggregates, a shorter window of daily Ap. Training from 2005 instead of 1932
was clearly worse. A skilful forecast of solar-wind speed is the one input
with room to help at days 2–4.

Env: `python3 -m venv env && env/bin/pip install -r requirements.txt`
(LightGBM needs Homebrew `libomp`).
