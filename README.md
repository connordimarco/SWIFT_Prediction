# SWIFT_Prediction

Daily forecasts, 1–30 days ahead, of the two drivers a satellite-drag
(re-entry) calculation needs: the F10.7 solar radio flux and the geomagnetic
Ap index. Built for the CCMC Swift re-entry challenge.

Both use one LightGBM per lead day on the last 60 days of
raw daily series, one fixed parameter set, trained through 2021, early-
stopped on 2022–23, scored on 2024 onward (cycle-25 maximum), with an
uncertainty band taken from past forecast errors.

## F10.7

Inputs: F10.7, sunspot number, SWPC active-region summaries and GONG/JSOC
far-side detections; the target is the ratio of future flux to the current
81-day mean. Trained 1947–2021. Test: pooled RMSE 29.1 sfu; against SWPC's
operational 27-day outlook on 125 matched issues, 28.6 vs 32.1 sfu (−11%),
32.2 vs 37.3 on days ≥150 sfu (−14%).

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

- `scripts/build_ap.py` — download the GFZ file → `data/ap_daily.csv`.
- `ap/` — `shared.py` (windows, splits, the band), `train.py` (30 boosters +
  the error record behind the band, ~15 min), `score.py`, `plots.py`,
  `predict.py` (issue a forecast → `ap/forecasts/`); saved boosters in
  `ap/models/`, test predictions, scorecard and figures in `ap/out/`.
  Reproduce: `env/bin/python scripts/build_ap.py && cd ap && ../env/bin/python train.py && ../env/bin/python score.py && ../env/bin/python plots.py`.
  Daily: `env/bin/python scripts/build_ap.py && (cd ap && ../env/bin/python predict.py)`,
  any time after 00 UT.

Env: `python3 -m venv env && env/bin/pip install -r requirements.txt`
(LightGBM needs Homebrew `libomp`).
