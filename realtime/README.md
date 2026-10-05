# realtime — daily E24 F10.7 forecast (1–30 days)

The frozen single-LightGBM model E24 (`model/train_e24.py`) run as an operational
forecast. One file per issued forecast lands in `forecasts/`; that archive is
the record of what was predicted when (CCMC Swift re-entry challenge entry).

## Model (unchanged from `model/`)

- 30 LightGBM regressors, one per lead day, cfg-44 parameters.
- Inputs: the last 60 days of 26 daily series — flare-robust adjusted F10.7,
  SILSO sunspot number, 10 SWPC Solar Region Summary aggregates, 14 far-side
  (GONG f6x + JSOC SARD) aggregates. AR/far-side may be missing (optional-NaN).
- Target: flux(t+h) / 81-day trailing mean(t); multiplied back and converted
  from 1-AU adjusted to observed flux at predict time.
- Trained on origins 1947-05-03..2021-12-31, early-stopped on 2022–2023.
  Test (2024-01 →, 882 origins): pooled RMSE 29.1 sfu, r 0.70, skill vs
  persistence +0.25; vs SWPC's 27-day outlook on 125 matched issues:
  28.6 vs 32.1 sfu pooled (−11%), 32.2 vs 37.3 on days ≥150 sfu (−14%).
- Band: per-lead quantiles of log(observed/forecast) over all 2024→ origins,
  applied multiplicatively (`models/band.json`). In-sample on that era.

## Files

| file | role |
|---|---|
| `e24.py` | shared: feature layout, origin validity, model I/O, adj→obs |
| `train.py` | one-time: fit + save the 30 boosters → `models/` (~45 min, 10 cores) |
| `calibrate.py` | residual quantiles → `models/band.json` (rerun after refreshes) |
| `refresh_data.sh` | daily: re-pull LISIRD/SILSO/SRS/far-side, rebuild `data/*.csv`, bridge the tail |
| `bridge_tail.py` | fills the last days: SSN from SILSO EISN, F10.7 from SWPC if LISIRD lags |
| `predict.py` | issue the forecast for the last usable origin (or a given date) |
| `verify.py` | retrained == `model/out` predictions; live feature rows == training rows; band coverage |

## Daily run

```
sh realtime/refresh_data.sh && env/bin/python realtime/predict.py
```

Timing: LISIRD posts the 17/20/23 UT Penticton readings the same day and the
SRS for day t is issued at ~00:30 UT on t+1, so running at **01:30 UT
(21:30 ET)** gives a complete origin = the UT day just ended. Running earlier
still works (origin = yesterday or today with partial readings; the SRS
aggregates forward-fill one day, `srs_present=0`).

## Deploying on a Linux box (e.g. solsticedisk, RHEL 8, python3.12)

```
git clone https://github.com/connordimarco/SWIFT_Prediction.git && cd SWIFT_Prediction
python3.12 -m venv env && env/bin/pip install -r requirements.txt
rsync -a <mac>:Documents/Work/F10.7_Prediction/data/ data/ --exclude midl --exclude swpc_prf   # ~150 MB; or rebuild (far-side fetch ~4 h)
sh realtime/refresh_data.sh && env/bin/python realtime/verify.py && env/bin/python realtime/predict.py
```

The models load from `lead_NN.joblib` (needs the pinned versions) or fall back
to the portable `lead_NN.txt` boosters; `verify.py` step 4 proves both agree.
Then a crontab line, e.g. `30 21 * * * cd <repo> && sh realtime/refresh_data.sh && env/bin/python realtime/predict.py >> realtime/out/daily.log 2>&1`
(21:30 ET; adjust for the box's timezone).

Freeze: do not retrain before Swift re-enters. `models/` is gitignored
(rebuild with `train.py`, bit-identical — `verify.py` checks); `forecasts/`
is tracked.
