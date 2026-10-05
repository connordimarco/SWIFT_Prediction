"""Daily Ap forecast, 1-30 days ahead: shared pieces for train / score / plots / predict.

One LightGBM per lead day (the same cfg-44 parameters as the F10.7 model) on
the last 60 days of the eight 3-hourly ap values, the sunspot number and
observed F10.7 — everything in GFZ's one Kp/ap file (scripts/build_ap.py).
Trained on origins 1932-2021 (F10.7 is NaN before 1947 and never gates an
origin), early-stopped on 2022-2023, tested from 2024.

The band is multiplicative and follows the solar cycle: per lead, quantiles
of log((observed + 1) / (forecast + 1)) over the out-of-sample forecasts
issued in the 11 years before the origin (band_quantiles). A band fitted to
the two validation years alone covered 82% where it claimed 90%.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "model"))
from single_model import CFG44  # noqa: E402,F401  (the one parameter set, shared with F10.7)

DATA = os.path.join(ROOT, "data")
MODELS = os.path.join(HERE, "models")
OUT = os.path.join(HERE, "out")
FORECASTS = os.path.join(HERE, "forecasts")
RECORD = os.path.join(MODELS, "error_record.npz")

HIST, HORIZON = 60, 30
AP3H = [f"ap{i}" for i in range(1, 9)]  # 00-03 UT .. 21-24 UT
COLS = AP3H + ["ssn", "f107_obs"]
NAMES = [f"{c}_lag{lag}" for c in COLS for lag in range(HIST - 1, -1, -1)]
SPLITS = {
    "train": ("1932-01-01", "2021-12-31"),
    "val": ("2022-01-01", "2023-12-31"),
    "test": ("2024-01-01", "2099-12-31"),
}
QS = [5, 10, 25, 50, 75, 90, 95]
WINDOW_YEARS = 11  # one solar cycle of forecast errors behind every band
MIN_RECORD = 200   # fewest past forecasts a band may be built from
STORM = 30.0       # Ap at or above this is scored separately


def load_table():
    df = pd.read_csv(os.path.join(DATA, "ap_daily.csv"), index_col="time", parse_dates=True)
    assert (df.index[1:] - df.index[:-1] == pd.Timedelta(days=1)).all(), "ap_daily.csv is not a gap-free daily grid"
    return df


def valid_origins(df, lo, hi, need_truth):
    """Origins in [lo, hi] whose 60-day 3-hourly ap window is gap-free (and,
    if need_truth, whose 30-day target window is complete)."""
    ok = df[AP3H].notna().all(axis=1).rolling(HIST).sum() == HIST
    if need_truth:
        ok &= df["Ap"].notna().rolling(HORIZON).sum().shift(-HORIZON) == HORIZON
    return df.index[(df.index >= lo) & (df.index <= hi) & ok]


def split_origins(df, split):
    return valid_origins(df, *SPLITS[split], need_truth=True)


def feature_rows(df, orig):
    """(n, 600): each series' last 60 days, oldest first, series by series."""
    vals = df[COLS].to_numpy(dtype=np.float32)
    pos = df.index.get_indexer(orig)
    assert (pos >= HIST - 1).all()
    return np.stack([vals[p - HIST + 1 : p + 1].T.ravel() for p in pos])


def targets(df, orig):
    """(n, 30): daily Ap at t+1 .. t+30."""
    tgt = df["Ap"].to_numpy(dtype=np.float64)
    pos = df.index.get_indexer(orig)
    return np.stack([tgt[p + 1 : p + 1 + HORIZON] for p in pos])


def load_models():
    """The 30 text boosters (saved at their early-stopped iteration) + meta."""
    import lightgbm as lgb

    meta = json.load(open(os.path.join(MODELS, "meta.json")))
    assert meta["feature_names"] == NAMES, "feature layout changed since training"
    return [lgb.Booster(model_file=os.path.join(MODELS, f"lead_{h:02d}.txt")) for h in range(1, HORIZON + 1)], meta


def predict(models, X):
    """-> (n, 30) Ap forecasts (the conditional mean, so it sits above the typical day)."""
    return np.stack([m.predict(X) for m in models], axis=1)


def log_ratio(Y, P):
    return np.log((Y + 1.0) / (np.maximum(P, 0.0) + 1.0))


def error_record(df, models):
    """Every out-of-sample forecast error, in time order -> times (n,), resid (n, 30).

    The part before the test era was frozen at training time (out-of-fold
    over the training years, then the validation years). The test era is
    recomputed here from the frozen models for every origin whose 30 days
    have been observed, so the band keeps up with new data without retraining."""
    z = np.load(RECORD)
    orig = split_origins(df, "test")
    R = log_ratio(targets(df, orig), predict(models, feature_rows(df, orig)))
    return np.concatenate([z["times"], orig.to_numpy()]), np.vstack([z["resid"], R])


def band_quantiles(times, resid, t):
    """(len(QS), 30) log-ratio quantiles for an origin t, from the forecasts
    issued in (t - 11 years, t - 30 days] — the ones fully observed by t."""
    times = np.asarray(times, dtype="datetime64[ns]")
    t = pd.Timestamp(t)
    i0 = np.searchsorted(times, (t - pd.Timedelta(days=365.25 * WINDOW_YEARS)).to_datetime64(), side="right")
    i1 = np.searchsorted(times, (t - pd.Timedelta(days=HORIZON)).to_datetime64(), side="right")
    if i1 - i0 < MIN_RECORD:
        i0 = 0
    assert i1 - i0 >= MIN_RECORD, f"no error record before {t.date()} to build a band from"
    return np.nanpercentile(resid[i0:i1], QS, axis=0)


def apply_band(p, q):
    """Point forecast p (30,) and log-ratio quantiles q -> (len(QS), 30) Ap values, never below 0."""
    return np.maximum((np.maximum(p, 0.0) + 1.0) * np.exp(q) - 1.0, 0.0)


def long_frame(orig, P, B):
    """P (n, 30) forecasts and B (n, len(QS), 30) band values -> one row per origin and lead."""
    rows = []
    for i, t in enumerate(orig):
        tdates = pd.date_range(t + pd.Timedelta(days=1), periods=HORIZON)
        for h in range(HORIZON):
            rows.append((t.date(), h + 1, tdates[h].date(), round(float(P[i, h]), 2),
                         *(round(float(b), 2) for b in B[i, :, h])))
    return pd.DataFrame(rows, columns=["t_date", "lead", "target_date", "pred"] + [f"q{q:02d}" for q in QS])
