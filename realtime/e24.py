"""E24 as a frozen realtime model: shared pieces for train / calibrate / predict.

E24 (model/train_e24.py) is one LightGBM per lead
(1..30 days) with cfg-44 params, trained on origins 1947-2021 (val 2022-2023
for early stopping), features = 60-day windows of 26 daily series
(flare-robust adjusted F10.7, SSN, 10 SRS active-region aggregates, 14 far-side
aggregates; the AR/far-side families are optional-NaN), and target
    y(t+h) = f107_adj_rob(t+h) / env81(t),   env81 = trailing 81-day mean,
multiplied back by env81(t) and converted to observed flux at predict time.
Everything here reuses model/common.py and model/single_model.py so the live feature row is
built exactly as the training rows were.

The band is multiplicative and depends on how active the Sun is: per lead,
quantiles of log(observed / forecast) over every earlier out-of-sample
forecast issued at a similar 81-day flux level (band_quantiles; calibrate.py
builds the record). Relative errors are several times larger at solar
maximum than at minimum, so a band that ignores the level is too wide in
quiet years and too narrow in active ones.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "model"))
import common  # noqa: E402
from single_model import CFG44  # noqa: E402

MODELS = os.path.join(HERE, "models")
OUT = os.path.join(HERE, "out")
FORECASTS = os.path.join(HERE, "forecasts")
FLUX = "f107_adj_rob"
REQUIRED = ["f107_adj", "ssn"]  # windows that must be gap-free (train47 rule)
COLS = [FLUX if c == "f107_adj" else c for c in common.FEATURE_COLS] + list(common.FS_COLS)
NAMES = [f"{c}_lag{lag}" for c in COLS for lag in range(common.HIST - 1, -1, -1)]
HIST, HORIZON = common.HIST, common.HORIZON
QS = [5, 10, 25, 50, 75, 90, 95]
RECORD = os.path.join(MODELS, "error_record.npz")
TEST_START = "2024-01-01"  # first origin the models never saw in training or early stopping
LEVEL_BINS = 6     # the record is split into sixths by 81-day flux level
MIN_RECORD = 200   # fewest past forecasts a band may be built from


def load_table():
    return common.load_daily()


def envelope(df):
    return common.envelope(df, FLUX)


def valid_origins(df, lo, hi, need_truth):
    """Origins in [lo, hi] whose 60-day REQUIRED windows are gap-free (and,
    if need_truth, whose 30-day target window is complete)."""
    x_ok = df[REQUIRED].notna().all(axis=1).rolling(HIST).sum() == HIST
    ok = x_ok & envelope(df).notna()
    if need_truth:
        ok &= df["f107_adj"].notna().rolling(HORIZON).sum().shift(-HORIZON) == HORIZON
    return df.index[(df.index >= lo) & (df.index <= hi) & ok]


def feature_rows(df, orig):
    vals = df[COLS].to_numpy()
    pos = df.index.get_indexer(orig)
    assert (pos >= HIST - 1).all()
    return np.stack([vals[p - HIST + 1 : p + 1].T.ravel() for p in pos])


def targets(df, orig):
    tgt = df[FLUX].to_numpy()
    pos = df.index.get_indexer(orig)
    return np.stack([tgt[p + 1 : p + 1 + HORIZON] for p in pos])


def load_models(fmt=None):
    """Models in two interchangeable formats: lead_NN.joblib (the sklearn
    wrapper, needs the training-time lightgbm/sklearn/python) or lead_NN.txt
    (plain LightGBM text boosters saved at best_iteration, loadable by any
    lightgbm). fmt = "joblib" | "txt" | None (env E24_MODEL_FORMAT, else
    joblib with txt fallback)."""
    import lightgbm as lgb

    meta = json.load(open(os.path.join(MODELS, "meta.json")))
    assert meta["feature_names"] == NAMES, "feature layout changed since training"
    fmt = fmt or os.environ.get("E24_MODEL_FORMAT")
    if fmt in (None, "joblib"):
        try:
            import joblib

            return [joblib.load(os.path.join(MODELS, f"lead_{h:02d}.joblib")) for h in range(1, HORIZON + 1)], meta
        except Exception as e:  # version mismatch etc.
            if fmt == "joblib":
                raise
            print(f"joblib models unusable ({type(e).__name__}); using text boosters", file=sys.stderr)
    return [lgb.Booster(model_file=os.path.join(MODELS, f"lead_{h:02d}.txt")) for h in range(1, HORIZON + 1)], meta


def predict_adj(models, X, env):
    """-> (n, 30) adjusted-flux predictions for feature rows X with envelope env (n,)."""
    P = np.stack([m.predict(X, num_iteration=getattr(m, "best_iteration_", None)) for m in models], axis=1)
    return P * np.asarray(env)[:, None]


def to_obs(P_adj, orig):
    """Adjusted -> observed per target date; returns (n, 30)."""
    out = np.empty_like(P_adj)
    for i, t in enumerate(orig):
        tdates = pd.date_range(t + pd.Timedelta(days=1), periods=HORIZON)
        out[i] = common.adj_to_obs(P_adj[i], tdates)
    return out


def observed(df, orig):
    """(n, 30) observed flux at t+1..t+30 (NaN where not yet measured)."""
    obs = df["f107_obs"].to_numpy()
    pos = df.index.get_indexer(orig)
    return np.stack([obs[p + 1 : p + 1 + HORIZON] for p in pos])


def error_record(df, models):
    """Every out-of-sample forecast error, in time order -> times (n,),
    resid (n, 30) = log(observed / forecast), level (n,) = the 81-day mean
    flux at each origin.

    The part before TEST_START was frozen by calibrate.py (out-of-fold over
    the training years, then the validation years). The test era is
    recomputed here from the frozen models for every origin whose 30 days
    have been observed, so the band keeps up with new data without retraining."""
    z = np.load(RECORD)
    orig = valid_origins(df, TEST_START, "2099-12-31", need_truth=True)
    P = to_obs(predict_adj(models, feature_rows(df, orig), envelope(df).loc[orig].to_numpy()), orig)
    times = np.concatenate([z["times"], orig.to_numpy()])
    resid = np.vstack([z["resid"], np.log(observed(df, orig) / P)])
    return times, resid, envelope(df).reindex(pd.DatetimeIndex(times)).to_numpy()


def band_table(times, resid, level, t, qs=QS):
    """Band quantiles for an origin t at every flux level -> edges (LEVEL_BINS - 1,),
    Q (LEVEL_BINS, len(qs), 30). Built from every forecast issued up to
    t - 30 days (the ones fully observed by t), split at the quantiles of
    their 81-day flux level; bin b holds levels in [edges[b-1], edges[b])."""
    times = np.asarray(times, dtype="datetime64[ns]")
    i1 = np.searchsorted(times, (pd.Timestamp(t) - pd.Timedelta(days=HORIZON)).to_datetime64(), side="right")
    assert i1 >= MIN_RECORD * LEVEL_BINS, f"no error record before {pd.Timestamp(t).date()} to build a band from"
    edges = np.quantile(level[:i1], np.linspace(0, 1, LEVEL_BINS + 1)[1:-1])
    bins = np.digitize(level[:i1], edges)
    return edges, np.stack([np.nanpercentile(resid[:i1][bins == b], qs, axis=0) for b in range(LEVEL_BINS)])


def band_quantiles(times, resid, level, t, level_t, qs=QS):
    """(len(qs), 30) log-ratio quantiles for an origin t whose 81-day flux
    level is level_t. Multiply the forecast by exp() of these for the band."""
    edges, Q = band_table(times, resid, level, t, qs)
    return Q[np.digitize(level_t, edges)]


def long_frame(orig, P_obs, col="pred_obs"):
    rows = []
    for i, t in enumerate(orig):
        tdates = pd.date_range(t + pd.Timedelta(days=1), periods=HORIZON)
        for h in range(HORIZON):
            rows.append((t.date(), h + 1, tdates[h].date(), round(float(P_obs[i, h]), 2)))
    return pd.DataFrame(rows, columns=["t_date", "lead", "target_date", col])
