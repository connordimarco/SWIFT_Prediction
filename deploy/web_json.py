#!/usr/bin/env python3
"""Website views -> deploy/web/ (published by deploy/f107_daily.sh).

  forecast.json   observed daily F10.7 (last 183 days, measured days only)
                  + the newest issued forecast (point forecast and q05..q95)
                  + every percentile 1..99 of that forecast (`percentiles`),
                  for the Swift re-entry calculation's sampled paths
  hindcast.json   the forecast the frozen model gives from every origin since
                  2024-01-01 (data it was never trained or tuned on), computed
                  from today's data archive, + the band and the observed
                  series, so the page can draw any start date

The band of an issued forecast is refitted for every origin on the earlier
forecast errors at its 81-day flux level, one of six (e24.band_quantiles).
hindcast.json stores one table per month, that of the month's first origin,
as log-ratio quantiles per level (`bands[m].qNN[level][lead]`), plus each
origin's level index (`level`): the page draws
forecast * exp(bands[month].qNN[level[i]][lead]). `band` keeps the older
per-lead layout, filled with the newest origin's band, for pages that have
not switched yet.
"""
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "realtime"))
import e24  # noqa: E402

OUT = os.path.join(ROOT, "deploy", "web")
DAYS = 183  # trailing ~6 months; the page pans through it
HINDCAST_START = "2024-01-01"
BAND_QS = [5, 25, 75, 95]  # the two shaded ranges the page draws
PCTS = list(range(1, 100))  # every percentile of the newest forecast


def write(name, obj):
    tmp = os.path.join(OUT, name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    os.replace(tmp, os.path.join(OUT, name))


def observed(since):
    d = pd.read_csv(os.path.join(ROOT, "data", "daily.csv"), parse_dates=["date"])
    d = d[(d.f107_filled == 0) & d.f107_obs.notna() & (d.date >= since)]
    return {"t": d.date.dt.strftime("%Y-%m-%d").tolist(), "f107": [round(float(v), 1) for v in d.f107_obs]}


def main():
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    os.makedirs(OUT, exist_ok=True)

    df = e24.load_table()
    models, meta = e24.load_models()
    record = e24.error_record(df, models)

    # the newest issued forecast, plus every percentile of it: the issued
    # point forecast times exp(band quantile), as realtime/predict.py does
    # for q05..q95
    fc = pd.read_csv(os.path.join(ROOT, "realtime", "forecasts", "latest.csv"))
    t = pd.Timestamp(fc.origin[0])
    level_t = e24.envelope(df).loc[pd.DatetimeIndex([t])].to_numpy()[0]
    pct = fc.f107.to_numpy() * np.exp(e24.band_quantiles(*record, t, level_t, qs=PCTS))
    since = (t - pd.Timedelta(days=DAYS - 1)).strftime("%Y-%m-%d")
    write("forecast.json", {
        "generated_utc": now,
        "origin": str(fc.origin[0]),
        "issued_at": str(fc.issued_at[0]),
        "model": str(fc.model[0]),
        "observed": observed(since),
        "forecast": {"t": fc.target_date.astype(str).tolist(),
                     **{k: fc[k].round(1).tolist()
                        for k in ["f107", "q05", "q10", "q25", "q50", "q75", "q90", "q95"]}},
        # percentiles.f107[i] is the forecast at percentile levels[i], one value per day
        "percentiles": {"levels": PCTS, "f107": np.round(pct, 1).tolist()},
    })

    orig = e24.valid_origins(df, HINDCAST_START, "2099-12-31", need_truth=False)
    X = e24.feature_rows(df, orig)
    env = e24.envelope(df).loc[orig].to_numpy()
    P = e24.to_obs(e24.predict_adj(models, X, env), orig)
    qi = [e24.QS.index(q) for q in BAND_QS]
    months = orig.to_period("M")
    bands, level = [], np.empty(len(orig), dtype=int)
    for month in months.unique():
        edges, Q = e24.band_table(*record, orig[months == month][0])
        level[months == month] = np.digitize(env[months == month], edges)
        bands.append({"month": str(month),
                      **{f"q{q:02d}": np.round(Q[:, i], 4).tolist() for i, q in zip(qi, BAND_QS)}})
    newest = Q[level[-1]]
    write("hindcast.json", {
        "generated_utc": now,
        "model": meta["model"],
        "origins": [t.strftime("%Y-%m-%d") for t in orig],
        "f107": np.round(P, 1).tolist(),
        "level": level.tolist(),
        "bands": bands,
        "band": [{f"q{q:02d}": round(float(newest[i, h]), 5) for i, q in enumerate(e24.QS)} for h in range(e24.HORIZON)],
        "observed": observed((orig[0] - pd.Timedelta(days=60)).strftime("%Y-%m-%d")),
    })
    print(f"web views: forecast origin {fc.origin[0]}; hindcast {len(orig)} origins "
          f"{orig[0].date()}..{orig[-1].date()}, {len(bands)} monthly bands")


if __name__ == "__main__":
    main()
