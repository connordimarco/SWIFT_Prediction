#!/usr/bin/env python3
"""Ap website views -> deploy/web_ap/ (published by deploy/f107_daily.sh).

  forecast.json   observed daily Ap (last 183 days) + the newest issued Ap
                  forecast (point forecast and q05..q95) + every percentile
                  1..99 of that forecast (`percentiles`), for the Swift
                  re-entry calculation's sampled paths + every forecast
                  issued so far (`issued`, from the archive in ap/forecasts/),
                  for the page's live record
  hindcast.json   the forecast the frozen model gives from every origin since
                  2024-01-01 (data it was never trained or tuned on), computed
                  from today's data archive, + the band and the observed
                  series, so the page can draw any start date

The band of an issued forecast is refitted for every origin on the 11 years
of errors before it (shared.band_quantiles). It moves little within a month
(at most 1.1 Ap on 2024-2026), so hindcast.json stores one band per month,
the band of that month's first origin, as log-ratio quantiles: the page
draws max((forecast + 1) * exp(q) - 1, 0), as shared.apply_band does.
"""
import datetime as dt
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "ap"))
import shared as S  # noqa: E402

OUT = os.path.join(ROOT, "deploy", "web_ap")
DAYS = 183  # trailing ~6 months; the page pans through it
HINDCAST_START = "2024-01-01"
BAND_QS = [5, 25, 75, 95]  # the two shaded ranges the page draws
PCTS = list(range(1, 100))  # every percentile of the newest forecast


def write(name, obj):
    tmp = os.path.join(OUT, name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    os.replace(tmp, os.path.join(OUT, name))


def issued():
    """Every Ap forecast issued so far (ap/forecasts/ap_<origin>.csv): origins and point forecasts."""
    out = {"origins": [], "ap": []}
    for path in sorted(glob.glob(os.path.join(S.FORECASTS, "ap_*.csv"))):
        f = pd.read_csv(path)
        out["origins"].append(str(f.origin[0]))
        out["ap"].append(f.ap.round(1).tolist())
    return out


def observed(df, since):
    d = df.loc[since:, "Ap"].dropna()
    return {"t": d.index.strftime("%Y-%m-%d").tolist(), "ap": [int(round(v)) for v in d]}


def main():
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    os.makedirs(OUT, exist_ok=True)
    df = S.load_table()
    models, meta = S.load_models()
    times, resid = S.error_record(df, models)

    # the newest issued forecast, plus every percentile of it, built from the
    # issued point forecast the way ap/predict.py builds q05..q95
    fc = pd.read_csv(os.path.join(S.FORECASTS, "latest.csv"))
    t = pd.Timestamp(fc.origin[0])
    pct = S.apply_band(fc.ap.to_numpy(), S.band_quantiles(times, resid, t, qs=PCTS))
    since = t - pd.Timedelta(days=DAYS - 1)
    write("forecast.json", {
        "generated_utc": now,
        "origin": str(fc.origin[0]),
        "issued_at": str(fc.issued_at[0]),
        "model": str(fc.model[0]),
        "observed": observed(df, since),
        "forecast": {"t": fc.target_date.astype(str).tolist(),
                     **{k: fc[k].round(1).tolist()
                        for k in ["ap", "q05", "q10", "q25", "q50", "q75", "q90", "q95"]}},
        # percentiles.ap[i] is the forecast at percentile levels[i], one value per day
        "percentiles": {"levels": PCTS, "ap": np.round(pct, 1).tolist()},
        "issued": issued(),
    })

    orig = S.valid_origins(df, HINDCAST_START, "2099-12-31", need_truth=False)
    P = S.predict(models, S.feature_rows(df, orig))
    qi = [S.QS.index(q) for q in BAND_QS]
    first = pd.Series(orig, index=orig.to_period("M")).groupby(level=0).first()
    band = []
    for month, t in first.items():
        Q = S.band_quantiles(times, resid, t)[qi]
        band.append({"month": str(month),
                     **{f"q{q:02d}": np.round(Q[i], 3).tolist() for i, q in enumerate(BAND_QS)}})
    clim = json.load(open(os.path.join(S.OUT, "scorecard.json")))["climatology"]
    write("hindcast.json", {
        "generated_utc": now,
        "model": meta["model"],
        "climatology": round(float(clim), 1),  # mean Ap over the training years
        "origins": [t.strftime("%Y-%m-%d") for t in orig],
        "ap": np.round(P, 1).tolist(),
        "band": band,
        "observed": observed(df, orig[0] - pd.Timedelta(days=60)),
    })
    print(f"ap web views: forecast origin {fc.origin[0]}; hindcast {len(orig)} origins "
          f"{orig[0].date()}..{orig[-1].date()}, {len(band)} monthly bands")


if __name__ == "__main__":
    main()
