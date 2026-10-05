#!/usr/bin/env python3
"""Percentile paths for the Swift re-entry ensemble -> deploy/web_ap/percentile_paths.json
(published with the Ap views by deploy/f107_daily.sh).

For every 3rd past forecast since 2024 (data the models never trained on),
where the observed F10.7 and Ap fell within that forecast's own band, day by
day: the percentile of the observed value (50 = right on the median). One
past forecast gives one path of 30 F10.7 and 30 Ap percentiles. A path keeps
how forecast errors carry over from one day to the next and how F10.7 and Ap
errors go together, which the per-day bands alone do not say.

The re-entry calculation applies each path to today's forecast (the
`percentiles` table in each forecast.json) and runs the orbit once per path;
the spread of re-entry dates over all paths gives true 50% and 90% ranges.
"""
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "realtime"))
sys.path.insert(0, os.path.join(ROOT, "ap"))
import e24  # noqa: E402
import shared as S  # noqa: E402

OUT = os.path.join(ROOT, "deploy", "web_ap", "percentile_paths.json")
START = "2024-01-01"
EVERY = 3  # every 3rd day: neighbouring days share 29 of their 30 days


def percentile_of(record, value):
    """Percentile (0-100) of `value` among the non-missing numbers in `record`."""
    record = record[~np.isnan(record)]
    return 100.0 * np.mean(record <= value)


def f107_path(times, resid, level, j):
    """Percentiles of forecast j's observed F10.7, against the errors its band
    was built from (same selection as e24.band_table: forecasts issued up to
    30 days before it, at its 81-day flux level)."""
    t = pd.Timestamp(times[j])
    i1 = np.searchsorted(times, (t - pd.Timedelta(days=e24.HORIZON)).to_datetime64(), side="right")
    edges = np.quantile(level[:i1], np.linspace(0, 1, e24.LEVEL_BINS + 1)[1:-1])
    same_level = np.digitize(level[:i1], edges) == np.digitize(level[j], edges)
    R = resid[:i1][same_level]
    return [percentile_of(R[:, h], resid[j, h]) for h in range(e24.HORIZON)]


def ap_path(times, resid, j):
    """Percentiles of forecast j's observed Ap, against the errors its band was
    built from (same selection as shared.band_quantiles: forecasts issued in
    the 11 years up to 30 days before it)."""
    t = pd.Timestamp(times[j])
    i0 = np.searchsorted(times, (t - pd.Timedelta(days=365.25 * S.WINDOW_YEARS)).to_datetime64(), side="right")
    i1 = np.searchsorted(times, (t - pd.Timedelta(days=S.HORIZON)).to_datetime64(), side="right")
    if i1 - i0 < S.MIN_RECORD:
        i0 = 0
    R = resid[i0:i1]
    return [percentile_of(R[:, h], resid[j, h]) for h in range(S.HORIZON)]


def main():
    df = e24.load_table()
    models, _ = e24.load_models()
    f_times, f_resid, f_level = e24.error_record(df, models)
    f_times = np.asarray(f_times, dtype="datetime64[ns]")

    df_ap = S.load_table()
    models_ap, _ = S.load_models()
    a_times, a_resid = S.error_record(df_ap, models_ap)
    a_times = np.asarray(a_times, dtype="datetime64[ns]")

    # past forecasts with all 30 days observed for both indices
    f_index = {t: j for j, t in enumerate(f_times) if not np.isnan(f_resid[j]).any()}
    a_index = {t: j for j, t in enumerate(a_times) if not np.isnan(a_resid[j]).any()}
    days = sorted(t for t in f_index if t in a_index and t >= np.datetime64(START))
    days = days[::EVERY]

    f107, ap = [], []
    for t in days:
        f107.append(f107_path(f_times, f_resid, f_level, f_index[t]))
        ap.append(ap_path(a_times, a_resid, a_index[t]))
    # today's forecast percentiles run from 1 to 99, so the paths do too
    f107 = np.clip(np.array(f107), 1, 99)
    ap = np.clip(np.array(ap), 1, 99)

    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"made_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "about": "percentile (1-99) of the observed value within each past forecast's "
                            "band, per lead day; f107[i] and ap[i] come from the forecast issued on origins[i]",
                   "origins": [str(pd.Timestamp(t).date()) for t in days],
                   "f107": np.round(f107, 1).tolist(),
                   "ap": np.round(ap, 1).tolist()}, f, separators=(",", ":"))
    os.replace(tmp, OUT)

    # checks: a calibrated band puts ~50% of days inside 25-75 and ~90% inside 5-95
    for name, P in (("F10.7", f107), ("Ap", ap)):
        print(f"{name}: inside 25-75 {np.mean((P > 25) & (P < 75)):.0%}, inside 5-95 {np.mean((P > 5) & (P < 95)):.0%}, "
              f"next-day correlation {np.corrcoef(P[:, :-1].ravel(), P[:, 1:].ravel())[0, 1]:.2f}")
    print(f"same-day F10.7/Ap correlation {np.corrcoef(f107.ravel(), ap.ravel())[0, 1]:.2f}")
    print(f"percentile paths: {len(days)} past forecasts {pd.Timestamp(days[0]).date()}..{pd.Timestamp(days[-1]).date()}")


if __name__ == "__main__":
    main()
