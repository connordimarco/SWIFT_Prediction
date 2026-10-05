#!/usr/bin/env python3
"""Issue today's Ap forecast -> forecasts/ap_<origin>.csv (+ latest.csv).

  predict.py [YYYY-MM-DD]   origin date (default: last usable day in data/ap_daily.csv)

Refresh the table first: `env/bin/python scripts/build_ap.py`. The origin is
a complete UT day (all eight 3-hourly ap values in), so run after 00 UT.

Output columns: origin, lead, target_date, ap (point forecast = expected
daily Ap), q05 q10 q25 q50 q75 q90 q95 (band; q50 is the typical day, below
the point forecast because Ap is skewed), model, issued_at. One file per
origin; an existing file is NOT overwritten unless --force is given, so the
archive is a record of what was issued when.
"""

import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

import shared as S


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    df = S.load_table()
    models, meta = S.load_models()
    usable = S.valid_origins(df, "2000-01-01", "2099-12-31", need_truth=False)
    t = pd.Timestamp(args[0]) if args else usable[-1]
    if t not in usable:
        sys.exit(f"{t.date()} is not a usable origin (60-day ap window incomplete); last usable: {usable[-1].date()}")
    P = S.predict(models, S.feature_rows(df, pd.DatetimeIndex([t])))[0]
    times, resid = S.error_record(df, models)
    B = S.apply_band(P, S.band_quantiles(times, resid, t))
    tdates = pd.date_range(t + pd.Timedelta(days=1), periods=S.HORIZON)
    rows = [dict(origin=t.date(), lead=h + 1, target_date=tdates[h].date(), ap=round(float(P[h]), 1),
                 **{f"q{q:02d}": round(float(B[i, h]), 1) for i, q in enumerate(S.QS)}) for h in range(S.HORIZON)]
    out = pd.DataFrame(rows)
    out["model"] = meta["model"]
    out["issued_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    os.makedirs(S.FORECASTS, exist_ok=True)
    path = os.path.join(S.FORECASTS, f"ap_{t.date()}.csv")
    if os.path.exists(path) and not force:
        sys.exit(f"{path} exists (use --force to overwrite)")
    out.to_csv(path, index=False)
    out.to_csv(os.path.join(S.FORECASTS, "latest.csv"), index=False)
    last = df.loc[t]
    n_band = int(np.sum((times > np.datetime64(t - pd.Timedelta(days=365.25 * S.WINDOW_YEARS))) & (times <= np.datetime64(t - pd.Timedelta(days=S.HORIZON)))))
    print(f"origin {t.date()}: Ap {last.Ap:.0f}, 3-hourly ap {[int(v) for v in last[S.AP3H]]}, SSN {last.ssn:.0f}, F10.7 {last.f107_obs:.1f}"
          f" (kp_def={int(last.kp_def)}); band from {n_band} past forecasts")
    print(out[["lead", "target_date", "ap", "q05", "q50", "q95"]].iloc[[0, 1, 2, 6, 13, 26, 29]].to_string(index=False))
    print(f"wrote {os.path.relpath(path, S.ROOT)}")


if __name__ == "__main__":
    main()
