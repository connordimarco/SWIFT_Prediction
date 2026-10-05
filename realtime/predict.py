#!/usr/bin/env python3
"""Issue today's E24 F10.7 forecast -> forecasts/f107_e24_<origin>.csv (+ latest.csv).

  predict.py [YYYY-MM-DD]   origin date (default: last usable day in data/daily.csv)

Output columns: origin, lead, target_date, f107 (point forecast, observed
flux, sfu), q05 q10 q25 q50 q75 q90 q95 (band), model, issued_at. One file
per origin; an existing file for the same origin is NOT overwritten unless
--force is given, so the archive is a record of what was issued when.
"""

import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

import e24


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    df = e24.load_table()
    models, meta = e24.load_models()
    usable = e24.valid_origins(df, "2000-01-01", "2099-12-31", need_truth=False)
    t = pd.Timestamp(args[0]) if args else usable[-1]
    if t not in usable:
        sys.exit(f"{t.date()} is not a usable origin (60-day F10.7/SSN window incomplete); last usable: {usable[-1].date()}")
    orig = pd.DatetimeIndex([t])
    X = e24.feature_rows(df, orig)
    env = e24.envelope(df).loc[orig].to_numpy()
    P = e24.to_obs(e24.predict_adj(models, X, env), orig)[0]
    tdates = pd.date_range(t + pd.Timedelta(days=1), periods=e24.HORIZON)
    # band: the model's earlier out-of-sample errors at this 81-day flux level
    B = P * np.exp(e24.band_quantiles(*e24.error_record(df, models), t, env[0]))
    rows = []
    for h in range(e24.HORIZON):
        q = {f"q{qq:02d}": round(float(B[i, h]), 1) for i, qq in enumerate(e24.QS)}
        rows.append(dict(origin=t.date(), lead=h + 1, target_date=tdates[h].date(), f107=round(float(P[h]), 1), **q))
    out = pd.DataFrame(rows)
    out["model"] = meta["model"]
    out["issued_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    os.makedirs(e24.FORECASTS, exist_ok=True)
    path = os.path.join(e24.FORECASTS, f"f107_e24_{t.date()}.csv")
    if os.path.exists(path) and not force:
        sys.exit(f"{path} exists (use --force to overwrite)")
    out.to_csv(path, index=False)
    out.to_csv(os.path.join(e24.FORECASTS, "latest.csv"), index=False)
    last = df.loc[t]
    print(f"origin {t.date()}: F10.7 obs {last.f107_obs:.1f}, env81 {env[0]:.1f}, SSN {last.ssn:.0f}"
          f" (ssn_filled={int(last.ssn_filled)}, srs_present={int(last.srs_present)}, far-side {'ok' if np.isfinite(last.fs_n) else 'missing'})")
    print(out[["lead", "target_date", "f107", "q10", "q90"]].iloc[[0, 2, 6, 13, 20, 29]].to_string(index=False))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
