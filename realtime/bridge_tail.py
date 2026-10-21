#!/usr/bin/env python3
"""Patch the tail of data/daily.csv with realtime feeds (run after build_dataset.py).

  ssn   SILSO's definitive/provisional daily file lags by days to a month;
        days after its last entry are filled from SILSO EISN (estimated ISN,
        same v2 scale, updated daily through today). Flag ssn_filled = 2.
        Each download is kept in data/eisn_saved.csv for when SILSO is down.
  f107  LISIRD is normally same-day (17/20/23 UT readings). If a recent day
        is missing there, SWPC's daily-solar-indices observed value is used
        (adjusted = observed x d_AU^2). Flag f107_filled = 2.
Prints what was bridged; exits non-zero if the last 7 days are still not usable.
"""

import io
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

import e24
from e24 import common

DAILY = os.path.join(e24.ROOT, "data", "daily.csv")
EISN_SAVED = os.path.join(e24.ROOT, "data", "eisn_saved.csv")
EISN = "https://www.sidc.be/SILSO/DATA/EISN/EISN_current.csv"
SWPC = "https://services.swpc.noaa.gov/text/daily-solar-indices.txt"


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "F107-realtime/1.0"}), timeout=60) as r:
        return r.read().decode("utf-8", "replace")


def main():
    df = pd.read_csv(DAILY, index_col="date", parse_dates=True)
    today = pd.Timestamp.now("UTC").normalize().tz_localize(None)
    # extend the index to today so a same-day origin can exist once LISIRD posts
    idx = pd.date_range(df.index[0], max(df.index[-1], today), freq="D")
    df = df.reindex(idx)
    df.index.name = "date"
    for c in ("f107_flare", "f107_nread", "f107_filled", "ssn_filled", "srs_present"):
        df[c] = df[c].fillna(0).astype(int)
    ar_cols = [c for c in df.columns if c.startswith("ar_")]
    df[ar_cols] = df[ar_cols].ffill()

    # --- SSN from EISN. Every good download is merged into data/eisn_saved.csv,
    # and that copy is used when SILSO is down (21 Oct 2026: it was, and all
    # of October's SSN went missing because nothing had been kept).
    e = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    if os.path.exists(EISN_SAVED):
        e = pd.read_csv(EISN_SAVED, index_col="date", parse_dates=True)["eisn"]
    eisn_ok = True
    try:
        new = pd.read_csv(io.StringIO(get(EISN)), header=None, usecols=[0, 1, 2, 4], names=["y", "m", "d", "eisn"])
    except Exception as err:
        eisn_ok = False
        print(f"WARNING: EISN download failed ({err}); using the saved copy ({len(e)} days)")
    if eisn_ok:
        new["date"] = pd.to_datetime(new[["y", "m", "d"]].rename(columns={"y": "year", "m": "month", "d": "day"}))
        new = new.set_index("date")["eisn"].astype(float)
        new = new[new >= 0]  # -1 = no value yet
        e = pd.concat([e[~e.index.isin(new.index)], new]).sort_index()
        e.rename("eisn").to_csv(EISN_SAVED, index_label="date")
    need = df.index[df.ssn.isna() & (df.index >= today - pd.Timedelta(days=120))]
    fill = e.reindex(need).dropna()
    df.loc[fill.index, "ssn"] = fill.values
    df.loc[fill.index, "ssn_filled"] = 2
    print(f"ssn: last SILSO day {df.index[(df.ssn.notna()) & (df.ssn_filled != 2)].max().date()}; "
          f"bridged {len(fill)} days from EISN ({', '.join(str(d.date()) for d in fill.index[-5:])})")

    # --- F10.7 from SWPC if LISIRD is behind
    need = df.index[df.f107_adj.isna() & (df.index >= today - pd.Timedelta(days=40))]
    if len(need):
        rows = []
        for line in get(SWPC).splitlines():
            p = line.split()
            if len(p) >= 4 and p[0].isdigit() and len(p[0]) == 4:
                rows.append((pd.Timestamp(f"{p[0]}-{p[1]}-{p[2]}"), float(p[3])))
        s = pd.Series(dict(rows)).reindex(need).dropna()
        s = s[s > 0]
        if len(s):
            dist2 = common.sun_earth_distance_au(s.index) ** 2
            df.loc[s.index, "f107_obs"] = s.values
            df.loc[s.index, "f107_adj"] = s.values * dist2
            df.loc[s.index, "f107_obs_rob"] = s.values
            df.loc[s.index, "f107_adj_rob"] = s.values * dist2
            df.loc[s.index, "f107_filled"] = 2
        print(f"f107: bridged {len(s)} days from SWPC ({', '.join(str(d.date()) for d in s.index)})")
    else:
        print("f107: LISIRD current, nothing to bridge")

    df.to_csv(DAILY, float_format="%.4f")
    last = df.index[df.f107_adj.notna() & df.ssn.notna()].max()
    print(f"daily.csv now {df.index[0].date()} -> {df.index[-1].date()}; last usable origin {last.date()}")
    if (today - last).days > 2:
        sys.exit(f"WARNING: last usable origin is {(today - last).days} days old")
    if not eisn_ok:
        sys.exit("WARNING: EISN download failed; used the saved copy")


if __name__ == "__main__":
    main()
