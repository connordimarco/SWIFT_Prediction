#!/usr/bin/env python3
"""data/ap_daily.csv <- GFZ Potsdam's Kp/ap/Ap/SN/F10.7 file (1932 -> yesterday).

One row per UT day: the eight 3-hourly ap values (ap1 = 00-03 UT .. ap8 =
21-24 UT), their daily mean Ap, the international sunspot number and the
observed F10.7 (NaN before 1947). The most recent rows are GFZ's nowcast
values (kp_def = 0) and get replaced by definitive ones on a later pull.

    python scripts/build_ap.py            # download, then build
    python scripts/build_ap.py --cached   # rebuild from the last download
"""

import os
import sys
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
URL = "https://kp.gfz.de/fileadmin/files_for_gfz_cms/Kp_ap_Ap_SN_F107_since_1932.txt"
RAW = os.path.join(DATA, "Kp_ap_Ap_SN_F107_since_1932.txt")
OUT = os.path.join(DATA, "ap_daily.csv")

COLS = ("Y M D days days_m Bsr dB Kp1 Kp2 Kp3 Kp4 Kp5 Kp6 Kp7 Kp8 "
        "ap1 ap2 ap3 ap4 ap5 ap6 ap7 ap8 Ap SN F107obs F107adj Dflag").split()
AP3H = [f"ap{i}" for i in range(1, 9)]


def main():
    os.makedirs(DATA, exist_ok=True)
    if "--cached" not in sys.argv:
        tmp = RAW + ".part"
        urllib.request.urlretrieve(URL, tmp)
        os.replace(tmp, RAW)
    raw = pd.read_csv(RAW, comment="#", sep=r"\s+", names=COLS)
    raw.index = pd.to_datetime(dict(year=raw.Y, month=raw.M, day=raw.D))
    raw.index.name = "time"

    df = raw[AP3H + ["Ap"]].astype(float)
    df["ssn"] = raw.SN.astype(float)
    df["f107_obs"] = raw.F107obs.astype(float)
    df = df.mask(df < 0)  # GFZ marks missing with -1
    df["kp_def"] = raw.Dflag  # 0 = nowcast, 1 = Kp definitive, 2 = Kp and SN definitive

    assert df.index.is_monotonic_increasing and not df.index.has_duplicates
    assert len(df) == (df.index[-1] - df.index[0]).days + 1, "gap in the daily grid"
    ok = df[AP3H].notna().all(axis=1)
    assert np.allclose(df.Ap[ok], df.loc[ok, AP3H].mean(axis=1).round(), atol=0.51), "Ap is not the mean of ap1..8"

    df.to_csv(OUT, float_format="%.1f")
    print(f"{OUT}: {len(df)} days {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"  missing: Ap {int(df.Ap.isna().sum())}, ssn {int(df.ssn.isna().sum())}, "
          f"f107_obs {int(df.f107_obs.isna().sum())} (first value {df.f107_obs.first_valid_index().date()})")
    print(f"  nowcast (kp_def = 0) from {df.index[df.kp_def == 0][0].date()}")


if __name__ == "__main__":
    main()
