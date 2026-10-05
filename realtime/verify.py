#!/usr/bin/env python3
"""Checks that the realtime path reproduces the E24 experiment cell.

 1. out/predictions_test.csv (written by train.py) vs model/train_e24.py's
    out/predictions.csv on the canonical test origins: must match to 0.01 sfu.
 2. The standalone feature builder (what predict.py uses) vs the matrix's
    build_samples on 50 random test origins: identical rows, identical
    predictions.
 4. Text-booster models (portable) agree with the joblib ones.
 3. Band check: every test-era forecast whose 30 days have been observed,
    with the band it would have been issued with (from earlier forecasts
    only): share of days inside the 90 / 80 / 50% bands. Out-of-sample, so
    this is a real check; reported, not asserted (the years overlap heavily).
"""

import os
import sys

import numpy as np
import pandas as pd

import e24
from e24 import common

CELL = os.path.join(e24.ROOT, "model", "out", "predictions.csv")


def main():
    ok = True
    a = pd.read_csv(os.path.join(e24.OUT, "predictions_test.csv"), parse_dates=["t_date", "target_date"])
    b = pd.read_csv(CELL, parse_dates=["t_date", "target_date"])
    m = a.merge(b, on=["t_date", "lead", "target_date"], suffixes=("_rt", "_cell"))
    d = (m.pred_obs_rt - m.pred_obs_cell).abs()
    print(f"1. retrained vs cell: {len(m)} pairs (cell {len(b)}, retrained {len(a)}); max |diff| {d.max():.3f} sfu; "
          f"{(d > 0.011).sum()} pairs differ by > 0.01")
    ok &= len(m) == len(b) and d.max() <= 0.011

    df = e24.load_table()
    models, _ = e24.load_models()
    Xte, _, ote, _ = common.build_samples(df, "test", extra=common.FS_COLS, flux=e24.FLUX)
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(len(ote), 50, replace=False))
    X1 = e24.feature_rows(df, ote[pick])
    same = np.array_equal(np.nan_to_num(X1, nan=-9e9), np.nan_to_num(Xte[pick], nan=-9e9))
    env = e24.envelope(df).loc[ote[pick]].to_numpy()
    P1 = e24.to_obs(e24.predict_adj(models, X1, env), ote[pick])
    ref = a.set_index(["t_date", "lead"]).pred_obs
    P0 = np.array([[ref.loc[(t, h + 1)] for h in range(e24.HORIZON)] for t in ote[pick]])
    dd = np.abs(P1 - P0).max()
    print(f"2. standalone feature rows == build_samples rows: {same}; predict path vs batch: max |diff| {dd:.3f} sfu")
    ok &= same and dd <= 0.011

    txt, _ = e24.load_models("txt")
    Pt = e24.to_obs(e24.predict_adj(txt, X1, env), ote[pick])
    dt_ = np.abs(Pt - P1).max()
    print(f"4. text boosters vs joblib models: max |diff| {dt_:.4f} sfu")
    ok &= dt_ <= 0.011

    if os.path.exists(e24.RECORD):
        times, R, level = e24.error_record(df, models)
        te = times >= np.datetime64(e24.TEST_START)
        Q = np.stack([e24.band_quantiles(times, R, level, t, lv) for t, lv in zip(times[te], level[te])])  # (n, len(QS), 30)
        Rt, seen = R[te], np.isfinite(R[te])
        inside = lambda lo, hi: float((((Rt >= Q[:, e24.QS.index(lo)]) & (Rt <= Q[:, e24.QS.index(hi)]))[seen]).mean())
        print(f"3. band on {int(te.sum())} test-era origins, each with the band from earlier forecasts only: "
              f"90% covers {inside(5, 95):.3f}, 80% covers {inside(10, 90):.3f}, 50% covers {inside(25, 75):.3f}")
    else:
        print("3. no error_record.npz yet (run calibrate.py)")
    print("ALL OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
