#!/usr/bin/env python3
"""Build the frozen part of the E24 error record -> models/error_record.npz.

The band of every issued forecast is taken from the model's earlier
out-of-sample errors at a similar flux level (e24.band_quantiles). The models were
fitted to 1947-2021, so their own errors there are not out-of-sample; this
script makes honest ones:

  * training years: out-of-fold. The years are cut into 3-year blocks dealt
    round-robin into 5 folds; each fold is predicted by models refit on the
    other four (minus 90 days either side of every held-out block, so no
    input or target window overlaps it), with each lead's tree count fixed
    at the frozen model's early-stopped value. Short interleaved blocks, not
    five long ones, so every refit still sees the eras where the
    active-region (1996->) and far-side (2010->) inputs exist.
  * validation years (2022-2023): the frozen models' own forecasts.

Errors are log(observed / forecast) per lead. The test era (2024->) is not
stored: e24.error_record recomputes it from the frozen models at predict
time. One-time, ~40 min on 10 cores; the frozen models are not touched.
"""

import os
import time

import lightgbm as lgb
import numpy as np

import e24
from e24 import common

P = dict(e24.CFG44)
P["n_jobs"] = int(os.environ.get("LGBM_THREADS", 10))
FOLDS = 5
BLOCK_YEARS = 3
PURGE = e24.HIST + e24.HORIZON


def main():
    df = e24.load_table()
    models, meta = e24.load_models()
    best = meta["best_iteration"]
    Xtr, ytr, otr, names = common.build_samples(df, "train47", required=e24.REQUIRED, extra=common.FS_COLS, flux=e24.FLUX)
    assert names == e24.NAMES
    env = e24.envelope(df)
    dtr = env.loc[otr].to_numpy()
    ok = np.isfinite(dtr) & (dtr > 0)
    Xtr, ytr, otr, dtr = Xtr[ok], ytr[ok], otr[ok], dtr[ok]
    fold = np.asarray(((otr.year - otr[0].year) // BLOCK_YEARS) % FOLDS)
    day = np.asarray((otr - otr[0]).days)
    print(f"train {Xtr.shape} ({otr[0].date()}..{otr[-1].date()}); {FOLDS} folds of {BLOCK_YEARS}-year blocks, purge {PURGE} d", flush=True)

    oof = np.empty(ytr.shape)
    Ytr = e24.observed(df, otr)
    t0 = time.time()
    for k in range(FOLDS):
        te = fold == k
        held = np.zeros(day.max() + 1)
        held[day[te]] = 1
        c = np.r_[0, np.cumsum(held)]
        tr = ~te & (c[np.minimum(day + PURGE + 1, len(held))] - c[np.maximum(day - PURGE, 0)] == 0)
        for h in range(e24.HORIZON):
            m = lgb.LGBMRegressor(**dict(P, n_estimators=best[h])).fit(Xtr[tr], ytr[tr, h] / dtr[tr])
            oof[te, h] = m.predict(Xtr[te]) * dtr[te]
        err = e24.to_obs(oof[te], otr[te]) - Ytr[te]
        print(f"fold {k + 1}/{FOLDS}: {int(te.sum())} held out, {int(tr.sum())} to fit; "
              f"rmse lead 1 {np.sqrt(np.nanmean(err[:, 0] ** 2)):.2f}  lead 30 {np.sqrt(np.nanmean(err[:, -1] ** 2)):.2f} sfu  [{time.time() - t0:.0f}s]", flush=True)

    ova = e24.valid_origins(df, "2022-01-01", "2023-12-31", need_truth=True)
    Pva = e24.to_obs(e24.predict_adj(models, e24.feature_rows(df, ova), env.loc[ova].to_numpy()), ova)
    times = otr.append(ova).to_numpy()
    resid = np.log(np.vstack([Ytr, e24.observed(df, ova)]) / np.vstack([e24.to_obs(oof, otr), Pva]))
    assert (np.diff(times) > np.timedelta64(0)).all() and times[-1] < np.datetime64(e24.TEST_START)
    np.savez_compressed(e24.RECORD, times=times, resid=resid.astype(np.float32))
    print(f"saved {len(times)} forecasts ({otr[0].date()}..{ova[-1].date()}) to {os.path.relpath(e24.RECORD, e24.ROOT)}; "
          f"{int(np.isnan(resid).sum())} missing errors  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
