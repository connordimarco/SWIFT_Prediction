#!/usr/bin/env python3
"""Train the 30 Ap boosters and the error record behind the band (ap/models/).

1. One LightGBM per lead on origins 1932-2021, early-stopped on 2022-2023.
2. Out-of-fold forecasts over the training years: five contiguous blocks,
   each predicted by models refit on the other four (minus 90 days either
   side, so no input or target window overlaps the held-out block), with
   every lead's tree count fixed at its early-stopped value. These, plus the
   validation forecasts, are the honest errors the band is calibrated on.
3. Test-era forecasts with their bands -> out/predictions_test.csv.

About 15 minutes on 10 cores (step 2 is five more fits of everything).
"""

import datetime as dt
import json
import os
import time

import lightgbm as lgb
import numpy as np

import shared as S

P = dict(S.CFG44)
P["n_jobs"] = int(os.environ.get("LGBM_THREADS", 10))
FOLDS = 5
PURGE = S.HIST + S.HORIZON


def main():
    df = S.load_table()
    otr, ova, ote = (S.split_origins(df, s) for s in ("train", "val", "test"))
    Xtr, Xva, Xte = (S.feature_rows(df, o) for o in (otr, ova, ote))
    Ytr, Yva, Yte = (S.targets(df, o) for o in (otr, ova, ote))
    print(f"train {Xtr.shape} ({otr[0].date()}..{otr[-1].date()})  val {Xva.shape}  "
          f"test {Xte.shape} ({ote[0].date()}..{ote[-1].date()})", flush=True)

    os.makedirs(S.MODELS, exist_ok=True)
    os.makedirs(S.OUT, exist_ok=True)
    pva, pte = np.empty(Yva.shape), np.empty(Yte.shape)
    best = []
    t0 = time.time()
    for h in range(S.HORIZON):
        m = lgb.LGBMRegressor(**P)
        m.fit(Xtr, Ytr[:, h], eval_set=[(Xva, Yva[:, h])], eval_metric="rmse",
              callbacks=[lgb.early_stopping(150, verbose=False)])
        pva[:, h] = m.predict(Xva, num_iteration=m.best_iteration_)
        pte[:, h] = m.predict(Xte, num_iteration=m.best_iteration_)
        m.booster_.save_model(os.path.join(S.MODELS, f"lead_{h + 1:02d}.txt"), num_iteration=m.best_iteration_)
        best.append(int(m.best_iteration_))
        print(f"lead {h + 1:2d}: best_iter {m.best_iteration_:4d}  val_rmse {m.best_score_['valid_0']['rmse']:.3f}  [{time.time() - t0:.0f}s]", flush=True)

    n = len(otr)
    edges = np.linspace(0, n, FOLDS + 1).astype(int)
    oof = np.empty(Ytr.shape)
    for k in range(FOLDS):
        a, b = edges[k], edges[k + 1]
        tr = np.r_[0:max(0, a - PURGE), min(n, b + PURGE):n]
        for h in range(S.HORIZON):
            oof[a:b, h] = lgb.LGBMRegressor(**dict(P, n_estimators=best[h])).fit(Xtr[tr], Ytr[tr, h]).predict(Xtr[a:b])
        print(f"out-of-fold block {k + 1}/{FOLDS} ({otr[a].date()}..{otr[b - 1].date()}): "
              f"rmse lead 1 {np.sqrt(np.mean((oof[a:b, 0] - Ytr[a:b, 0]) ** 2)):.2f}  "
              f"lead 30 {np.sqrt(np.mean((oof[a:b, -1] - Ytr[a:b, -1]) ** 2)):.2f}  [{time.time() - t0:.0f}s]", flush=True)

    times = otr.append(ova).to_numpy()
    resid = S.log_ratio(np.vstack([Ytr, Yva]), np.vstack([oof, pva]))
    np.savez_compressed(S.RECORD, times=times, resid=resid.astype(np.float32))

    meta = dict(
        model="AP_3h60_ssn_f107", params={k: v for k, v in P.items() if k != "n_jobs"},
        feature_names=S.NAMES, n_features=len(S.NAMES), hist_days=S.HIST, horizon_days=S.HORIZON,
        train_origins=[str(otr[0].date()), str(otr[-1].date()), int(len(otr))],
        val_origins=[str(ova[0].date()), str(ova[-1].date()), int(len(ova))],
        best_iteration=best, climatology=round(float(df.Ap.loc[otr].mean()), 3),
        band=dict(kind="log((obs+1)/(forecast+1)) quantiles, trailing window", window_years=S.WINDOW_YEARS,
                  record_origins=[str(otr[0].date()), str(ova[-1].date()), int(len(times))], oof_folds=FOLDS, purge_days=PURGE),
        data_last_day=str(df.index[-1].date()), trained_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        lightgbm=lgb.__version__,
    )
    json.dump(meta, open(os.path.join(S.MODELS, "meta.json"), "w"), indent=1)

    all_times = np.concatenate([times, ote.to_numpy()])
    all_resid = np.vstack([resid, S.log_ratio(Yte, pte)])
    B = np.stack([S.apply_band(pte[i], S.band_quantiles(all_times, all_resid, t)) for i, t in enumerate(ote)])
    S.long_frame(ote, pte, B).to_csv(os.path.join(S.OUT, "predictions_test.csv"), index=False)
    print(f"saved 30 boosters, meta.json and the error record ({len(times)} forecasts) to ap/models; "
          f"test predictions for {len(ote)} origins to ap/out  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
