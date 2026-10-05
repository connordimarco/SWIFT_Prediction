#!/usr/bin/env python3
"""Score the test-era Ap forecasts -> out/scorecard.json.

Reads out/predictions_test.csv (written by train.py) and scores it against
observed Ap on every test origin. References: climatology (the mean Ap of
the training origins — the honest one for Ap), persistence (Ap on the origin
day) and 27-day recurrence. Band checks: the share of days inside the
q05-q95, q10-q90 and q25-q75 bands (nominal 0.90 / 0.80 / 0.50).
"""

import json
import os

import numpy as np
import pandas as pd

import shared as S


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def metrics(g):
    y, p = g.y.to_numpy(), g.pred.to_numpy()
    m = dict(
        n=int(len(y)), rmse=rmse(p, y), mae=float(np.mean(np.abs(p - y))), bias=float(np.mean(p - y)),
        r=float(np.corrcoef(y, p)[0, 1]) if len(y) > 2 else float("nan"),
        cover90=float(np.mean((y >= g.q05) & (y <= g.q95))),
        cover80=float(np.mean((y >= g.q10) & (y <= g.q90))),
        cover50=float(np.mean((y >= g.q25) & (y <= g.q75))),
        below90=float(np.mean(y < g.q05)), above90=float(np.mean(y > g.q95)),
        width90=float(np.mean(g.q95 - g.q05)),
    )
    for ref in ("clim", "persist", "recur"):
        m[f"rmse_{ref}"] = rmse(g[ref], y)
        m[f"skill_vs_{ref}"] = float(1.0 - m["rmse"] / m[f"rmse_{ref}"])
    return m


def main():
    df = S.load_table()
    orig = S.split_origins(df, "test")
    pred = pd.read_csv(os.path.join(S.OUT, "predictions_test.csv"), parse_dates=["t_date", "target_date"])
    assert len(pred) == len(orig) * S.HORIZON and pred.t_date.isin(orig).all(), "predictions are not on the test origins"
    ap = df["Ap"]
    pred["y"] = ap.reindex(pred.target_date).to_numpy()
    pred["persist"] = ap.reindex(pred.t_date).to_numpy()
    pred["recur"] = ap.reindex(pred.target_date - pd.to_timedelta(27 * np.ceil(pred.lead / 27), unit="D")).to_numpy()
    pred["clim"] = float(ap.loc[S.split_origins(df, "train")].mean())

    per_lead = [dict(lead=int(h), **metrics(g)) for h, g in pred.groupby("lead")]
    storm = pred[pred.y >= S.STORM]
    card = dict(
        model=json.load(open(os.path.join(S.MODELS, "meta.json")))["model"],
        n_origins=int(len(orig)), test_span=[str(orig[0].date()), str(orig[-1].date())],
        climatology=float(pred.clim.iloc[0]),
        pooled=metrics(pred),
        bands={"L1": metrics(pred[pred.lead == 1]), "L2_7": metrics(pred[(pred.lead >= 2) & (pred.lead <= 7)]),
               "L8_30": metrics(pred[pred.lead >= 8])},
        by_year={int(y): metrics(g) for y, g in pred.groupby(pred.target_date.dt.year)},
        storm_days=dict(rule=f"Ap >= {S.STORM:g}", **metrics(storm)),
        largest_forecast={str(h): float(pred[pred.lead == h].pred.max()) for h in (1, 2, 3, 27)},
        per_lead=per_lead,
    )
    with open(os.path.join(S.OUT, "scorecard.json"), "w") as f:
        json.dump(card, f, indent=1)

    p = card["pooled"]
    print(f"{card['model']}: {card['n_origins']} origins {card['test_span'][0]}..{card['test_span'][1]}  "
          f"pooled RMSE {p['rmse']:.2f} (climatology {p['rmse_clim']:.2f})  r {p['r']:.3f}  "
          f"band 90/80/50 covers {p['cover90']:.3f}/{p['cover80']:.3f}/{p['cover50']:.3f}")
    print(f"  {'lead':>4} {'rmse':>6} {'r':>6} {'vs_clim':>8} {'vs_pers':>8} {'vs_recur':>8} {'cov90':>6} {'cov80':>6} {'cov50':>6} {'width90':>8}")
    for m in per_lead:
        print(f"  {m['lead']:>4} {m['rmse']:6.2f} {m['r']:6.3f} {m['skill_vs_clim']:+8.3f} {m['skill_vs_persist']:+8.3f} "
              f"{m['skill_vs_recur']:+8.3f} {m['cover90']:6.3f} {m['cover80']:6.3f} {m['cover50']:6.3f} {m['width90']:8.1f}")
    s = card["storm_days"]
    print(f"  storm days ({s['rule']}): n {s['n']}  RMSE {s['rmse']:.1f} (climatology {s['rmse_clim']:.1f})  bias {s['bias']:+.1f}  cov90 {s['cover90']:.2f}")
    print("  by year: " + "  ".join(f"{y} cov90 {m['cover90']:.3f} vs_clim {m['skill_vs_clim']:+.3f}" for y, m in card["by_year"].items()))


if __name__ == "__main__":
    main()
