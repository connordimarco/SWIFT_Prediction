#!/usr/bin/env python3
"""Figures for the Ap forecast -> out/*.png (run after train.py and score.py).

  rmse_by_lead.png          test RMSE by lead against climatology, persistence, 27-day recurrence
  forecast_vs_observed.png  the whole test era at 1, 3 and 27 days ahead, with the 90% band
  scatter.png               forecast against observed at the same three leads
  reliability.png           what was observed, on average, at each forecast level
  band_by_lead.png          share of days inside the 90 / 80 / 50% bands, and the 90% width
  band_history.png          90% band coverage year by year over the whole error record
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import shared as S

INK, MUTED, GRAY = "#0b0b0b", "#52514e", "#8a8983"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
LEADS = (1, 3, 27)


def ahead(h):
    return f"{h} day{'s' if h > 1 else ''} ahead"


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)


def save(fig, name):
    fig.savefig(os.path.join(S.OUT, name), dpi=130)
    plt.close(fig)


def rmse_by_lead(card):
    pl = card["per_lead"]
    leads = [m["lead"] for m in pl]
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    ax.plot(leads, [m["rmse_persist"] for m in pl], color=ORANGE, lw=1.5, label="persistence")
    ax.plot(leads, [m["rmse_recur"] for m in pl], color=AQUA, lw=1.5, label="27-day recurrence")
    ax.plot(leads, [m["rmse_clim"] for m in pl], color=GRAY, lw=1.5, ls="--", label="climatology")
    ax.plot(leads, [m["rmse"] for m in pl], color=BLUE, lw=2, marker="o", ms=3, label="model")
    ax.set_ylim(0, None)
    ax.set_title(f"Daily Ap: test RMSE by lead ({card['test_span'][0]} to {card['test_span'][1]})", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("lead (days)", color=MUTED, fontsize=9)
    ax.set_ylabel("RMSE (Ap)", color=MUTED, fontsize=9)
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="lower right")
    clean(ax)
    save(fig, "rmse_by_lead.png")


def forecast_vs_observed(pred, card):
    per_lead = {m["lead"]: m for m in card["per_lead"]}
    fig, axes = plt.subplots(len(LEADS), 1, figsize=(12, 2.6 * len(LEADS)), sharex=True, sharey=True, constrained_layout=True)
    for ax, h in zip(axes, LEADS):
        s = pred[pred.lead == h].sort_values("target_date")
        m = per_lead[h]
        ax.fill_between(s.target_date, s.q05, s.q95, color=BLUE, alpha=0.18, lw=0, label="90% band")
        ax.plot(s.target_date, s.y, color=INK, lw=0.7, label="observed")
        ax.plot(s.target_date, s.pred, color=BLUE, lw=1.2, label="forecast")
        ax.set_title(f"Ap, {ahead(h)}:  r {m['r']:.2f}   RMSE {m['rmse']:.1f}   (climatology {m['rmse_clim']:.1f})   "
                     f"band covers {m['cover90']:.0%}", fontsize=10, color=INK, loc="left")
        ax.set_ylabel("Ap", color=MUTED, fontsize=9)
        clean(ax)
    axes[0].legend(fontsize=8, frameon=False, ncol=3, loc="upper right")
    save(fig, "forecast_vs_observed.png")


def scatter(pred, card):
    per_lead = {m["lead"]: m for m in card["per_lead"]}
    hi = float(pred.y.max()) * 1.03
    fig, axes = plt.subplots(1, len(LEADS), figsize=(4.0 * len(LEADS), 4.6), sharex=True, sharey=True, constrained_layout=True)
    for ax, h in zip(axes, LEADS):
        s = pred[pred.lead == h]
        ax.plot([0, hi], [0, hi], color=GRAY, lw=1, ls="--")
        ax.scatter(s.y, s.pred, s=9, color=BLUE, alpha=0.35, lw=0)
        ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.set_aspect("equal")
        ax.set_title(f"{ahead(h)}  (r {per_lead[h]['r']:.2f})", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("observed Ap", color=MUTED, fontsize=9)
        clean(ax)
    axes[0].set_ylabel("forecast Ap", color=MUTED, fontsize=9)
    save(fig, "scatter.png")


def reliability(pred):
    """Days grouped by forecast level (8 equal-count bins): the mean observed
    should sit on the diagonal; the median sits below it because the forecast
    is the expected value of a skewed quantity."""
    fig, axes = plt.subplots(1, len(LEADS), figsize=(4.0 * len(LEADS), 4.4), sharex=True, sharey=True, constrained_layout=True)
    hi = 42
    for ax, h in zip(axes, LEADS):
        s = pred[pred.lead == h]
        g = s.groupby(pd.qcut(s.pred, 8, duplicates="drop"), observed=True).agg(pred=("pred", "mean"), mean=("y", "mean"), median=("y", "median"))
        ax.plot([0, hi], [0, hi], color=GRAY, lw=1, ls="--", label="perfect")
        ax.plot(g.pred, g["mean"], color=BLUE, lw=2, marker="o", ms=5, label="mean observed")
        ax.plot(g.pred, g["median"], color=ORANGE, lw=1.5, marker="s", ms=4, label="median observed")
        ax.set_xlim(0, hi); ax.set_ylim(0, hi)
        ax.set_title(ahead(h), fontsize=10, color=INK, loc="left")
        ax.set_xlabel("forecast Ap (8 equal-count bins)", color=MUTED, fontsize=9)
        clean(ax)
    axes[0].set_ylabel("observed Ap on those days", color=MUTED, fontsize=9)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    save(fig, "reliability.png")


def band_by_lead(card):
    pl = card["per_lead"]
    leads = [m["lead"] for m in pl]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    for key, nominal, color in (("cover90", 0.90, BLUE), ("cover80", 0.80, ORANGE), ("cover50", 0.50, AQUA)):
        axes[0].axhline(nominal, color=color, lw=0.8, ls="--")
        axes[0].plot(leads, [m[key] for m in pl], color=color, lw=1.8, marker="o", ms=3, label=f"{nominal:.0%} band")
    axes[0].set_ylim(0.3, 1.0)
    axes[0].set_title("Share of test days inside each band (dashed = what it should be)", fontsize=10, color=INK, loc="left")
    axes[0].legend(fontsize=8, frameon=False, ncol=3, loc="lower right")
    axes[1].plot(leads, [m["width90"] for m in pl], color=BLUE, lw=1.8, marker="o", ms=3)
    axes[1].set_ylim(0, None)
    axes[1].set_title("Mean width of the 90% band (Ap)", fontsize=10, color=INK, loc="left")
    for ax in axes:
        ax.set_xlabel("lead (days)", color=MUTED, fontsize=9)
        clean(ax)
    save(fig, "band_by_lead.png")


def band_history(df, card):
    """The band rule applied to every forecast in the error record. Quantiles
    are refreshed every 30 origins here (per origin in train/predict), which
    is close enough for a yearly mean."""
    times, R = S.error_record(df, S.load_models()[0])
    times = pd.DatetimeIndex(times)
    win = int(round(365.25 * S.WINDOW_YEARS))
    ins = np.full(R.shape, np.nan)
    for i in range(win + S.HORIZON, len(R), S.HORIZON):
        lo, hi = np.nanpercentile(R[i - S.HORIZON - win : i - S.HORIZON], [5, 95], axis=0)
        blk = R[i : i + S.HORIZON]
        ins[i : i + S.HORIZON] = (blk >= lo) & (blk <= hi)
    done = np.isfinite(ins[:, 0])
    yearly = pd.Series(ins[done].mean(axis=1), index=times[done]).groupby(lambda t: t.year).agg(["mean", "size"])
    yearly = yearly[yearly["size"] >= S.MIN_RECORD]
    test0 = pd.Timestamp(card["test_span"][0]).year
    fig, ax = plt.subplots(figsize=(11, 3.6), constrained_layout=True)
    ax.axhline(0.90, color=INK, lw=0.8, ls="--")
    ax.axvspan(test0 - 0.5, yearly.index[-1] + 0.5, color=GRAY, alpha=0.15, lw=0)
    ax.plot(yearly.index, yearly["mean"], color=BLUE, lw=1.5, marker="o", ms=3.5)
    ax.text(yearly.index[-1] + 0.4, 0.62, "test era", color=MUTED, fontsize=8, ha="right")
    ax.set_ylim(0.6, 1.0)
    ax.set_title(f"Share of days inside the 90% band, by year  (all leads; {yearly.index[0]}–{yearly.index[-1]} mean "
                 f"{ins[done].mean():.3f}, lowest year {yearly['mean'].min():.2f}, highest {yearly['mean'].max():.2f})",
                 fontsize=10, color=INK, loc="left")
    ax.set_ylabel("coverage", color=MUTED, fontsize=9)
    clean(ax)
    save(fig, "band_history.png")


def main():
    df = S.load_table()
    card = json.load(open(os.path.join(S.OUT, "scorecard.json")))
    pred = pd.read_csv(os.path.join(S.OUT, "predictions_test.csv"), parse_dates=["t_date", "target_date"])
    pred["y"] = df["Ap"].reindex(pred.target_date).to_numpy()
    rmse_by_lead(card)
    forecast_vs_observed(pred, card)
    scatter(pred, card)
    reliability(pred)
    band_by_lead(card)
    band_history(df, card)
    print("wrote ap/out/{rmse_by_lead,forecast_vs_observed,scatter,reliability,band_by_lead,band_history}.png")


if __name__ == "__main__":
    main()
