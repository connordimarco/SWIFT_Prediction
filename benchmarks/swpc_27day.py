#!/usr/bin/env python3
"""Benchmark: SWPC's operational 27-day F10.7 outlook vs our cells.

Source: the weekly "Preliminary Report and Forecast" PDFs (SWPC warehouse
WeeklyPDF tarballs), each carrying a "Twenty-seven Day Outlook" table of
daily 10.7 cm flux starting on the issue date (issued ~02 UT Mondays, so it
knows data through the day before). Apples to apples: our forecast from
origin t = issue_date - 1 at lead h corresponds to SWPC's value for
issue_date + h - 1 (h = 1..27).

Usage: swpc_27day.py <pdf_dir> <model_dir> [<model_dir> ...]   (e.g. data/swpc_prf model)
Prints RMSE by lead band for SWPC, persistence and each cell on the common
(origin, target) pairs, plus the parse log. Writes <pdf_dir>/swpc_27day.csv
(issue_date, target_date, f107_swpc).
"""

import glob
import os
import re
import sys

import numpy as np
import pandas as pd

try:
    import pymupdf as fitz
except ImportError:  # older wheels
    import fitz

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
import common

MONTHS = {m: i + 1 for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}


def parse_pdf(path, with_ap=False):
    """-> (issue_date, {target_date: flux}) or raises.
    with_ap=True gives {target_date: (flux, Ap)} instead."""
    doc = fitz.open(path)
    txt = "\n".join(p.get_text() for p in doc)
    m = re.search(r"SWPC PRF \d+\s+(\d{1,2}) (\w+) (\d{4})", txt)
    issue = pd.Timestamp(int(m.group(3)), MONTHS[m.group(2)[:3]], int(m.group(1)))
    i = txt.find("Twenty-seven Day Outlook")
    body = txt[i:]
    j = body.rfind("Kp Index", 0, 400)  # end of the two-column header
    lines = [l.strip() for l in body[j + len("Kp Index"):].split("\n") if l.strip()]
    window = pd.date_range(issue, periods=27)
    by_dom = {d.day: d for d in window}
    out, k = {}, 0
    while k < len(lines) and len(out) < 27:
        dm = re.fullmatch(r"(\d{1,2})(?:\s+[A-Za-z]{3})?", lines[k])
        if dm and k + 3 < len(lines) and all(re.fullmatch(r"-?\d+", lines[k + q]) for q in (1, 2, 3)):
            flux, ap = float(lines[k + 1]), float(lines[k + 2])
            out[by_dom[int(dm.group(1))]] = (flux, ap) if with_ap else flux
            k += 4
        else:
            k += 1
    if len(out) != 27:
        raise ValueError(f"parsed {len(out)} rows")
    return issue, out


def main(pdf_dir, cells):
    rows, bad = [], []
    for p in sorted(glob.glob(os.path.join(pdf_dir, "**", "*.pdf"), recursive=True)):
        try:
            issue, vals = parse_pdf(p)
            rows += [(issue, t, v) for t, v in vals.items()]
        except Exception as e:  # noqa: BLE001
            bad.append((os.path.basename(p), str(e)))
    sw = pd.DataFrame(rows, columns=["issue_date", "target_date", "f107_swpc"]).drop_duplicates(["issue_date", "target_date"])
    sw.to_csv(os.path.join(pdf_dir, "swpc_27day.csv"), index=False)
    print(f"parsed {sw.issue_date.nunique()} outlooks ({sw.issue_date.min().date()} -> {sw.issue_date.max().date()}); "
          f"{len(bad)} PDFs failed: {bad[:5]}")
    df = common.load_daily()
    orig = common.origins("test", df)
    truth = df["f107_obs"]
    sw["t_date"] = sw.issue_date - pd.Timedelta(days=1)
    sw["lead"] = (sw.target_date - sw.t_date).dt.days
    sw = sw[sw.t_date.isin(orig)].copy()
    sw["y"] = truth.reindex(sw.target_date).to_numpy()
    sw["persist"] = truth.reindex(sw.t_date).to_numpy()
    sw = sw.dropna(subset=["y"])
    print(f"common set: {sw.t_date.nunique()} outlook origins in the test period, {len(sw)} (origin, target) pairs")
    preds = {"SWPC 27-day outlook": sw.f107_swpc, "persistence": sw.persist}
    for c in cells:
        p = pd.read_csv(os.path.join(c, "out", "predictions.csv"), parse_dates=["t_date", "target_date"])
        p = p.merge(sw[["t_date", "target_date"]], on=["t_date", "target_date"], how="inner")
        assert len(p) == len(sw), (c, len(p), len(sw))
        preds[os.path.basename(c.rstrip("/"))] = p.set_index(["t_date", "target_date"]).pred_obs.reindex(
            pd.MultiIndex.from_frame(sw[["t_date", "target_date"]])).to_numpy()
    bands = {"L1-7": (1, 7), "L8-14": (8, 14), "L15-27": (15, 27), "pooled": (1, 27), "hi-act(y>=150)": None}
    res = {}
    for name, pr in preds.items():
        pr = np.asarray(pr, dtype=float)
        r = {}
        for b, rng in bands.items():
            m = (sw.y >= 150).to_numpy() if rng is None else ((sw.lead >= rng[0]) & (sw.lead <= rng[1])).to_numpy()
            r[b] = np.sqrt(np.mean((pr[m] - sw.y.to_numpy()[m]) ** 2))
        res[name] = r
    out = pd.DataFrame(res).T
    pd.set_option("display.width", 200)
    print("\nRMSE (sfu) on the common set:")
    print(out.round(2).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
