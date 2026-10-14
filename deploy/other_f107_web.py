#!/usr/bin/env python3
"""Two other groups' F10.7 forecasts -> deploy/web/other_f107.json (for
"Show NJIT" and "Show ADAPT-SIFT" on the website's F10.7 chart).

- NJIT (Zhenduo Wang and Yasser Abduallah): 60 days, issued once a day
  around 05 UT, with a date on every value.
- ADAPT-SIFT (NSO, AFRL): issue time + 0..7 days, a new issue every 2 hours;
  the newest issue of each UT day is kept.

Neither address keeps past issues, so every fetch is added to
data/other_f107/{njit,sift}.json and the issues build up into a record from
2026-10-14 on. A failed fetch leaves the record as it is (exit 1).
"""
import datetime as dt
import json
import os
import re
import sys
import urllib.request

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RECORD = os.path.join(ROOT, "data", "other_f107")
OUT = os.path.join(ROOT, "deploy", "web", "other_f107.json")
NJIT = "https://nature.njit.edu/solardb/api/db/f107_raw.php"
SIFT = "https://gong.nso.edu/ftp/adapt/sift/sift_gong_f107_forecast_current.json"
MON = {m: i + 1 for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}


def njit_issue():
    """-> (issue date, [60 daily F10.7 from the issue date on])."""
    txt = urllib.request.urlopen(NJIT, timeout=60).read().decode()
    head = re.search(r"^#Issued: (\d{4}) (\w{3}) (\d{2})", txt, re.M)
    if not head:
        raise ValueError("NJIT: no #Issued line")
    issued = dt.date(int(head[1]), MON[head[2]], int(head[3]))
    # values look like "14Oct26 119"
    rows = re.findall(r"(\d{2})([A-Z][a-z]{2})(\d{2}) (\d+)", txt)
    days = [dt.date(2000 + int(y), MON[m], int(d)) for d, m, y, _ in rows]
    if days != [issued + dt.timedelta(days=k) for k in range(60)]:
        raise ValueError(f"NJIT: {len(rows)} values, not 60 days from {issued}")
    return issued.isoformat(), [float(r[3]) for r in rows]


def sift_issues():
    """-> {UT date: {"issued": "YYYY-MM-DDTHH:MMZ", "f107": [8 values]}}, the
    newest issue of each date in the current file. Value k is for the issue
    time + k days; missing values are None."""
    doc = json.loads(urllib.request.urlopen(SIFT, timeout=60).read())["siftForecast_F10.7"]
    missing = doc["forecastSetInfo"]["missingValue"]
    out = {}
    for entry in doc["forecastSet"]:
        f = entry["siftForecast"]
        vals = [None if v == missing else v for v in f["forecastValues"]]
        if all(v is None for v in vals):
            continue
        issued = f["forecastDateTime"][0]
        day = issued[:10]
        if day not in out or issued > out[day]["issued"]:
            out[day] = {"issued": issued, "f107": vals}
    return out


def load(name):
    path = os.path.join(RECORD, name)
    return json.load(open(path)) if os.path.exists(path) else {}


def save(name, record):
    path = os.path.join(RECORD, name)
    with open(path + ".tmp", "w") as f:
        json.dump(record, f, indent=1, sort_keys=True)
    os.replace(path + ".tmp", path)


def main():
    os.makedirs(RECORD, exist_ok=True)
    failed = False

    njit = load("njit.json")
    try:
        day, vals = njit_issue()
        njit[day] = vals
        save("njit.json", njit)
    except Exception as e:  # noqa: BLE001 — keep the record as it is
        print(f"WARNING: NJIT F10.7 forecast failed: {e}")
        failed = True

    sift = load("sift.json")
    try:
        for day, issue in sift_issues().items():
            if day not in sift or issue["issued"] > sift[day]["issued"]:
                sift[day] = issue
        save("sift.json", sift)
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: ADAPT-SIFT F10.7 forecast failed: {e}")
        failed = True

    nd, sd = sorted(njit), sorted(sift)
    with open(OUT + ".tmp", "w") as f:
        json.dump({"generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "njit": {"source": "NJIT 60-day F10.7 forecast; values for issue date + 0..59 days",
                            "issues": nd, "f107": [njit[d] for d in nd]},
                   "sift": {"source": "ADAPT-SIFT F10.7 forecast, newest issue of each UT day; "
                                      "values for issue time + 0..7 days",
                            "issues": sd, "issued_at": [sift[d]["issued"] for d in sd],
                            "f107": [sift[d]["f107"] for d in sd]}},
                  f, separators=(",", ":"))
    os.replace(OUT + ".tmp", OUT)
    print(f"other F10.7 view: NJIT {len(nd)} issues, newest {nd[-1] if nd else '-'}; "
          f"ADAPT-SIFT {len(sd)} issues, newest {sift[sd[-1]]['issued'] if sd else '-'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
