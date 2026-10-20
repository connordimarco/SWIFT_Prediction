#!/usr/bin/env python3
"""Other groups' F10.7 and Ap forecasts -> deploy/web/other_f107.json, for the
website (F10.7 chart, re-entry date grid, forecast performance) and for the
Swift re-entry runs.

  njit      NJIT 60-day F10.7, issued daily ~05 UT
  sift      ADAPT-SIFT 0-7 day F10.7, every 2 hours (newest issue of each UT day kept;
            also from the CCMC challenge page's daily copy)
  swpc45    SWPC 45-day F10.7 and Ap, daily
  swpc27    SWPC 27-day outlook F10.7 and Ap, weekly (from swpc.json)
  swpc3     SWPC 3-day: F10.7 from the daily report (RSGA, 22 UT), 3-hour Kp
            from the 3-day forecast (00:30 UT issue)
  cu        CU Boulder (SWORD): 7-day F10.7 from B10.7, 3-day 3-hour ap
  esa       ESA SOLMAG short-term F10.7 and Ap (nominal)
  schatten  Schatten monthly F10.7 and Ap (nominal timing)
  esalong   ESA SOLMAG long-term, monthly F10.7 and Ap

CU, ESA and Schatten come from the CCMC Swift challenge page. Most addresses
keep no past issues, so every file fetched is saved under data/other_f107/raw/
and the record builds up from the first fetch. --backfill also fetches the
ADAPT-SIFT archive from 5 Oct 2026, the first Swift re-entry run.

Every source is written in one form: "origins", the last day of data each issue
could use, and per issue the daily "f107" and/or "ap" for origin + 1, + 2, ...
(the layout of our own forecasts), plus "ap3", the 3-hour ap from origin + 1
00 UT, where the source has it. Missing values are null. "available" is the
day each issue could first be had (CU's: the day CCMC posted it), so a re-entry
run or a score for a given day uses only what was out by then. A failed fetch
leaves the source as it was (exit 1).

    python deploy/other_f107_web.py [--backfill] [--out FILE]
"""
import datetime as dt
import ftplib
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RECORD = os.path.join(ROOT, "data", "other_f107")
RAW = os.path.join(RECORD, "raw")
SWPC_JSON = os.path.join(ROOT, "deploy", "web", "swpc.json")
OUT = os.path.join(ROOT, "deploy", "web", "other_f107.json")
NJIT = "https://nature.njit.edu/solardb/api/db/f107_raw.php"
SIFT = "https://gong.nso.edu/ftp/adapt/sift/sift_gong_f107_forecast_current.json"
SIFT_ARCHIVE = "https://gong.nso.edu/ftp/adapt/sift/archive/{ym}/sift_f107_forecast_{ymd}.json"
SWPC45 = "https://services.swpc.noaa.gov/text/45-day-forecast.txt"
SWPC_FTP = "ftp.swpc.noaa.gov"
CCMC = "https://ccmc.gsfc.nasa.gov/swift-challenge/index.php"
MON = {m: i + 1 for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}
MAX_DAYS = 60  # values kept per issue
BACKFILL_FROM = dt.date(2026, 10, 5)  # the first Swift re-entry run
KEEP_FROM = BACKFILL_FROM - dt.timedelta(days=1)  # issues from this origin on (the first run used them)
# Kp in thirds (0, 0+, 1-, 1, ...) -> ap
KP_TO_AP = [0, 2, 3, 4, 5, 6, 7, 9, 12, 15, 18, 22, 27, 32, 39, 48, 56, 67, 80, 94,
            111, 132, 154, 179, 207, 236, 300, 400]
NAMES = {
    "njit": "NJIT 60-day F10.7 forecast",
    "sift": "ADAPT-SIFT 7-day F10.7 forecast",
    "swpc45": "SWPC 45-day F10.7 and Ap forecast",
    "swpc27": "SWPC 27-day outlook",
    "swpc3": "SWPC 3-day F10.7 (daily report) and Kp (3-day forecast)",
    "cu": "CU Boulder 7-day F10.7 (from B10.7) and 3-day 3-hour ap",
    "esa": "ESA SOLMAG short-term F10.7 and Ap (nominal)",
    "schatten": "Schatten monthly F10.7 and Ap (nominal timing)",
    "esalong": "ESA SOLMAG long-term monthly F10.7 and Ap",
}


def get(url):
    return urllib.request.urlopen(url, timeout=60).read()


def day(s):
    return dt.date.fromisoformat(s)


def raw_dir(source):
    path = os.path.join(RAW, source)
    os.makedirs(path, exist_ok=True)
    return path


def save_raw(source, name, data):
    path = os.path.join(raw_dir(source), name)
    with open(path + ".tmp", "wb") as f:
        f.write(data)
    os.replace(path + ".tmp", path)


def have_raw(source, name):
    return os.path.exists(os.path.join(raw_dir(source), name))


def raw_files(source):
    """-> [(file name, text)] of a source's saved files, in name order."""
    folder = raw_dir(source)
    return [(n, open(os.path.join(folder, n)).read()) for n in sorted(os.listdir(folder))
            if not n.endswith(".tmp")]


def load(name):
    path = os.path.join(RECORD, name)
    return json.load(open(path)) if os.path.exists(path) else {}


def save(name, record):
    path = os.path.join(RECORD, name)
    with open(path + ".tmp", "w") as f:
        json.dump(record, f, indent=1, sort_keys=True)
    os.replace(path + ".tmp", path)


def issued_line(txt):
    """':Issued: 2026 Oct 19 2200 UTC' -> datetime"""
    m = re.search(r"^:Issued: (\d{4}) (\w{3}) +(\d+) (\d{2})(\d{2}) UTC", txt, re.M)
    if not m:
        raise ValueError("no :Issued: line")
    return dt.datetime(int(m[1]), MON[m[2]], int(m[3]), int(m[4]), int(m[5]))


def month_day(text, year):
    """'Oct 20' (or '20 Oct') in a file issued in `year` -> date"""
    m = re.match(r"(\w{3}) (\d+)|(\d+) (\w{3})", text.strip())
    mon, d = (m[1], m[2]) if m[1] else (m[4], m[3])
    return dt.date(year, MON[mon], int(d))


# ── fetching: each saves new files under raw/ and never deletes ──

def fetch_njit(record):
    """NJIT's issue -> record {issue date: [60 daily values from the issue date]}"""
    txt = get(NJIT).decode()
    head = re.search(r"^#Issued: (\d{4}) (\w{3}) (\d{2})", txt, re.M)
    if not head:
        raise ValueError("NJIT: no #Issued line")
    issued = dt.date(int(head[1]), MON[head[2]], int(head[3]))
    # values look like "14Oct26 119"
    rows = re.findall(r"(\d{2})([A-Z][a-z]{2})(\d{2}) (\d+)", txt)
    days = [dt.date(2000 + int(y), MON[m], int(d)) for d, m, y, _ in rows]
    if days != [issued + dt.timedelta(days=k) for k in range(60)]:
        raise ValueError(f"NJIT: {len(rows)} values, not 60 days from {issued}")
    record[issued.isoformat()] = [float(r[3]) for r in rows]


def sift_issues(doc):
    """One SIFT json -> {UT date: {"issued": "YYYY-MM-DDTHH:MMZ", "f107": [8 values]}},
    the newest issue of each date. Value k is for the issue time + k days."""
    doc = doc["siftForecast_F10.7"]
    missing = doc["forecastSetInfo"]["missingValue"]
    out = {}
    for entry in doc["forecastSet"]:
        f = entry["siftForecast"]
        vals = [None if v == missing else v for v in f["forecastValues"]]
        if all(v is None for v in vals):
            continue
        issued = f["forecastDateTime"][0]
        d = issued[:10]
        if d not in out or issued > out[d]["issued"]:
            out[d] = {"issued": issued, "f107": vals}
    return out


def sift_text_issues(txt):
    """CCMC's daily copy of the ADAPT-SIFT text file -> the same form as sift_issues.
    Columns: JD, M (0 = forecast there), Q, H, UT, LastMag, LastObs, Diff, 0d ... 7d."""
    out = {}
    for jd, values in re.findall(r"^ +(\d{7}\.\d{4}) +0 +\d +\d +\d{4}((?: +-?[\d.]+){11})", txt, re.M):
        t = dt.datetime(2000, 1, 1, 12) + dt.timedelta(days=float(jd) - 2451545.0, seconds=30)
        issued = t.strftime("%Y-%m-%dT%H:%MZ")
        d = issued[:10]
        if d not in out or issued > out[d]["issued"]:
            out[d] = {"issued": issued, "f107": [float(v) for v in values.split()][3:]}
    return out


def fetch_sift(record):
    for d, issue in sift_issues(json.loads(get(SIFT))).items():
        if d not in record or issue["issued"] > record[d]["issued"]:
            record[d] = issue


def fetch_swpc45():
    txt = get(SWPC45).decode()
    save_raw("swpc45", issued_line(txt).strftime("%Y%m%d") + ".txt", txt.encode())


def ftp_get_new(folder, pick):
    """Over one FTP connection (a new one per file is slow), download the files
    in `folder` that pick(name) names as (source, saved name) and that are not saved yet."""
    with ftplib.FTP(SWPC_FTP, timeout=60) as ftp:
        ftp.login()
        ftp.cwd(folder)
        for name in ftp.nlst():
            target = pick(name)
            if target and not have_raw(*target):
                chunks = []
                ftp.retrbinary("RETR " + name, chunks.append)
                save_raw(*target, b"".join(chunks))


def fetch_swpc3():
    """The daily reports (RSGA) and the 00:30 UT 3-day forecasts SWPC's FTP still has (75 days)."""
    today = dt.date.today()

    def year(mm, dd):  # file names have no year: the newest MMDD is this year, a later one last year
        return today.year if (int(mm), int(dd)) <= (today.month, today.day) else today.year - 1

    def wanted(mm, dd):
        return dt.date(year(mm, dd), int(mm), int(dd)) >= KEEP_FROM

    def rsga(name):
        m = re.fullmatch(r"(\d{2})(\d{2})RSGA\.txt", name)
        return ("rsga", f"{year(m[1], m[2])}{name}") if m and wanted(m[1], m[2]) else None

    def three_day(name):
        m = re.fullmatch(r"(\d{2})(\d{2})0030three_day_forecast\.txt", name)
        return ("three_day", f"{year(m[1], m[2])}{m[1]}{m[2]}.txt") if m and wanted(m[1], m[2]) else None

    ftp_get_new("/pub/forecasts/RSGA/", rsga)
    ftp_get_new("/pub/forecasts/three_day/", three_day)


def page_dates(page):
    """The CCMC page's 'Updated' date of every file it lists -> {file name: 'YYYY-MM-DD'}."""
    text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", page)))
    return {m[1]: m[2] for m in re.finditer(r"(\S+\.\w+) \w+ · [\d.]+ \w+ · Updated (\d{4}-\d{2}-\d{2})", text)}


def fetch_ccmc():
    """CU's, ESA's and the Schatten files from the CCMC challenge page (all it lists).
    The day CCMC posted each CU file is kept in cu_posted.json."""
    page = get(CCMC).decode(errors="replace")
    posted = load("cu_posted.json")
    for name, d in page_dates(page).items():
        if name.startswith(("CU_SWORD_", "SWORD_ap_forecast_")):
            posted.setdefault(name, d)
    save("cu_posted.json", posted)
    links = set(re.findall(r"swift_action=download&(?:amp;)?area=(drivers|submissions)&(?:amp;)?path=([^\"'&\s]+)",
                           page))
    for area, path in sorted(links):
        path = urllib.parse.unquote(path)
        parts = path.split("/")
        if area == "submissions" and parts[0] == "SWORD" and (
                re.fullmatch(r"CU_SWORD_\d{8}_F10_B10_7day_predict\.txt", parts[-1])
                or re.fullmatch(r"SWORD_ap_forecast_\d{8}T\d{4}\.csv", parts[-1])):
            source, name = "cu", parts[-1]
        elif area == "drivers" and parts[0] == "ESA_SOLMAG" and parts[-1] in ("short_term.dat", "long_term.dat"):
            source = "esa" if parts[-1] == "short_term.dat" else "esalong"
            name = parts[1] + ".dat"
        elif area == "drivers" and parts[0] == "Schatten" and parts[-1].endswith(".txt"):
            # no date inside the file or its path: name it by the date the page says it was posted
            posted_on = page_dates(page).get(parts[-1])
            if not posted_on:
                continue
            source, name = "schatten", posted_on.replace("-", "") + ".txt"
        elif area == "drivers" and parts[0] == "ADAPT_SIFT" and parts[-1] == "sift_gong_f107_forecast_current.txt":
            source, name = "sift_ccmc", parts[1] + ".txt"
        elif area == "drivers" and parts[0] == "SWPC_45_Day" and parts[-1].endswith(".txt"):
            # renamed by its issue date once read
            source, name = "swpc45", None
        else:
            continue
        if name and have_raw(source, name):
            continue
        data = get(CCMC + "?swift_action=download&area=" + area + "&path=" + urllib.parse.quote(path))
        if source == "swpc45":
            name = issued_line(data.decode()).strftime("%Y%m%d") + ".txt"
            if have_raw(source, name):
                continue
        save_raw(source, name, data)


def backfill():
    """ADAPT-SIFT's daily archive files from BACKFILL_FROM on, one every few
    seconds; stops at the first refusal (GONG blocks fast downloaders).
    SWPC's FTP folders already reach back 75 days."""
    d = BACKFILL_FROM
    while d < dt.date.today():
        name = d.strftime("%Y%m%d") + ".json"
        if not have_raw("sift_archive", name):
            try:
                save_raw("sift_archive", name, get(SIFT_ARCHIVE.format(ym=d.strftime("%Y%m"), ymd=d.strftime("%Y%m%d"))))
            except urllib.error.HTTPError as e:
                print(f"  ADAPT-SIFT archive refused {d} ({e.code}); stopping")
                return
            time.sleep(5)
        d += dt.timedelta(days=1)


# ── reading the saved files: each source -> {origin date: issue} ──
# an issue is {"issued": str, "f107": {date: v}, "ap": {date: v}, "ap3": {date: [8 values]}}

def read_njit(record):
    return {day(i) - dt.timedelta(days=1): {"issued": i, "f107": {day(i) + dt.timedelta(days=k): v
                                                                  for k, v in enumerate(vals)}}
            for i, vals in record.items()}


def read_sift(record):
    """The live record, with CCMC's daily copies and the 2026 archive where the
    record has no issue. The newest issue of UT day D (about 22 UT) has data
    through D: origin D, and its value k is for D + k."""
    issues = {}
    for _, txt in raw_files("sift_archive"):
        try:
            issues.update(sift_issues(json.loads(txt)))
        except (ValueError, KeyError):
            continue
    for _, txt in raw_files("sift_ccmc"):
        for d, issue in sift_text_issues(txt).items():
            if d not in issues or issue["issued"] > issues[d]["issued"]:
                issues[d] = issue
    for d, issue in record.items():
        if d not in issues or issue["issued"] >= issues[d]["issued"]:
            issues[d] = issue
    return {day(d): {"issued": s["issued"], "f107": {day(d) + dt.timedelta(days=k): v
                                                       for k, v in enumerate(s["f107"]) if k >= 1}}
            for d, s in issues.items()}


def read_swpc45():
    out = {}
    for name, txt in raw_files("swpc45"):
        if name.endswith(".json"):  # filled in from the weather files of the re-entry runs
            doc = json.loads(txt)
            out[day(doc["origin"])] = {"issued": doc["issued"],
                                       "f107": {day(t): v for t, v in doc["f107"].items()},
                                       "ap": {day(t): v for t, v in doc["ap"].items()}}
            continue
        issued = issued_line(txt)
        part = {}
        for block, key in (("45-DAY AP FORECAST", "ap"), ("45-DAY F10.7 CM FLUX FORECAST", "f107")):
            body = txt.split(block, 1)[1]
            body = re.split(r"\n[A-Z0-9-]+ [A-Z]", body, 1)[0]  # up to the next block heading
            part[key] = {dt.date(2000 + int(y), MON[m], int(d)): float(v)
                         for d, m, y, v in re.findall(r"(\d{2})([A-Z][a-z]{2})(\d{2}) (\d+)", body)}
        first = min(part["f107"])
        out[first - dt.timedelta(days=1)] = {"issued": issued.strftime("%Y-%m-%dT%H:%MZ"), **part}
    return out


def read_swpc27():
    """swpc.json: weekly issues I (data through I - 1), 27 values from I."""
    if not os.path.exists(SWPC_JSON):
        return {}
    s = json.load(open(SWPC_JSON))
    out = {}
    for i, issued in enumerate(s["issues"]):
        first = day(issued)
        out[first - dt.timedelta(days=1)] = {
            "issued": issued,
            "f107": {first + dt.timedelta(days=k): v for k, v in enumerate(s["f107"][i])},
            "ap": {first + dt.timedelta(days=k): v for k, v in enumerate(s["ap"][i])}}
    return out


def kp_value(text):
    """'4.67' or '4.67 (G1)' -> ap"""
    return KP_TO_AP[round(float(text.split()[0]) * 3)]


def read_swpc3():
    """RSGA issued on D at 22 UT: F10.7 for D + 1..3 (origin D). 3-day forecast
    issued on D + 1 at 00:30 UT: Kp for every 3 hours of D + 1..3 (origin D)."""
    out = {}
    for _, txt in raw_files("rsga"):
        try:
            issued = issued_line(txt)
            m = re.search(r"Predicted\s+(\d+ \w{3})-(\d+ \w{3})\s+(\d+)/(\d+)/(\d+)", txt.split("10.7 cm Flux")[1])
            first = month_day(m[1], issued.year)
            if first < issued.date():  # 'Predicted 01 Jan-03 Jan' in a report of 31 Dec
                first = month_day(m[1], issued.year + 1)
        except (ValueError, IndexError, TypeError):
            continue
        issue = out.setdefault(issued.date(), {"issued": issued.strftime("%Y-%m-%dT%H:%MZ")})
        issue["f107"] = {first + dt.timedelta(days=k): float(m[3 + k]) for k in range(3)}
    for _, txt in raw_files("three_day"):
        try:
            issued = issued_line(txt)
            head = re.search(r"NOAA Kp index breakdown (\w{3} \d+)-(\w{3} \d+) (\d{4})", txt)
            first = month_day(head[1], issued.year)
            rows = re.findall(r"^(\d{2})-(\d{2})UT\s+(\S+(?: \(G\d\))?)\s+(\S+(?: \(G\d\))?)\s+(\S+(?: \(G\d\))?)",
                              txt, re.M)
            if len(rows) != 8:
                continue
            ap3 = {first + dt.timedelta(days=k): [kp_value(r[2 + k]) for r in rows] for k in range(3)}
        except (ValueError, IndexError, TypeError, KeyError):
            continue
        issue = out.setdefault(first - dt.timedelta(days=1), {})
        # an issue counts as made when its last part was
        issue["issued"] = max(issue.get("issued", ""), issued.strftime("%Y-%m-%dT%H:%MZ"))
        issue["ap3"] = ap3
    return out


def read_cu():
    """B10.7 file 'Issue date: D' -> F10.7 for D + 1.. (origin D). ap files: the
    newest issue made during UT day D (origin D), 3-hour bins from D + 1 00 UT."""
    posted = load("cu_posted.json")
    out, ap_issues = {}, {}
    for name, txt in raw_files("cu"):
        if "B10" in name:
            m = re.search(r"Issue date: (\d{4}-\d{2}-\d{2})", txt)
            if not m:
                continue
            rows = re.findall(r"^(\d{4}-\d{2}-\d{2}), *[\d.]+, *([\d.]+)", txt, re.M)
            issue = out.setdefault(day(m[1]), {"issued": m[1]})
            issue["f107"] = {day(d): float(v) for d, v in rows}
            issue["available"] = max(issue.get("available", ""), posted.get(name, m[1]))
        else:
            rows = re.findall(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}),(\d{4}-\d{2}-\d{2} \d{2}):\d{2},[^,]+,[^,]+,([\d.]+)",
                              txt, re.M)
            if not rows:
                continue
            issued = dt.datetime.strptime(rows[0][0], "%Y-%m-%d %H:%M")
            if issued.date() not in ap_issues or issued > ap_issues[issued.date()][0]:
                ap_issues[issued.date()] = (issued, rows, posted.get(name, rows[0][0][:10]))
    for d, (issued, rows, posted_on) in ap_issues.items():
        ap3 = {}
        for _, start, value in rows:
            t = dt.datetime.strptime(start, "%Y-%m-%d %H")
            if t.date() > d:
                ap3.setdefault(t.date(), [None] * 8)[t.hour // 3] = float(value)
        issue = out.setdefault(d, {})
        # an issue counts as made when its last part was
        issue["issued"] = max(issue.get("issued", ""), issued.strftime("%Y-%m-%dT%H:%MZ"))
        issue["available"] = max(issue.get("available", ""), posted_on)
        issue["ap3"] = ap3
    return out


def read_esa():
    """short_term.dat: observed days, then the 27-day+ forecast (nominal, +95%, -95%),
    whose rows have no Kp ('0-0-0-...'). Origin = the day before the first forecast day."""
    out = {}
    for name, txt in raw_files("esa"):
        block = txt.split("27d forecast -> nominal", 1)
        if len(block) < 2:
            continue
        rows = re.findall(r"^(\d{2})/(\d{2})/(\d{4}) +(\d+) +\d+ +\d+ +(\d+) +0-0-0-0-0-0-0-0-",
                          block[1].split("#", 1)[0], re.M)
        if not rows:
            continue
        days = [dt.date(int(y), int(m), int(d)) for d, m, y, _, _ in rows]
        update = re.search(r"last update: (\d{2}-\w{3}-\d{4})", txt)
        if days[0] - dt.timedelta(days=1) in out:
            continue  # a later copy of an issue already read (copies are read in date order)
        out[days[0] - dt.timedelta(days=1)] = {
            "available": f"{name[:4]}-{name[4:6]}-{name[6:8]}",
            "issued": dt.datetime.strptime(update[1].title(), "%d-%b-%Y").date().isoformat() if update else "",
            "f107": {t: float(r[3]) for t, r in zip(days, rows)},
            "ap": {t: float(r[4]) for t, r in zip(days, rows)}}
    return out


def monthly_issues(versions, today):
    """Monthly values -> one issue a day: each version [(first day, {(year, month): (f107, ap)}, issued)]
    holds from its first day until the next version; the issue made on day d
    (origin d - 1) has the month's value on each of the next MAX_DAYS days."""
    out = {}
    versions = sorted(versions, key=lambda v: v[0])
    for i, (first, months, issued) in enumerate(versions):
        last = versions[i + 1][0] if i + 1 < len(versions) else today + dt.timedelta(days=1)
        d = first
        while d < last:
            # the copy of day d could be had from d (and the version itself from `first`)
            issue = {"issued": issued, "available": d.isoformat(), "f107": {}, "ap": {}}
            for k in range(MAX_DAYS):
                t = d + dt.timedelta(days=k)
                f107, ap = months.get((t.year, t.month), (None, None))
                if f107 is not None:
                    issue["f107"][t] = f107
                if ap is not None:
                    issue["ap"][t] = ap
            out[d - dt.timedelta(days=1)] = issue
            d += dt.timedelta(days=1)
    return out


def read_schatten(today):
    """Columns: month, year, then nominal timing mean, +2 sigma, -2 sigma F10.7, ap, ..."""
    versions = []
    for name, txt in raw_files("schatten"):
        months = {(int(y), int(m)): (float(f), float(a))
                  for m, y, f, a in re.findall(r"^ *(\d+) +(\d{4}) +([\d.]+) +[\d.]+ +[\d.]+ +([\d.]+)", txt, re.M)}
        if months:
            first = dt.datetime.strptime(name[:8], "%Y%m%d").date()
            versions.append((first, months, first.isoformat()))
    return monthly_issues(versions, today)


def read_esalong(today):
    """Columns: month, year, sunspot obs, pre, F10.7 obs, pre, err, AA obs, pre, Ap obs, pre, err.
    Each version (its 'Last Update') holds from the first CCMC copy of it."""
    versions = {}
    for name, txt in raw_files("esalong"):
        update = re.search(r"Last Update: (\d{4}) (\w{3}) +(\d+)", txt)
        if not update or update[0] in [v[0] for v in versions.values()]:
            continue
        months = {}
        for row in re.findall(r"^ *(\d+) +(\d{4})((?: +-?[\d.]+){10})", txt, re.M):
            v = [float(x) for x in row[2].split()]
            months[(int(row[1]), int(row[0]))] = (v[3] if v[3] > 0 else v[2], v[8] if v[8] > 0 else v[7])
        issued = dt.date(int(update[1]), MON[update[2]], int(update[3])).isoformat()  # ESA's own 'Last Update'
        versions[dt.datetime.strptime(name[:8], "%Y%m%d").date()] = (update[0], months, issued)
    return monthly_issues([(first, months, issued) for first, (_, months, issued) in versions.items()], today)


def as_lists(issues):
    """{origin: issue} -> the web form: origins, and per issue the values for
    origin + 1, + 2, ... (null where missing), at most MAX_DAYS days."""
    origins = sorted(o for o in issues if o >= KEEP_FROM)
    doc = {"origins": [o.isoformat() for o in origins], "issued": [issues[o].get("issued", "") for o in origins],
           "available": [issues[o].get("available") or issues[o].get("issued", "")[:10] for o in origins]}
    for key in ("f107", "ap"):
        if not any(issues[o].get(key) or (key == "ap" and issues[o].get("ap3")) for o in origins):
            continue
        rows = []
        for o in origins:
            by_day = dict(issues[o].get(key) or {})
            if key == "ap":  # a daily Ap from every full day of 3-hour values
                for t, bins in (issues[o].get("ap3") or {}).items():
                    if None not in bins:
                        by_day.setdefault(t, round(sum(bins) / 8, 1))
            later = [t for t in by_day if t > o]
            n = min((max(later) - o).days, MAX_DAYS) if later else 0
            rows.append([by_day.get(o + dt.timedelta(days=k + 1)) for k in range(n)])
        doc[key] = rows
    if any(issues[o].get("ap3") for o in origins):
        rows = []
        for o in origins:
            bins = issues[o].get("ap3") or {}
            later = [t for t in bins if t > o]
            n = (max(later) - o).days if later else 0
            rows.append([v for k in range(n) for v in bins.get(o + dt.timedelta(days=k + 1), [None] * 8)])
        doc["ap3"] = rows
    return doc


def main():
    args = sys.argv[1:]
    out_path = args[args.index("--out") + 1] if "--out" in args else OUT
    os.makedirs(RECORD, exist_ok=True)
    today = dt.date.today()
    failed = []

    njit, sift = load("njit.json"), load("sift.json")
    steps = [("NJIT", lambda: fetch_njit(njit)), ("ADAPT-SIFT", lambda: fetch_sift(sift)),
             ("SWPC 45-day", fetch_swpc45), ("SWPC 3-day", fetch_swpc3),
             ("CCMC challenge page", fetch_ccmc)]
    if "--backfill" in args:
        steps.append(("archives", backfill))
    for name, step in steps:
        try:
            step()
        except Exception as e:  # noqa: BLE001 — keep what is saved and carry on
            print(f"WARNING: {name} fetch failed: {e}")
            failed.append(name)
    save("njit.json", njit)
    save("sift.json", sift)

    sources = {"njit": read_njit(njit), "sift": read_sift(sift), "swpc45": read_swpc45(),
               "swpc27": read_swpc27(), "swpc3": read_swpc3(), "cu": read_cu(), "esa": read_esa(),
               "schatten": read_schatten(today), "esalong": read_esalong(today)}
    nd, sd = sorted(njit), sorted(sift)
    doc = {"generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           # the F10.7 chart's "Show NJIT" and "Show ADAPT-SIFT"
           "njit": {"source": "NJIT 60-day F10.7 forecast; values for issue date + 0..59 days",
                    "issues": nd, "f107": [njit[d] for d in nd]},
           "sift": {"source": "ADAPT-SIFT F10.7 forecast, newest issue of each UT day; "
                              "values for issue time + 0..7 days",
                    "issues": sd, "issued_at": [sift[d]["issued"] for d in sd],
                    "f107": [sift[d]["f107"] for d in sd]},
           "sources": {k: {"name": NAMES[k], **as_lists(v)} for k, v in sources.items()}}
    with open(out_path + ".tmp", "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    os.replace(out_path + ".tmp", out_path)
    print("other forecasts: " + ", ".join(
        f"{k} {len(v)}" + (f" to {max(v)}" if v else "") for k, v in sources.items())
        + (f"; FAILED: {', '.join(failed)}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
