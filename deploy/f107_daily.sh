#!/bin/bash
# Daily E24 F10.7 forecast: refresh data -> issue the forecast -> website
# views (deploy/web/) -> optional rsync to a web host. Then the same for Ap:
# GFZ Kp/ap file -> issue the Ap forecast -> views (deploy/web_ap/) -> rsync.
# The two halves are independent; a failure in one never stops the other.
# Meant for cron after ~01:30 UT, once the SRS for the UT day just ended is
# out (GFZ posts the finished day just before 00 UT), e.g.
#   30 21 * * * bash <repo>/deploy/f107_daily.sh >> <repo>/deploy/daily.log 2>&1
# One summary line per run; details go to the same log.
#
# Machine-specific settings live in deploy/local.env (gitignored), e.g.
#   F107_PUBLISH_DEST=user@host:/path/to/site/data/f107/
#   AP_PUBLISH_DEST=user@host:/path/to/site/data/ap/
#   TMPDIR=/some/scratch/dir
# Without a *_PUBLISH_DEST the views are built but not published.
set -uo pipefail
cd "$(dirname "$0")/.."
exec 9>deploy/.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) SKIP previous run still going"; exit 0; }
if [ -f deploy/local.env ]; then set -a; . deploy/local.env; set +a; fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=env/bin/python
echo "== $(date -u +%FT%TZ) start"

nice sh realtime/refresh_data.sh; refresh=$?
# predict.py exits 1 when the origin was already issued (it never overwrites
# the archive); that is not a failure here.
nice $PY realtime/predict.py; predict=$?
nice $PY deploy/web_json.py; web=$?
# SWPC 27-day outlook issues for the page's "Show SWPC"; on failure the
# previous swpc.json stays in deploy/web/ and is re-published as is.
nice $PY deploy/swpc_web.py; swpc=$?
publish() {  # publish <dir> <dest> -> ok | FAIL
    timeout -k 10 120 rsync -a --timeout=45 --chmod=D755,F644 --exclude="*.tmp" \
        -e "ssh -o BatchMode=yes -o ConnectTimeout=10" \
        --delete "$1" "$2" >&2 && echo ok || echo FAIL
}
pub=skip
if [ $web -eq 0 ] && [ -n "${F107_PUBLISH_DEST:-}" ]; then
    pub=$(publish deploy/web/ "$F107_PUBLISH_DEST")
fi

# Ap: same never-overwrite rule as F10.7 (exit 1 = origin already issued).
nice $PY scripts/build_ap.py; ap_data=$?
(cd ap && nice ../$PY predict.py); ap_predict=$?
nice $PY deploy/ap_web.py; ap_web=$?
ap_pub=skip
if [ $ap_web -eq 0 ] && [ -n "${AP_PUBLISH_DEST:-}" ]; then
    ap_pub=$(publish deploy/web_ap/ "$AP_PUBLISH_DEST")
fi
echo "$(date -u +%FT%TZ) DONE refresh=$refresh predict=$predict web=$web swpc=$swpc pub=$pub latest=$(sed -n 2p realtime/forecasts/latest.csv | cut -d, -f1)" \
     "ap_data=$ap_data ap_predict=$ap_predict ap_web=$ap_web ap_pub=$ap_pub ap_latest=$(sed -n 2p ap/forecasts/latest.csv | cut -d, -f1)"
