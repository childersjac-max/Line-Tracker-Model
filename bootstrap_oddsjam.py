#!/usr/bin/env python3
# bootstrap_oddsjam.py
# =====================================================================
# ONE-SHOT MAX-HISTORY BOOTSTRAP FOR ODDSJAM
# =====================================================================
# Pulls as much historical data as your OddsJam plan will give you:
#   • all 6 supported leagues (NFL, NBA, MLB, NHL, CFB, CBB)
#   • 8 snapshots per day (3a/7a/10a/1p/4p/7p/9p/11p UTC)
#   • up to 365 days back (configurable, capped by provider)
#   • completed scores for the same window
#
# Resume support is on by default — re-running after a network blip just
# skips (sport, timestamp) pairs that already exist on disk, so you can
# run this overnight without losing progress.
#
# USAGE:
#   export ODDS_DATA_SOURCE=oddsjam
#   export ODDSJAM_API_KEY=<your key>
#   python bootstrap_oddsjam.py                # 365 days, all 6 sports
#   python bootstrap_oddsjam.py --days 90      # last 90 days only
#   python bootstrap_oddsjam.py --sports NBA,NHL
#   python bootstrap_oddsjam.py --hours 9,15,21
#   python bootstrap_oddsjam.py --no-resume    # force re-pull
#   python bootstrap_oddsjam.py --dry-run      # estimate API calls only
#
# After the pull, retrain:
#   python train.py
# =====================================================================

import argparse
import os
import sys
from datetime import datetime, timezone, timedelta


SPORT_ALIAS_TO_KEY = {
    "NFL":  "americanfootball_nfl",
    "CFB":  "americanfootball_ncaaf", "NCAAF": "americanfootball_ncaaf",
    "NBA":  "basketball_nba",
    "CBB":  "basketball_ncaab",       "NCAAB": "basketball_ncaab",
    "MLB":  "baseball_mlb",
    "NHL":  "icehockey_nhl",
}


def _resolve_sports(arg: str | None) -> list[str]:
    if not arg:
        return list(SPORT_ALIAS_TO_KEY.values() | set())  # all 6
    out = []
    seen = set()
    for token in arg.split(","):
        t = token.strip().upper()
        if not t:
            continue
        key = SPORT_ALIAS_TO_KEY.get(t, t.lower())
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Pull max OddsJam historical odds + scores into ./line_history and ./outcomes.json"
    )
    ap.add_argument("--days", type=int, default=365,
                    help="Days back to pull (default 365; clamped to provider's max)")
    ap.add_argument("--sports", default="",
                    help="Comma-separated sport list (NFL,NBA,MLB,NHL,CFB,CBB). "
                         "Default = all 6.")
    ap.add_argument("--hours", default="",
                    help="Comma-separated UTC hours to sample per day. "
                         "Default = 3,7,10,13,16,19,21,23 (8x/day).")
    ap.add_argument("--sleep", type=float, default=0.5,
                    help="Seconds to sleep between API calls (default 0.5).")
    ap.add_argument("--no-resume", action="store_true",
                    help="Re-pull every (sport, timestamp) pair even if saved.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Just estimate the call count and exit.")
    args = ap.parse_args()

    # Force the OddsJam source for this script.
    os.environ["ODDS_DATA_SOURCE"] = "oddsjam"
    if not os.environ.get("ODDSJAM_API_KEY"):
        print("ERROR: ODDSJAM_API_KEY is not set in your environment.")
        return 2

    sports = _resolve_sports(args.sports)
    if args.hours:
        os.environ["DAILY_HOURS"] = args.hours
    if sports:
        os.environ["HISTORICAL_SPORTS"] = ",".join(sports)
    os.environ["HISTORICAL_PULL_SLEEP"] = str(args.sleep)

    # Import after env vars are set so module-level resolution picks them up.
    from data import historical as H
    from data.sources import get_source

    src = get_source("oddsjam")
    cap = int(getattr(src, "max_historical_days", args.days) or args.days)
    days = min(args.days, cap)

    n_sports = len(H.HISTORICAL_SPORTS)
    n_hours  = len(H.DAILY_HOURS)
    est_odds_calls   = n_sports * n_hours * days
    est_score_calls  = n_sports * max(1, days // max(1, getattr(src, "max_scores_days", 365) or 365))
    est_total        = est_odds_calls + est_score_calls
    est_secs         = est_total * args.sleep

    print("=" * 60)
    print("OddsJam max-history bootstrap")
    print("=" * 60)
    print(f"  source ........ {src.name}")
    print(f"  sports ({n_sports}) .. {list(H.HISTORICAL_SPORTS.keys())}")
    print(f"  hours/day ({n_hours}) {H.DAILY_HOURS}")
    print(f"  days back ..... {days} (provider cap = {cap})")
    print(f"  resume mode ... {'OFF' if args.no_resume else 'ON'}")
    print(f"  sleep/call .... {args.sleep:.2f}s")
    print()
    print(f"  estimated historical-odds calls  : {est_odds_calls:>7,}")
    print(f"  estimated scores calls           : {est_score_calls:>7,}")
    print(f"  estimated total calls            : {est_total:>7,}")
    print(f"  estimated wall-clock (no resume) : {est_secs/60:>7,.1f} min "
          f"({est_secs/3600:.1f} h)")
    print("=" * 60)

    if args.dry_run:
        print("\n--dry-run set; exiting without pulling.")
        return 0

    print()
    H.run_historical_pull(days_back=days, resume=not args.no_resume)
    print("\nDone. Next: `python train.py` to retrain on the new history.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
