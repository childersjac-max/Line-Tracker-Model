# bootstrap_history.py
# =====================================================================
# ONE-TIME SCRIPT — run manually via GitHub Actions to seed the model
# with historical odds data so it has line movement to learn from.
#
# The Odds API provides historical snapshots for the past few days.
# We pull multiple timestamps per event to simulate movement history.
#
# Run once:
#   python bootstrap_history.py
# =====================================================================

import os
import json
import time
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path

from data.sources import get_source

ODDS_API_KEY     = os.environ.get("ODDS_API_KEY", "")
ODDSJAM_API_KEY  = os.environ.get("ODDSJAM_API_KEY", "")
ODDS_DATA_SOURCE = os.environ.get("ODDS_DATA_SOURCE", "the_odds_api").strip().lower()
LINE_HISTORY_DIR = Path("./line_history")
LINE_HISTORY_DIR.mkdir(exist_ok=True)

_SOURCE = None
def _src():
    global _SOURCE
    if _SOURCE is None:
        _SOURCE = get_source(ODDS_DATA_SOURCE)
    return _SOURCE

SPORTS = {
    "basketball_nba":      "NBA",
    "baseball_mlb":        "MLB",
    "icehockey_nhl":       "NHL",
    "americanfootball_ncaaf": "CFB",
}

PUBLIC_BOOKS = ["draftkings", "fanduel", "betmgm", "bovada", "williamhill_us", "bet365"]
SHARP_BOOKS  = ["pinnacle", "circa", "bookmaker"]


def fetch_historical_odds(sport_key, date_str, markets="h2h,spreads,totals"):
    """
    Fetch historical odds snapshot for a specific date.
    date_str format: 2024-01-15T12:00:00Z
    """
    return _src().fetch_historical_odds(
        sport_key, date_str,
        markets=[m.strip() for m in markets.split(",") if m.strip()],
        regions="us,eu",
    )


def extract_book_odds(event):
    """Same as scraper — flatten bookmaker data."""
    out = {
        "event_id":      event.get("id"),
        "sport_key":     event.get("sport_key"),
        "commence_time": event.get("commence_time"),
        "home_team":     event.get("home_team"),
        "away_team":     event.get("away_team"),
        "h2h":     {},
        "spreads": {},
        "totals":  {},
    }
    for bk in event.get("bookmakers", []):
        bk_key = bk["key"]
        for market in bk.get("markets", []):
            mkey = market["key"]
            for o in market.get("outcomes", []):
                name  = o.get("name")
                price = o.get("price")
                point = o.get("point")
                if mkey == "h2h":
                    out["h2h"].setdefault(name, {})[bk_key] = price
                elif mkey == "spreads":
                    out["spreads"].setdefault(name, {})[bk_key] = {"line": point, "price": price}
                elif mkey == "totals":
                    out["totals"].setdefault(name, {})[bk_key] = {"line": point, "price": price}
    return out


def hours_to_game(commence_str, snapshot_ts):
    try:
        commence = datetime.fromisoformat(commence_str.replace("Z", "+00:00"))
        snap_dt  = datetime.fromisoformat(snapshot_ts.replace("Z", "+00:00"))
        return (commence - snap_dt).total_seconds() / 3600.0
    except Exception:
        return None


def load_history(event_id):
    p = LINE_HISTORY_DIR / f"{event_id}.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def save_history(history):
    p = LINE_HISTORY_DIR / f"{history['event_id']}.json"
    with open(p, "w") as f:
        json.dump(history, f, indent=2)


def bootstrap():
    print(f"\n=== BOOTSTRAP HISTORICAL ODDS ===\n")

    if ODDS_DATA_SOURCE == "the_odds_api" and not ODDS_API_KEY:
        print("Set ODDS_API_KEY env var first.")
        return
    if ODDS_DATA_SOURCE == "oddsjam" and not ODDSJAM_API_KEY:
        print("Set ODDSJAM_API_KEY env var first.")
        return
    print(f"  [data source] {_src().name}")

    # Pull snapshots from the past 3 days at 6-hour intervals
    # This gives ~12 movement snapshots per event
    now = datetime.now(timezone.utc)
    timestamps = []
    for days_back in range(3, 0, -1):
        for hour in [0, 6, 12, 18]:
            dt = (now - timedelta(days=days_back)).replace(
                hour=hour, minute=0, second=0, microsecond=0
            )
            timestamps.append(dt.strftime("%Y-%m-%dT%H:%M:%SZ"))

    # Also add today's snapshots
    for hour in [0, 6, 12]:
        if hour <= now.hour:
            dt = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            timestamps.append(dt.strftime("%Y-%m-%dT%H:%M:%SZ"))

    print(f"Pulling {len(timestamps)} historical snapshots per sport...")
    print(f"Timestamps: {timestamps[0]} → {timestamps[-1]}\n")

    total_events = 0
    total_snaps  = 0

    for sport_key, label in SPORTS.items():
        print(f"--- {label} ---")
        for ts in timestamps:
            events = fetch_historical_odds(sport_key, ts)
            if not events:
                time.sleep(0.5)
                continue

            for event in events:
                flat     = extract_book_odds(event)
                event_id = flat["event_id"]
                if not event_id:
                    continue

                # Skip games that started before this snapshot
                htg = hours_to_game(flat.get("commence_time", ""), ts)
                if htg is not None and htg < -1.0:
                    continue

                snap = {
                    "timestamp":     ts,
                    "hours_to_game": htg,
                    "h2h":     flat["h2h"],
                    "spreads": flat["spreads"],
                    "totals":  flat["totals"],
                    "splits":  {},
                }

                history = load_history(event_id)
                if history is None:
                    history = {
                        "event_id":      event_id,
                        "sport_key":     sport_key,
                        "home_team":     flat["home_team"],
                        "away_team":     flat["away_team"],
                        "commence_time": flat["commence_time"],
                        "snapshots":     [snap],
                    }
                    total_events += 1
                else:
                    # Avoid duplicate timestamps
                    existing_ts = {s["timestamp"] for s in history["snapshots"]}
                    if ts not in existing_ts:
                        history["snapshots"].append(snap)

                save_history(history)
                total_snaps += 1

            print(f"  {ts}: {len(events)} events")
            time.sleep(0.3)  # Be polite to the API

        time.sleep(1.0)

    print(f"\n✅ Bootstrap complete!")
    print(f"   Events tracked: {total_events}")
    print(f"   Snapshots added: {total_snaps}")
    print(f"   History files: {len(list(LINE_HISTORY_DIR.glob('*.json')))}")
    print(f"\nNow run: python train.py --sport all --market all --synthetic --min-samples 10")


if __name__ == "__main__":
    bootstrap()
