# data/historical.py
# =====================================================================
# HISTORICAL DATA PULLER
# =====================================================================
# Pulls historical odds snapshots going back N days from the configured
# data source (The Odds API or OddsJam). Each day is sampled at multiple
# timestamps to simulate line movement, then completed scores are pulled
# to generate real outcome labels.
#
# This replaces synthetic training data with real historical data,
# dramatically improving model accuracy.
#
# API usage estimate (defaults):
#   THE ODDS API:  3 sports × 4 hrs × 30 days = 360 historical odds calls
#   ODDSJAM:       6 sports × 8 hrs × 365 days ≈ 17,520 calls
#                  (rate-limited; resume support skips already-pulled
#                   timestamps so you can re-run safely)
# =====================================================================

import os
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from data.sources import get_source

ODDS_API_KEY     = os.environ.get("ODDS_API_KEY", "")
ODDSJAM_API_KEY  = os.environ.get("ODDSJAM_API_KEY", "")
ODDS_DATA_SOURCE = os.environ.get("ODDS_DATA_SOURCE", "the_odds_api").strip().lower()
LINE_HISTORY_DIR = Path("./line_history")
OUTCOMES_FILE    = Path("./outcomes.json")

# Lazy source instance
_SOURCE = None
def _src():
    global _SOURCE
    if _SOURCE is None:
        _SOURCE = get_source(ODDS_DATA_SOURCE)
    return _SOURCE

LINE_HISTORY_DIR.mkdir(exist_ok=True)

# Sports we know how to pull. The Odds API path is expensive so the
# the-odds-api default is the original 3-sport set; OddsJam unlocks the
# full 6 by default. Override either with HISTORICAL_SPORTS env var
# (comma-separated sport_keys).
_FULL_SPORTS = {
    "americanfootball_nfl":   "NFL",
    "basketball_nba":         "NBA",
    "baseball_mlb":           "MLB",
    "icehockey_nhl":          "NHL",
    "americanfootball_ncaaf": "CFB",
    "basketball_ncaab":       "CBB",
}
_THE_ODDS_API_DEFAULT = {
    "basketball_nba": "NBA",
    "baseball_mlb":   "MLB",
    "icehockey_nhl":  "NHL",
}


def _resolve_historical_sports() -> dict:
    env = os.environ.get("HISTORICAL_SPORTS", "").strip()
    if env:
        keys = [k.strip() for k in env.split(",") if k.strip()]
        return {k: _FULL_SPORTS.get(k, k.upper()) for k in keys if k in _FULL_SPORTS}
    if ODDS_DATA_SOURCE == "oddsjam":
        return dict(_FULL_SPORTS)
    return dict(_THE_ODDS_API_DEFAULT)


HISTORICAL_SPORTS = _resolve_historical_sports()

# Sample these hours each day for line movement simulation.
# Override via DAILY_HOURS env var (comma-separated 0-23). OddsJam gets
# 8x/day by default for richer line-movement features.
_DEFAULT_HOURS_THE_ODDS_API = [9, 13, 17, 21]
_DEFAULT_HOURS_ODDSJAM      = [3, 7, 10, 13, 16, 19, 21, 23]


def _resolve_daily_hours() -> list[int]:
    env = os.environ.get("DAILY_HOURS", "").strip()
    if env:
        try:
            return sorted({int(h) for h in env.split(",") if h.strip()})
        except ValueError:
            pass
    return list(_DEFAULT_HOURS_ODDSJAM if ODDS_DATA_SOURCE == "oddsjam"
                else _DEFAULT_HOURS_THE_ODDS_API)


DAILY_HOURS = _resolve_daily_hours()

# Polite delay between historical-odds calls (seconds).
PULL_SLEEP_SEC = float(os.environ.get("HISTORICAL_PULL_SLEEP", "0.5"))

PUBLIC_BOOKS = ["draftkings", "fanduel", "betmgm", "bovada", "williamhill_us", "bet365"]
SHARP_BOOKS  = ["pinnacle", "circa", "bookmaker"]


# ── API helpers ───────────────────────────────────────────────────────

def fetch_historical_odds(sport_key, timestamp_str):
    """
    Fetch historical odds snapshot for a specific timestamp.
    timestamp_str: ISO format e.g. "2024-03-15T13:00:00Z"
    Returns list of event dicts in canonical (Odds API) shape.
    """
    return _src().fetch_historical_odds(
        sport_key, timestamp_str,
        markets=["h2h", "spreads", "totals"], regions="us,eu",
    )


def fetch_historical_scores(sport_key, days_from):
    """Fetch completed scores for the past N days. Source-aware: clamped
    to the source's max_scores_days when known."""
    src = _src()
    cap = getattr(src, "max_scores_days", 3)
    days_from = max(1, min(int(days_from), int(cap or days_from)))
    return src.fetch_scores(sport_key, days_from=days_from)


# ── Line history helpers ──────────────────────────────────────────────

def extract_book_odds(event):
    """Flatten bookmaker data — same as scraper."""
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


# ── Resume helpers ────────────────────────────────────────────────────

def _build_pulled_index() -> dict:
    """Scan line_history/ once and return {sport_key: set(timestamps)}.

    Used to skip (sport, timestamp) pairs we've already fetched. Massive
    speedup when re-running a 365-day OddsJam crawl after a partial run.
    A timestamp is considered 'pulled' for a sport if we have it on at
    least one event of that sport — historical pulls for a (sport, ts)
    pair return ALL events for that league at that moment, so partial
    coverage is essentially impossible.
    """
    idx: dict = {}
    if not LINE_HISTORY_DIR.exists():
        return idx
    for p in LINE_HISTORY_DIR.glob("*.json"):
        try:
            with open(p) as f:
                h = json.load(f)
        except Exception:
            continue
        sk = h.get("sport_key")
        if not sk:
            continue
        bucket = idx.setdefault(sk, set())
        for snap in h.get("snapshots", []):
            ts = snap.get("timestamp")
            if ts:
                bucket.add(ts)
    return idx


# ── Main historical pull ──────────────────────────────────────────────

def pull_historical_odds(days_back=30, resume=True):
    """
    Pull historical odds snapshots for the past N days.
    Builds line movement history for completed games.

    resume: skip (sport, timestamp) pairs already saved (default True).

    Returns: (n_events, n_snapshots)
    """
    print(f"\n=== PULLING HISTORICAL ODDS ({days_back} days) ===\n")

    now = datetime.now(timezone.utc)
    total_events    = 0
    total_snapshots = 0
    total_skipped   = 0

    # Generate timestamps: N days back, sampled at DAILY_HOURS
    timestamps = []
    for days_ago in range(days_back, 0, -1):
        day = now - timedelta(days=days_ago)
        for hour in DAILY_HOURS:
            ts = day.replace(hour=hour, minute=0, second=0, microsecond=0)
            if ts < now:
                timestamps.append(ts.strftime("%Y-%m-%dT%H:%M:%SZ"))

    print(f"Source:       {_src().name}")
    print(f"Sports:       {list(HISTORICAL_SPORTS.keys())}")
    print(f"Daily hours:  {DAILY_HOURS}")
    print(f"Timestamps:   {len(timestamps)} per sport "
          f"(across {days_back} days)")
    pulled_index = _build_pulled_index() if resume else {}
    if resume and pulled_index:
        already = sum(len(v) for v in pulled_index.values())
        print(f"Resume mode:  found {already} (sport,ts) pairs already pulled — will skip\n")
    else:
        print()

    for sport_key, label in HISTORICAL_SPORTS.items():
        print(f"─── {label} ───")
        sport_events  = 0
        sport_snaps   = 0
        sport_skipped = 0
        already_have  = pulled_index.get(sport_key, set())

        for ts in timestamps:
            if resume and ts in already_have:
                sport_skipped += 1
                total_skipped  += 1
                continue

            events = fetch_historical_odds(sport_key, ts)

            for event in events:
                flat     = extract_book_odds(event)
                event_id = flat["event_id"]
                if not event_id:
                    continue

                # Skip games already over at this snapshot time
                htg = hours_to_game(flat.get("commence_time", ""), ts)
                if htg is not None and htg < -3.0:
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
                    sport_events += 1
                    total_events += 1
                else:
                    existing_ts = {s["timestamp"] for s in history["snapshots"]}
                    if ts not in existing_ts:
                        history["snapshots"].append(snap)
                        history["snapshots"].sort(key=lambda s: s["timestamp"])

                save_history(history)
                sport_snaps  += 1
                total_snapshots += 1

            time.sleep(PULL_SLEEP_SEC)

        print(f"  {label}: {sport_events} new events, {sport_snaps} snapshots saved, "
              f"{sport_skipped} timestamps skipped (already pulled)")

    print(f"\n✅ Historical odds complete: {total_events} events, "
          f"{total_snapshots} snapshots, {total_skipped} skipped (resume).")
    return total_events, total_snapshots


def pull_historical_scores(days_back=30):
    """
    Pull completed game scores for the past N days.
    Maps results to event IDs in outcomes.json.

    For sources that cap days_from (The Odds API: 3) the call is clamped
    internally; for OddsJam (365) the full window is requested in chunks
    of source's `max_scores_days`.

    Returns: n_new_outcomes
    """
    print(f"\n=== PULLING HISTORICAL SCORES ({days_back} days) ===\n")

    if OUTCOMES_FILE.exists():
        with open(OUTCOMES_FILE) as f:
            outcomes = json.load(f)
    else:
        outcomes = {}

    new_count = 0
    src = _src()
    chunk = max(1, int(getattr(src, "max_scores_days", 3) or 3))

    for sport_key, label in HISTORICAL_SPORTS.items():
        print(f"  {label}: fetching scores ({days_back}d in chunks of {chunk}d)...")
        # Iterate in chunks so providers with a small days_from cap
        # (The Odds API = 3) still get coverage when called repeatedly
        # over many runs. For OddsJam (365) this is just one call.
        remaining = days_back
        while remaining > 0:
            window = min(remaining, chunk)
            games  = fetch_historical_scores(sport_key, days_from=window)

            for game in games:
                if not game.get("completed"):
                    continue
                eid  = game.get("id")
                home = game.get("home_team", "")
                away = game.get("away_team", "")
                if f"{eid}_home_ml" in outcomes:
                    continue

                scores = {
                    s["name"]: int(float(s["score"]))
                    for s in (game.get("scores") or [])
                    if s.get("score")
                }
                hs  = scores.get(home)
                as_ = scores.get(away)
                if hs is None or as_ is None:
                    continue

                outcomes[f"{eid}_home_ml"]    = int(hs > as_)
                outcomes[f"{eid}_away_ml"]    = int(as_ > hs)
                outcomes[f"{eid}_total"]      = hs + as_
                outcomes[f"{eid}_home_score"] = hs
                outcomes[f"{eid}_away_score"] = as_
                new_count += 1

            time.sleep(PULL_SLEEP_SEC)
            remaining -= window
            # Single-call providers (OddsJam) return everything in one shot
            # so further chunks would just be duplicates.
            if chunk >= days_back:
                break

    with open(OUTCOMES_FILE, "w") as f:
        json.dump(outcomes, f, indent=2)

    total_games = len([k for k in outcomes if k.endswith("_home_ml")])
    print(f"  Added {new_count} new outcomes. Total: {total_games} games labeled")
    return new_count


def run_historical_pull(days_back=30, resume=True):
    """Full historical pull: odds snapshots + scores."""
    if ODDS_DATA_SOURCE == "the_odds_api" and not ODDS_API_KEY:
        print("ERROR: ODDS_API_KEY not set")
        return
    if ODDS_DATA_SOURCE == "oddsjam" and not ODDSJAM_API_KEY:
        print("ERROR: ODDSJAM_API_KEY not set")
        return
    print(f"  [data source] {_src().name}")

    # Clamp days_back to the provider's max
    cap = getattr(_src(), "max_historical_days", days_back)
    if cap and days_back > cap:
        print(f"  [info] requested {days_back}d > provider cap {cap}d; clamping")
        days_back = cap

    n_events, n_snaps = pull_historical_odds(days_back=days_back, resume=resume)
    n_outcomes = pull_historical_scores(days_back=days_back)

    print(f"\n{'='*50}")
    print(f"HISTORICAL PULL COMPLETE")
    print(f"{'='*50}")
    print(f"  Events with history: {n_events}")
    print(f"  Total snapshots:     {n_snaps}")
    print(f"  New outcomes:        {n_outcomes}")
    print(f"\nRun train.py to retrain models on historical data.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30,
                    help="Days back to pull (default 30; OddsJam supports up to 365)")
    ap.add_argument("--no-resume", action="store_true",
                    help="Re-pull every (sport,timestamp) pair even if already saved")
    args = ap.parse_args()
    run_historical_pull(days_back=args.days, resume=not args.no_resume)
