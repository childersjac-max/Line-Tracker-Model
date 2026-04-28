# data/splits.py
# =====================================================================
# PUBLIC BETTING SPLITS — Action Network API
# =====================================================================
# Replaces the broken VSiN scraper with Action Network's public API
# which provides real-time ticket % and money % for all major markets.
#
# Action Network's API is publicly accessible with no key required.
# It provides:
#   - Ticket % (% of bets placed on each side)
#   - Money %  (% of money wagered on each side)
#   - Line consensus across books
#
# The gap between ticket % and money % is the core sharp signal:
#   money % >> ticket % = big bets on that side = sharp action
# =====================================================================

import json
import time
import requests
from datetime import datetime, timezone
from pathlib import Path

SPLITS_CACHE_FILE = Path("./splits_cache.json")

# Action Network sport IDs
AN_SPORTS = {
    "basketball_nba":         "2",
    "baseball_mlb":           "3",
    "icehockey_nhl":          "4",
    "americanfootball_nfl":   "1",
    "americanfootball_ncaaf": "6",
    "basketball_ncaab":       "5",
}

AN_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept":     "application/json",
    "Referer":    "https://www.actionnetwork.com/",
}

AN_BASE = "https://api.actionnetwork.com/web/v1"


def fetch_an_games(sport_id):
    """Fetch today's games from Action Network."""
    url = f"{AN_BASE}/scoreboard?sport={sport_id}&period=game&bookIds=15,30,68,69,123,71"
    try:
        r = requests.get(url, headers=AN_HEADERS, timeout=10)
        if r.status_code != 200:
            return []
        return r.json().get("games", [])
    except Exception as e:
        print(f"  [splits] fetch_an_games error: {e}")
        return []


def fetch_an_splits(game_id):
    """
    Fetch betting splits for a specific game from Action Network.
    Returns dict with market splits or empty dict.
    """
    url = f"{AN_BASE}/games/{game_id}/betting-splits"
    try:
        r = requests.get(url, headers=AN_HEADERS, timeout=10)
        if r.status_code != 200:
            return {}
        return r.json()
    except Exception as e:
        return {}


def parse_splits(splits_data, home_team, away_team):
    """
    Parse Action Network splits response into a clean dict.

    Returns:
    {
        moneyline: { sharp_side, sharp_money_pct, sharp_tickets_pct,
                     public_money_pct, public_tickets_pct, magnitude_pts },
        spread:    { ... same ... },
        total:     { sharp_side="over"/"under", ... }
    }
    """
    result = {}
    if not splits_data:
        return result

    for market_data in splits_data.get("splits", []):
        market_type = market_data.get("type", "").lower()

        if market_type not in ("moneyline", "spread", "total"):
            continue

        outcomes = market_data.get("outcomes", [])
        if len(outcomes) < 2:
            continue

        # Extract ticket and money percentages
        parsed_outcomes = []
        for o in outcomes:
            name        = o.get("name", "")
            ticket_pct  = float(o.get("bets_pct",  0) or 0)
            money_pct   = float(o.get("money_pct", 0) or 0)
            parsed_outcomes.append({
                "name":       name,
                "ticket_pct": ticket_pct,
                "money_pct":  money_pct,
            })

        if len(parsed_outcomes) < 2:
            continue

        # Determine sharp side = side with higher money % than ticket %
        o1, o2 = parsed_outcomes[0], parsed_outcomes[1]
        diff1 = o1["money_pct"] - o1["ticket_pct"]
        diff2 = o2["money_pct"] - o2["ticket_pct"]

        if diff1 >= diff2:
            sharp   = o1
            public  = o2
        else:
            sharp   = o2
            public  = o1

        magnitude = abs(diff1 - diff2)

        if market_type == "total":
            sharp_side = "over" if "over" in sharp["name"].lower() else "under"
        else:
            sharp_side = "home" if sharp["name"] == home_team else "away"

        result[market_type] = {
            "sharp_side":         sharp_side,
            "sharp_money_pct":    round(sharp["money_pct"],  1),
            "sharp_tickets_pct":  round(sharp["ticket_pct"], 1),
            "public_money_pct":   round(public["money_pct"],  1),
            "public_tickets_pct": round(public["ticket_pct"], 1),
            "magnitude_pts":      round(magnitude, 1),
            "money_vs_tickets":   round(sharp["money_pct"] - sharp["ticket_pct"], 1),
        }

    return result


def normalize_team_name(name):
    """Simple normalization for team name matching."""
    return name.upper().strip().replace(".", "").replace("-", " ")


def fetch_all_splits():
    """
    Fetch splits for all sports and all today's games.
    Returns dict: { sport_key: [ { home, away, splits } ] }
    """
    all_splits = {}

    for sport_key, sport_id in AN_SPORTS.items():
        print(f"  Fetching splits: {sport_key}")
        games = fetch_an_games(sport_id)
        sport_splits = []

        for game in games:
            game_id   = game.get("id")
            home_team = game.get("home_team", {}).get("full_name", "")
            away_team = game.get("away_team", {}).get("full_name", "")

            if not game_id:
                continue

            splits_raw = fetch_an_splits(game_id)
            splits     = parse_splits(splits_raw, home_team, away_team)

            sport_splits.append({
                "an_game_id": game_id,
                "home_team":  home_team,
                "away_team":  away_team,
                "splits":     splits,
            })
            time.sleep(0.2)

        all_splits[sport_key] = sport_splits
        print(f"    → {len(sport_splits)} games with splits")
        time.sleep(0.5)

    return all_splits


def load_splits_cache():
    if SPLITS_CACHE_FILE.exists():
        with open(SPLITS_CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_splits_cache(cache):
    with open(SPLITS_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def refresh_splits_cache():
    """Fetch fresh splits and save to cache. Call from scraper."""
    splits = fetch_all_splits()
    cache  = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "splits":    splits,
    }
    save_splits_cache(cache)
    total = sum(len(v) for v in splits.values())
    print(f"  Splits cache updated: {total} games")
    return splits


def get_game_splits(home_team, away_team, sport_key, splits_by_sport=None):
    """
    Look up splits for a specific game by team name matching.
    Returns splits dict (moneyline/spread/total) or empty dict.
    """
    if splits_by_sport is None:
        cache = load_splits_cache()
        splits_by_sport = cache.get("splits", {})

    sport_games = splits_by_sport.get(sport_key, [])
    home_norm   = normalize_team_name(home_team)
    away_norm   = normalize_team_name(away_team)

    for game in sport_games:
        g_home = normalize_team_name(game.get("home_team", ""))
        g_away = normalize_team_name(game.get("away_team", ""))

        # Match if any word in team name matches
        home_match = any(w in g_home for w in home_norm.split() if len(w) > 3)
        away_match = any(w in g_away for w in away_norm.split() if len(w) > 3)

        if home_match and away_match:
            return game.get("splits", {})

    return {}


def enrich_line_history_with_splits(histories, splits_by_sport=None):
    """
    Add Action Network splits data to the latest snapshot of each history.
    Replaces the VSiN splits data that was previously empty.

    Returns count of enriched records.
    """
    from data.line_tracker import save_history

    if splits_by_sport is None:
        cache = load_splits_cache()
        splits_by_sport = cache.get("splits", {})

    enriched = 0
    for hist in histories:
        if not hist.get("snapshots"):
            continue

        sport_key = hist.get("sport_key", "")
        home_team = hist.get("home_team", "")
        away_team = hist.get("away_team", "")

        splits = get_game_splits(home_team, away_team, sport_key, splits_by_sport)
        if not splits:
            continue

        # Attach to latest snapshot in the format the feature engine expects
        latest = hist["snapshots"][-1]
        latest["splits"] = splits
        enriched += 1
        save_history(hist)

    return enriched
