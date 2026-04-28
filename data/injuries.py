# data/injuries.py
# =====================================================================
# INJURY TRACKER
# =====================================================================
# Pulls injury data from ESPN's public API (no key required) and
# flags games where a key player is out or questionable.
#
# Adds a binary "has_major_injury" feature to line history snapshots
# so the model can distinguish injury-driven line moves from sharp money.
#
# Injury status hierarchy:
#   Out / Doubtful     → major_injury = 1  (line move likely injury-driven)
#   Questionable       → major_injury = 0.5
#   Probable / Active  → major_injury = 0
# =====================================================================

import json
import time
import requests
from pathlib import Path
from datetime import datetime, timezone

INJURY_CACHE_FILE = Path("./injury_cache.json")

# ESPN sport slugs
ESPN_SPORTS = {
    "basketball_nba":         "basketball/nba",
    "baseball_mlb":           "baseball/mlb",
    "icehockey_nhl":          "hockey/nhl",
    "americanfootball_nfl":   "football/nfl",
    "americanfootball_ncaaf": "football/college-football",
    "basketball_ncaab":       "basketball/mens-college-basketball",
}

# Player positions considered "key" — injury to these triggers the flag
KEY_POSITIONS = {
    "basketball_nba":         ["PG", "SG", "SF", "PF", "C", "G", "F"],
    "baseball_mlb":           ["SP", "P"],   # Starting pitcher is most impactful
    "icehockey_nhl":          ["G", "C", "LW", "RW", "D"],
    "americanfootball_nfl":   ["QB", "RB", "WR", "TE"],
    "americanfootball_ncaaf": ["QB", "RB", "WR"],
    "basketball_ncaab":       ["PG", "SG", "SF", "PF", "C"],
}

# Status strings from ESPN that indicate significant injury
MAJOR_STATUSES = {"out", "doubtful", "ir", "day-to-day"}
QUESTIONABLE_STATUSES = {"questionable"}


def fetch_espn_injuries(sport_key):
    """
    Fetch current injury report from ESPN for a sport.
    Returns list of injury dicts:
    {
        player_name, team, position, status, injury_type,
        is_major, sport_key
    }
    """
    slug = ESPN_SPORTS.get(sport_key)
    if not slug:
        return []

    url = f"https://site.api.espn.com/apis/site/v2/sports/{slug}/injuries"
    try:
        r = requests.get(url, timeout=10,
                         headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            print(f"  [injuries] {sport_key} → HTTP {r.status_code}")
            return []
        data = r.json()
    except Exception as e:
        print(f"  [injuries] {sport_key} error: {e}")
        return []

    injuries = []
    key_positions = set(KEY_POSITIONS.get(sport_key, []))

    for team_entry in data.get("injuries", []):
        team_name = team_entry.get("team", {}).get("displayName", "")
        for injury in team_entry.get("injuries", []):
            athlete    = injury.get("athlete", {})
            player     = athlete.get("displayName", "")
            position   = athlete.get("position", {}).get("abbreviation", "")
            status_raw = injury.get("status", "").lower()
            injury_type = injury.get("type", {}).get("description", "")

            is_major = 0.0
            if status_raw in MAJOR_STATUSES:
                is_major = 1.0
            elif status_raw in QUESTIONABLE_STATUSES:
                is_major = 0.5

            # Only flag key positions
            is_key_position = (position in key_positions) or (not key_positions)

            injuries.append({
                "player_name":  player,
                "team":         team_name,
                "position":     position,
                "status":       status_raw,
                "injury_type":  injury_type,
                "is_major":     is_major,
                "is_key_pos":   is_key_position,
                "sport_key":    sport_key,
            })

    return injuries


def fetch_all_injuries():
    """Fetch injuries for all sports. Returns dict keyed by sport_key."""
    all_injuries = {}
    for sport_key in ESPN_SPORTS:
        print(f"  Fetching injuries: {sport_key}")
        injuries = fetch_espn_injuries(sport_key)
        all_injuries[sport_key] = injuries
        print(f"    → {len(injuries)} injury records")
        time.sleep(0.3)
    return all_injuries


def load_injury_cache():
    if INJURY_CACHE_FILE.exists():
        with open(INJURY_CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_injury_cache(cache):
    with open(INJURY_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def refresh_injury_cache():
    """Fetch fresh injuries and save to cache. Call from scraper."""
    injuries = fetch_all_injuries()
    cache = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "injuries": injuries,
    }
    save_injury_cache(cache)
    total = sum(len(v) for v in injuries.values())
    print(f"  Injury cache updated: {total} total records")
    return injuries


def get_game_injury_flag(home_team, away_team, sport_key, injuries_by_sport=None):
    """
    Given a game's teams, return injury impact score for the game.

    Returns dict:
    {
        has_major_injury:  float  # 0=none, 0.5=questionable, 1.0=out/doubtful
        home_injury_score: float  # sum of major flags for home team
        away_injury_score: float  # sum of major flags for away team
        injured_players:   list   # names of key injured players
    }
    """
    if injuries_by_sport is None:
        cache = load_injury_cache()
        injuries_by_sport = cache.get("injuries", {})

    sport_injuries = injuries_by_sport.get(sport_key, [])

    home_norm = home_team.upper().strip() if home_team else ""
    away_norm = away_team.upper().strip() if away_team else ""

    home_score = 0.0
    away_score = 0.0
    injured_players = []

    for inj in sport_injuries:
        if not inj.get("is_key_pos"):
            continue
        team_norm = inj.get("team", "").upper().strip()
        is_major  = float(inj.get("is_major", 0))
        if is_major == 0:
            continue

        player_name = inj.get("player_name", "")

        # Fuzzy team match — check if team name contains or is contained by game teams
        if (team_norm and home_norm and
                (team_norm in home_norm or home_norm in team_norm)):
            home_score += is_major
            injured_players.append(f"{player_name} (Home, {inj.get('status','')})")
        elif (team_norm and away_norm and
                (team_norm in away_norm or away_norm in team_norm)):
            away_score += is_major
            injured_players.append(f"{player_name} (Away, {inj.get('status','')})")

    max_score = max(home_score, away_score)

    return {
        "has_major_injury":  min(max_score, 1.0),
        "home_injury_score": home_score,
        "away_injury_score": away_score,
        "injured_players":   injured_players,
    }


def enrich_line_history_with_injuries(histories, injuries_by_sport=None):
    """
    Add injury features to the latest snapshot of each line history.
    Call after refresh_injury_cache().

    Mutates histories in place and saves updated files.
    Returns count of enriched records.
    """
    from data.line_tracker import save_history

    if injuries_by_sport is None:
        cache = load_injury_cache()
        injuries_by_sport = cache.get("injuries", {})

    enriched = 0
    for hist in histories:
        if not hist.get("snapshots"):
            continue
        sport_key  = hist.get("sport_key", "")
        home_team  = hist.get("home_team", "")
        away_team  = hist.get("away_team", "")

        flags = get_game_injury_flag(home_team, away_team, sport_key, injuries_by_sport)

        # Attach to the latest snapshot
        latest = hist["snapshots"][-1]
        latest["injuries"] = flags
        enriched += 1
        save_history(hist)

    return enriched
