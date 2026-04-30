"""
OddsJam adapter (api-external.oddsjam.com).

Reads ODDSJAM_API_KEY from the environment. Activate by setting:
    ODDS_DATA_SOURCE=oddsjam
    ODDSJAM_API_KEY=<your key>

==================================================================
ONE-TIME 5-MINUTE SETUP STEP
==================================================================
OddsJam's developer portal is gated behind a paid subscription, so
the exact endpoint paths and JSON field names below were written
against OddsJam's publicly documented v2 API surface plus
community-known conventions. After you sign up, open
https://docs.oddsjam.com/ and confirm:

  1) BASE_URL (we use https://api-external.oddsjam.com/api/v2)
  2) Authentication style (we use ?key=<key> query param;
     some accounts use header X-API-Key instead -- toggle USE_HEADER_AUTH).
  3) Endpoint paths in PATHS below.
  4) Field names in the _normalize_* helpers (especially how OddsJam
     names: home_team, away_team, sportsbook, market, bet_name, line,
     price). If anything differs, edit the constants/helpers below;
     the rest of the codebase doesn't need to change.

If something is misnamed, fetch_current_odds() will return [] and
print the HTTP body. That's your debugging hook.
==================================================================
"""

import os
import requests
from typing import Iterable
from .base import OddsSource


# ---------- constants you may need to tweak after signup ------------
BASE_URL = "https://api-external.oddsjam.com/api/v2"
USE_HEADER_AUTH = False           # set True if your account uses X-API-Key header
HEADER_AUTH_NAME = "X-API-Key"

PATHS = {
    "games":            "/games",                # current/upcoming games for a sport
    "game_odds":        "/game-odds",            # odds (mainlines + props) for given games
    "historical_odds":  "/historical-odds",      # historical snapshot at a timestamp
    "scores":           "/scores",               # completed-game scores
    "injuries":         "/injuries",             # injury reports (optional)
    "arbitrage":        "/arbitrage",            # OddsJam arbitrage feed (optional)
    "positive_ev":      "/positive-ev",          # OddsJam +EV feed (not used; here for reference)
}

# How far back OddsJam will reliably serve historical odds.
# OddsJam's marketing claims multi-year coverage; 365 is a safe default
# for the bootstrap script. Adjust per your subscription tier.
MAX_HISTORICAL_DAYS = 365
MAX_SCORES_DAYS     = 365

# Map our internal Odds-API sport_keys -> OddsJam league identifiers.
# Confirm these against /sports in your dashboard. Any league name OddsJam
# returns for these sports should work; the strings below are the conventional
# ones in their public materials.
SPORT_TO_LEAGUE = {
    "americanfootball_nfl":   "NFL",
    "americanfootball_ncaaf": "NCAAF",
    "basketball_nba":         "NBA",
    "basketball_ncaab":       "NCAAB",
    "baseball_mlb":           "MLB",
    "icehockey_nhl":          "NHL",
}
SPORT_TO_SPORT = {                    # OddsJam top-level "sport" param
    "americanfootball_nfl":   "football",
    "americanfootball_ncaaf": "football",
    "basketball_nba":         "basketball",
    "basketball_ncaab":       "basketball",
    "baseball_mlb":           "baseball",
    "icehockey_nhl":          "hockey",
}

# Maps our market names to OddsJam's. Confirm against your dashboard.
MARKET_NAME = {
    "h2h":     "Moneyline",
    "spreads": "Point Spread",
    "totals":  "Total Points",
}
# Reverse lookup for normalization
MARKET_KEY_FROM_ODDSJAM = {v: k for k, v in MARKET_NAME.items()}

# Maps OddsJam sportsbook keys -> our internal canonical book keys
# (the ones used in PUBLIC_BOOKS / SHARP_BOOKS in scraper.py & configs).
BOOK_REMAP = {
    "DraftKings":     "draftkings",
    "FanDuel":        "fanduel",
    "BetMGM":         "betmgm",
    "Bovada":         "bovada",
    "Caesars":        "williamhill_us",   # Caesars = WilliamHill US in The Odds API world
    "Bet365":         "bet365",
    "Pinnacle":       "pinnacle",
    "Circa Sports":   "circa",
    "BookMaker":      "bookmaker",
    "BookMaker.eu":   "bookmaker",
}

# How long to wait per request
DEFAULT_TIMEOUT = 20


# ---------- adapter ----------
class OddsJamSource(OddsSource):
    name = "oddsjam"

    # Long-history hints (read by bootstrap_oddsjam.py)
    @property
    def max_historical_days(self) -> int:
        return MAX_HISTORICAL_DAYS

    @property
    def max_scores_days(self) -> int:
        return MAX_SCORES_DAYS

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("ODDSJAM_API_KEY", "")
        if not self.api_key:
            print("[oddsjam] WARNING: ODDSJAM_API_KEY not set; all calls will fail.")

    # ---------- helpers ----------
    def _get(self, path: str, params: dict | None = None, timeout: int = DEFAULT_TIMEOUT):
        params = dict(params or {})
        headers = {}
        if USE_HEADER_AUTH:
            headers[HEADER_AUTH_NAME] = self.api_key
        else:
            params["key"] = self.api_key
        try:
            r = requests.get(f"{BASE_URL}{path}", params=params, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            print(f"  [oddsjam] {path} -> network error: {e}")
            return None
        if r.status_code != 200:
            print(f"  [oddsjam] {path} -> HTTP {r.status_code}: {r.text[:300]}")
            return None
        try:
            return r.json()
        except ValueError:
            print(f"  [oddsjam] {path} -> non-JSON body: {r.text[:200]}")
            return None

    @staticmethod
    def _book_key(oj_book: str) -> str:
        return BOOK_REMAP.get(oj_book, oj_book.lower().replace(" ", "_") if oj_book else "unknown")

    @staticmethod
    def _to_iso(ts) -> str:
        """OddsJam timestamps are usually ISO 8601 already; pass through."""
        if not ts:
            return ""
        if isinstance(ts, str):
            return ts if ts.endswith("Z") or "+" in ts else ts + "Z"
        return str(ts)

    # ---------- current odds ----------
    def fetch_current_odds(
        self,
        sport_key: str,
        markets: Iterable[str] | None = None,
        regions: str = "us,eu",   # OddsJam ignores this, kept for interface parity
    ) -> list[dict]:
        markets = list(markets) if markets else ["h2h", "spreads", "totals"]
        league = SPORT_TO_LEAGUE.get(sport_key)
        if not league:
            print(f"  [oddsjam] no league mapping for {sport_key}")
            return []

        # 1) list games for the league
        games = self._get(PATHS["games"], {
            "sport":  SPORT_TO_SPORT.get(sport_key, ""),
            "league": league,
            "is_live": "false",
        })
        if not games:
            return []
        games_list = games.get("data") if isinstance(games, dict) else games
        if not games_list:
            return []
        game_ids = [g.get("id") or g.get("game_id") for g in games_list if (g.get("id") or g.get("game_id"))]
        if not game_ids:
            return []

        # 2) pull odds for those games
        oj_markets = [MARKET_NAME[m] for m in markets if m in MARKET_NAME]
        odds_payload = self._get(PATHS["game_odds"], {
            "sportsbook": "",                          # blank = all books on your plan
            "market":     ",".join(oj_markets),
            "game_id":    ",".join(game_ids[:50]),     # most plans cap batch size; chunk if needed
        })
        if not odds_payload:
            return []
        odds_rows = odds_payload.get("data") if isinstance(odds_payload, dict) else odds_payload

        # 3) normalize -> canonical Odds-API event shape
        return self._normalize_events(games_list, odds_rows or [], sport_key)

    def fetch_player_props_for_event(
        self,
        sport_key: str,
        event_id: str,
        markets: Iterable[str] | None = None,
    ) -> dict:
        markets = list(markets) if markets else []
        if not markets:
            return {}
        # OddsJam exposes player props through the same /game-odds endpoint
        # but with prop market names instead of mainline names. Confirm exact
        # market labels against your dashboard (e.g. "Player Points",
        # "Player Rebounds", "Pitcher Strikeouts").
        odds_payload = self._get(PATHS["game_odds"], {
            "sportsbook": "",
            "market":     ",".join(markets),
            "game_id":    event_id,
        })
        if not odds_payload:
            return {}
        odds_rows = odds_payload.get("data") if isinstance(odds_payload, dict) else odds_payload
        if not odds_rows:
            return {}
        # Reuse the per-event normalizer
        events = self._normalize_events(
            [{"id": event_id, "sport": SPORT_TO_SPORT.get(sport_key, ""),
              "league": SPORT_TO_LEAGUE.get(sport_key, "")}],
            odds_rows,
            sport_key,
            include_props=True,
        )
        return events[0] if events else {}

    # ---------- historical ----------
    def fetch_historical_odds(
        self,
        sport_key: str,
        timestamp_iso: str,
        markets: Iterable[str] | None = None,
        regions: str = "us,eu",
    ) -> list[dict]:
        markets = list(markets) if markets else ["h2h", "spreads", "totals"]
        league = SPORT_TO_LEAGUE.get(sport_key)
        if not league:
            return []
        oj_markets = [MARKET_NAME[m] for m in markets if m in MARKET_NAME]

        payload = self._get(PATHS["historical_odds"], {
            "sport":     SPORT_TO_SPORT.get(sport_key, ""),
            "league":    league,
            "market":    ",".join(oj_markets),
            "timestamp": timestamp_iso,
        }, timeout=30)
        if not payload:
            return []
        rows = payload.get("data") if isinstance(payload, dict) else payload
        if not rows:
            return []
        # OddsJam historical typically returns rows already grouped by game.
        # We feed them through the same normalizer; the games list is derived
        # from the rows themselves.
        games_seen = {}
        for row in rows:
            gid = row.get("game_id") or row.get("id")
            if not gid:
                continue
            if gid not in games_seen:
                games_seen[gid] = {
                    "id":            gid,
                    "home_team":     row.get("home_team"),
                    "away_team":     row.get("away_team"),
                    "start_date":    row.get("start_date") or row.get("commence_time"),
                    "sport":         SPORT_TO_SPORT.get(sport_key, ""),
                    "league":        league,
                }
        return self._normalize_events(list(games_seen.values()), rows, sport_key)

    # ---------- scores ----------
    def fetch_scores(self, sport_key: str, days_from: int = 3) -> list[dict]:
        league = SPORT_TO_LEAGUE.get(sport_key)
        if not league:
            return []
        payload = self._get(PATHS["scores"], {
            "sport":     SPORT_TO_SPORT.get(sport_key, ""),
            "league":    league,
            "days_from": days_from,
        })
        if not payload:
            return []
        rows = payload.get("data") if isinstance(payload, dict) else payload
        return self._normalize_scores(rows or [], sport_key)

    # ---------- arbitrage (optional, OddsJam-specific) ----------
    def fetch_arbitrage_opportunities(
        self,
        sport_key: str | None = None,
        markets: Iterable[str] | None = None,
    ) -> list[dict]:
        """Pull OddsJam's pre-computed arbitrage feed.

        Returns canonical arb records (see base.py docstring) or [] if the
        endpoint isn't available on your plan. Local arb detection in
        features/arbitrage.py runs independently, so an empty result here
        does NOT disable the arbitrage angle in the model."""
        markets = list(markets) if markets else ["h2h", "spreads", "totals"]
        params: dict = {}
        if sport_key:
            league = SPORT_TO_LEAGUE.get(sport_key)
            if league:
                params["sport"]  = SPORT_TO_SPORT.get(sport_key, "")
                params["league"] = league
        oj_markets = [MARKET_NAME[m] for m in markets if m in MARKET_NAME]
        if oj_markets:
            params["market"] = ",".join(oj_markets)

        payload = self._get(PATHS["arbitrage"], params)
        if not payload:
            return []
        rows = payload.get("data") if isinstance(payload, dict) else payload
        if not rows:
            return []
        out = []
        for row in rows:
            mkt_raw = row.get("market") or row.get("market_name") or ""
            mkt     = MARKET_KEY_FROM_ODDSJAM.get(mkt_raw, mkt_raw.lower().replace(" ", "_"))
            legs_raw = row.get("legs") or row.get("bets") or []
            legs = []
            for leg in legs_raw:
                legs.append({
                    "side":  leg.get("name") or leg.get("bet_name") or leg.get("side"),
                    "book":  self._book_key(leg.get("sportsbook") or leg.get("book") or ""),
                    "price": _to_int_american(leg.get("price") or leg.get("odds")
                                                or leg.get("american_odds")),
                    "line":  _to_float(leg.get("point") or leg.get("line")
                                        or leg.get("handicap")),
                })
            margin = row.get("profit_margin") or row.get("margin_pct") or row.get("arbitrage_percentage")
            try:
                margin = float(margin) if margin is not None else 0.0
            except (TypeError, ValueError):
                margin = 0.0
            # Some payloads express margin as decimal (0.012) rather than %
            if 0.0 < margin < 1.0:
                margin = margin * 100.0
            out.append({
                "event_id":      row.get("game_id") or row.get("id") or row.get("event_id"),
                "sport_key":     sport_key or row.get("sport_key", ""),
                "commence_time": self._to_iso(row.get("start_date") or row.get("commence_time")),
                "home_team":     row.get("home_team", ""),
                "away_team":     row.get("away_team", ""),
                "market":        mkt,
                "margin_pct":    margin,
                "legs":          legs,
            })
        return out

    # ---------- injuries (optional) ----------
    def fetch_injuries(self, sport_key: str) -> list[dict]:
        league = SPORT_TO_LEAGUE.get(sport_key)
        if not league:
            return []
        payload = self._get(PATHS["injuries"], {
            "sport":  SPORT_TO_SPORT.get(sport_key, ""),
            "league": league,
        })
        if not payload:
            return []
        rows = payload.get("data") if isinstance(payload, dict) else payload
        return rows or []

    # =================================================================
    # NORMALIZATION HELPERS
    # If OddsJam's field names ever differ from what we assume below,
    # this is the only place you need to fix.
    # =================================================================
    def _normalize_events(
        self,
        games: list[dict],
        odds_rows: list[dict],
        sport_key: str,
        include_props: bool = False,
    ) -> list[dict]:
        """Group flat odds_rows by game_id, then by sportsbook, then by market."""
        sport_title = sport_key.split("_")[-1].upper()

        # Index games by id for quick lookup
        games_by_id = {}
        for g in games:
            gid = g.get("id") or g.get("game_id")
            if gid:
                games_by_id[gid] = g

        # Bucket odds by game_id -> book_key -> market_key -> outcomes[]
        from collections import defaultdict
        buckets: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for row in odds_rows:
            gid = row.get("game_id") or row.get("id")
            if not gid:
                continue
            oj_book = row.get("sportsbook") or row.get("book") or ""
            oj_mkt  = row.get("market") or row.get("market_name") or ""
            mkey    = MARKET_KEY_FROM_ODDSJAM.get(oj_mkt)
            if mkey is None:
                # If it's a player prop and we want props, keep the OddsJam name.
                if include_props and oj_mkt:
                    mkey = oj_mkt.lower().replace(" ", "_")
                else:
                    continue
            book_key = self._book_key(oj_book)
            outcome = {
                "name":  row.get("name") or row.get("bet_name") or row.get("selection"),
                "price": _to_int_american(row.get("price") or row.get("odds")
                                          or row.get("american_odds")),
                "point": _to_float(row.get("point") or row.get("line")
                                    or row.get("handicap")),
                "description": row.get("player_name") or row.get("description"),
            }
            buckets[gid][book_key][mkey].append(outcome)

        events = []
        for gid, by_book in buckets.items():
            g = games_by_id.get(gid, {})
            bookmakers = []
            for bk_key, by_market in by_book.items():
                bookmakers.append({
                    "key":         bk_key,
                    "title":       bk_key.title(),
                    "last_update": "",
                    "markets":     [{"key": mk, "outcomes": outs} for mk, outs in by_market.items()],
                })
            events.append({
                "id":            gid,
                "sport_key":     sport_key,
                "sport_title":   sport_title,
                "commence_time": self._to_iso(g.get("start_date") or g.get("commence_time")),
                "home_team":     g.get("home_team", ""),
                "away_team":     g.get("away_team", ""),
                "bookmakers":    bookmakers,
            })
        return events

    def _normalize_scores(self, rows: list[dict], sport_key: str) -> list[dict]:
        out = []
        for row in rows:
            gid  = row.get("id") or row.get("game_id")
            if not gid:
                continue
            home = row.get("home_team", "")
            away = row.get("away_team", "")
            hs   = row.get("home_score") if row.get("home_score") is not None else row.get("home_team_score")
            as_  = row.get("away_score") if row.get("away_score") is not None else row.get("away_team_score")
            completed = bool(row.get("completed") or row.get("is_completed") or row.get("final"))
            if hs is None or as_ is None:
                continue
            out.append({
                "id":            gid,
                "sport_key":     sport_key,
                "commence_time": self._to_iso(row.get("start_date") or row.get("commence_time")),
                "completed":     completed,
                "home_team":     home,
                "away_team":     away,
                "scores": [
                    {"name": home, "score": str(int(float(hs)))},
                    {"name": away, "score": str(int(float(as_)))},
                ],
            })
        return out


# ---------- module-level coercers ----------
def _to_int_american(v):
    if v is None or v == "":
        return None
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return None


def _to_float(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
