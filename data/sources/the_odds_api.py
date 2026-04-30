"""
The Odds API adapter (the-odds-api.com).

Reads ODDS_API_KEY from the environment.
This is the original data path the codebase has always used; the response
shape is already the canonical shape, so no normalization is needed.
"""

import os
import requests
from typing import Iterable
from .base import OddsSource


class TheOddsApiSource(OddsSource):
    name = "the_odds_api"
    BASE = "https://api.the-odds-api.com/v4"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("ODDS_API_KEY", "")

    # ---------- helpers ----------
    def _get(self, path: str, params: dict, timeout: int = 15):
        params = {**params, "apiKey": self.api_key}
        r = requests.get(f"{self.BASE}{path}", params=params, timeout=timeout)
        if r.status_code != 200:
            print(f"  [the_odds_api] {path} -> HTTP {r.status_code}: {r.text[:200]}")
            return None
        return r

    # ---------- current odds ----------
    def fetch_current_odds(
        self,
        sport_key: str,
        markets: Iterable[str] | None = None,
        regions: str = "us,eu",
    ) -> list[dict]:
        markets = list(markets) if markets else ["h2h", "spreads", "totals"]
        r = self._get(
            f"/sports/{sport_key}/odds",
            {
                "regions":    regions,
                "markets":    ",".join(markets),
                "oddsFormat": "american",
            },
        )
        if r is None:
            return []
        return r.json()

    def fetch_player_props_for_event(
        self,
        sport_key: str,
        event_id: str,
        markets: Iterable[str] | None = None,
    ) -> dict:
        if not markets:
            return {}
        r = self._get(
            f"/sports/{sport_key}/events/{event_id}/odds",
            {
                "regions":    "us",
                "markets":    ",".join(markets),
                "oddsFormat": "american",
            },
        )
        if r is None:
            return {}
        return r.json()

    # ---------- historical ----------
    def fetch_historical_odds(
        self,
        sport_key: str,
        timestamp_iso: str,
        markets: Iterable[str] | None = None,
        regions: str = "us,eu",
    ) -> list[dict]:
        markets = list(markets) if markets else ["h2h", "spreads", "totals"]
        r = self._get(
            f"/historical/sports/{sport_key}/odds",
            {
                "regions":    regions,
                "markets":    ",".join(markets),
                "oddsFormat": "american",
                "date":       timestamp_iso,
            },
            timeout=20,
        )
        if r is None:
            return []
        data = r.json()
        # The Odds API wraps historical responses in {"data": [...], ...}
        if isinstance(data, dict):
            remaining = r.headers.get("x-requests-remaining", "?")
            print(f"    {timestamp_iso} -> {len(data.get('data', []))} events | {remaining} requests remaining")
            return data.get("data", [])
        return data or []

    # ---------- scores ----------
    def fetch_scores(self, sport_key: str, days_from: int = 3) -> list[dict]:
        r = self._get(f"/sports/{sport_key}/scores", {"daysFrom": days_from})
        if r is None:
            return []
        return r.json()
