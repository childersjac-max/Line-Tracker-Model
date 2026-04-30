"""Abstract base class for an odds-data source."""

from abc import ABC, abstractmethod
from typing import Iterable


class OddsSource(ABC):
    """All providers must return data in the canonical Odds-API JSON shape.
    See data/sources/__init__.py for the schema."""

    name: str = "base"

    # ---------- live / pre-match odds ----------
    @abstractmethod
    def fetch_current_odds(
        self,
        sport_key: str,
        markets: Iterable[str] | None = None,
        regions: str = "us,eu",
    ) -> list[dict]:
        """Return list of events with bookmakers/markets/outcomes."""
        ...

    @abstractmethod
    def fetch_player_props_for_event(
        self,
        sport_key: str,
        event_id: str,
        markets: Iterable[str] | None = None,
    ) -> dict:
        """Return a single event payload populated with player-prop markets.
        Should return {} if the provider has no props for that event."""
        ...

    # ---------- historical ----------
    @abstractmethod
    def fetch_historical_odds(
        self,
        sport_key: str,
        timestamp_iso: str,
        markets: Iterable[str] | None = None,
        regions: str = "us,eu",
    ) -> list[dict]:
        """Return list of events as they appeared at `timestamp_iso`
        (ISO 8601, UTC, e.g. '2026-03-15T13:00:00Z')."""
        ...

    # ---------- scores / results ----------
    @abstractmethod
    def fetch_scores(self, sport_key: str, days_from: int = 3) -> list[dict]:
        """Return completed-game score records (canonical shape).

        Providers should accept the largest `days_from` they can serve;
        callers may request up to a year for backfill purposes. Providers
        that cap the lookback (e.g. The Odds API at 3 days) should clamp
        internally and document the limit."""
        ...

    # ---------- optional ----------
    def fetch_injuries(self, sport_key: str) -> list[dict]:
        """Optional. Default = empty list. Implement if the provider supports it."""
        return []

    def fetch_arbitrage_opportunities(
        self,
        sport_key: str | None = None,
        markets: Iterable[str] | None = None,
    ) -> list[dict]:
        """Optional. Pre-computed arbitrage feed from the provider, if any.

        Default returns []. The arbitrage angle in features/arbitrage.py
        will still detect arbs locally from the cross-book odds we
        already pull, so this is purely an optional hint.

        Providers that implement this should return a list of records with
        the canonical shape:
          {
            "event_id":     str,
            "sport_key":    str,
            "commence_time": str,
            "home_team":    str,
            "away_team":    str,
            "market":       "h2h" | "spreads" | "totals",
            "margin_pct":   float,                 # >0 if arb
            "legs": [
              {"side": str, "book": str, "price": int, "line": float|None},
              {"side": str, "book": str, "price": int, "line": float|None},
            ],
          }
        """
        return []

    # ---------- max-history hints ----------
    @property
    def max_historical_days(self) -> int:
        """Maximum days back the provider will reliably serve historical
        odds for. Bootstrap scripts use this to decide how far to crawl.
        Override in subclasses; default is conservative."""
        return 30

    @property
    def max_scores_days(self) -> int:
        """Maximum days back the scores endpoint will return. The Odds
        API caps at 3; OddsJam goes much further."""
        return 3

    def __repr__(self) -> str:
        return f"<OddsSource name={self.name!r}>"
