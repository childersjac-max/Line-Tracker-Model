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
        """Return completed-game score records (canonical shape)."""
        ...

    # ---------- optional ----------
    def fetch_injuries(self, sport_key: str) -> list[dict]:
        """Optional. Default = empty list. Implement if the provider supports it."""
        return []

    def __repr__(self) -> str:
        return f"<OddsSource name={self.name!r}>"
