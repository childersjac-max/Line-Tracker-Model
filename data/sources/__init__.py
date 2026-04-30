"""
Pluggable odds-data-source layer.

Switch providers by setting the env var ODDS_DATA_SOURCE:
  - "the_odds_api"  (default) -> data.sources.the_odds_api.TheOddsApiSource
  - "oddsjam"                  -> data.sources.oddsjam.OddsJamSource

Every adapter normalizes its responses to the canonical "Odds API" event shape
that the rest of the codebase already expects:

  {
    "id":            str,
    "sport_key":     str,
    "sport_title":   str,
    "commence_time": str (ISO 8601 UTC, e.g. "2026-04-30T19:00:00Z"),
    "home_team":     str,
    "away_team":     str,
    "bookmakers": [
      {
        "key":   str,
        "title": str,
        "last_update": str,
        "markets": [
          {
            "key": "h2h" | "spreads" | "totals" | "<player_prop_key>",
            "outcomes": [
              {"name": str, "price": int (american), "point": float|None,
               "description": str|None  (player name for props)},
              ...
            ]
          },
          ...
        ]
      },
      ...
    ]
  }

Scores are normalized to:
  {
    "id": str,
    "sport_key": str,
    "commence_time": str,
    "completed": bool,
    "home_team": str,
    "away_team": str,
    "scores": [{"name": team, "score": "<int as str>"}]
  }
"""

import os
from .base import OddsSource


def get_source(name: str | None = None) -> OddsSource:
    """Factory. Reads ODDS_DATA_SOURCE env var if `name` is None."""
    if name is None:
        name = os.environ.get("ODDS_DATA_SOURCE", "the_odds_api").strip().lower()

    if name in ("the_odds_api", "theoddsapi", "odds_api"):
        from .the_odds_api import TheOddsApiSource
        return TheOddsApiSource()

    if name == "oddsjam":
        from .oddsjam import OddsJamSource
        return OddsJamSource()

    raise ValueError(
        f"Unknown ODDS_DATA_SOURCE={name!r}. "
        f"Valid: 'the_odds_api', 'oddsjam'."
    )


__all__ = ["OddsSource", "get_source"]
