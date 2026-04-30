# features/arbitrage.py
# =====================================================================
# ARBITRAGE DETECTION
# =====================================================================
# A two-way arbitrage exists when the inverse-decimal-prices of the two
# best opposing prices, taken from possibly different books, sum to less
# than 1.0:
#
#     1/dec_A + 1/dec_B  <  1.0
#
# The "edge" / margin is:  margin_pct = (1 - (1/dec_A + 1/dec_B)) * 100
#
# This module is provider-agnostic: it works on any flat snapshot in our
# canonical shape ({h2h: {side: {book: price}}, spreads/totals: {side:
# {book: {line, price}}}}) -- so it produces signals identically whether
# the underlying source is The Odds API or OddsJam.
#
# OddsJam's flagship "Positive EV / Arbitrage" feed can ALSO be pulled
# directly via OddsJamSource.fetch_arbitrage_opportunities() and is
# merged in when present; locally-detected arbs are kept as a fallback
# so the angle still works on every provider.
# =====================================================================

from __future__ import annotations

from typing import Iterable, Optional

from utils.odds_math import american_to_decimal


# ---- minimum profit margin (%) to flag a side as part of an arb -----
# Set above zero to ignore razor-thin "arbs" that are usually stale-line
# artifacts. Override via configs.config.MIN_ARB_MARGIN_PCT.
DEFAULT_MIN_ARB_MARGIN_PCT = 0.5


def _safe_dec(price) -> Optional[float]:
    if price is None:
        return None
    try:
        d = american_to_decimal(price)
        if d <= 1.0:
            return None
        return float(d)
    except Exception:
        return None


def _two_way_arb(price_a, price_b) -> Optional[dict]:
    """Compute arb stats for a candidate pair; None if not an arb."""
    da = _safe_dec(price_a)
    db = _safe_dec(price_b)
    if da is None or db is None:
        return None
    inv_sum = (1.0 / da) + (1.0 / db)
    if inv_sum >= 1.0:
        return None
    margin = (1.0 - inv_sum) * 100.0
    # Equal-profit stake split (sum to 1.0)
    stake_a = (1.0 / da) / inv_sum
    stake_b = (1.0 / db) / inv_sum
    return {
        "margin_pct": margin,
        "inv_sum":    inv_sum,
        "stake_a":    stake_a,
        "stake_b":    stake_b,
    }


def _opposing_side(market_dict) -> dict:
    """Given snap[market], return {side: other_side_label}."""
    sides = list(market_dict.keys())
    if len(sides) < 2:
        return {}
    out = {}
    # We support 2-way markets (the only kind in MARKETS = h2h/spreads/totals).
    # Pair every side with every OTHER side; in practice only one pair exists.
    for s in sides:
        for o in sides:
            if o != s:
                out[s] = o
                break
    return out


def _iter_books_h2h(market_dict, side):
    for bk, price in (market_dict.get(side) or {}).items():
        if price is not None:
            yield bk, price, None  # h2h has no line


def _iter_books_pointed(market_dict, side):
    """Spreads / totals iterator. Yields (book, price, line)."""
    for bk, entry in (market_dict.get(side) or {}).items():
        if not isinstance(entry, dict):
            continue
        price = entry.get("price")
        line  = entry.get("line")
        if price is None:
            continue
        yield bk, price, line


def _lines_match(a, b) -> bool:
    """Spreads/totals: arb only counts when the two sides are at the
    SAME (mirror) line — for spreads home -L vs away +L, for totals
    Over L vs Under L."""
    if a is None or b is None:
        return False
    # For totals both legs are the same number; for spreads the magnitudes
    # match (one is +L, one is -L). Allow tiny float tolerance.
    return abs(abs(a) - abs(b)) < 0.001


def arb_features_for_side(
    flat_snapshot: dict,
    market: str,
    side: str,
    min_margin_pct: float = DEFAULT_MIN_ARB_MARGIN_PCT,
) -> dict:
    """
    Compute arbitrage signal features for ONE side of a market.

    Returns dict with:
      is_arb_side       (1.0 / 0.0)  — this side is one leg of an arb
      arb_margin_pct    (float)      — best arb margin found pairing
                                        this side's BEST price with any
                                        opposing side's available price
      arb_book          (str|None)   — the book offering THIS leg
      arb_partner_book  (str|None)   — the book offering the other leg
      arb_partner_price (int|None)   — opposing leg American price
      arb_partner_line  (float|None) — opposing leg line (spreads/totals)
      arb_book_count    (int)        — how many books are quoting this
                                        side at a price that COULD be an
                                        arb leg vs at least one partner
    """
    empty = {
        "is_arb_side":       0.0,
        "arb_margin_pct":    0.0,
        "arb_book":          None,
        "arb_partner_book":  None,
        "arb_partner_price": None,
        "arb_partner_line":  None,
        "arb_book_count":    0,
    }
    if not flat_snapshot:
        return empty
    market_dict = flat_snapshot.get(market) or {}
    pairs = _opposing_side(market_dict)
    other = pairs.get(side)
    if other is None:
        return empty

    is_pointed = market in ("spreads", "totals")
    this_iter  = list(_iter_books_pointed(market_dict, side))  if is_pointed else list(_iter_books_h2h(market_dict, side))
    other_iter = list(_iter_books_pointed(market_dict, other)) if is_pointed else list(_iter_books_h2h(market_dict, other))
    if not this_iter or not other_iter:
        return empty

    best = None
    book_count = 0
    for tb, tp, tl in this_iter:
        local_best_for_book = None
        for ob, op, ol in other_iter:
            if is_pointed and not _lines_match(tl, ol):
                continue
            res = _two_way_arb(tp, op)
            if res is None:
                continue
            if local_best_for_book is None or res["margin_pct"] > local_best_for_book["margin_pct"]:
                local_best_for_book = {**res, "this_book": tb, "this_price": tp,
                                        "this_line": tl, "partner_book": ob,
                                        "partner_price": op, "partner_line": ol}
        if local_best_for_book is not None:
            book_count += 1
            if best is None or local_best_for_book["margin_pct"] > best["margin_pct"]:
                best = local_best_for_book

    if best is None or best["margin_pct"] < min_margin_pct:
        out = dict(empty)
        out["arb_book_count"] = book_count
        return out

    return {
        "is_arb_side":       1.0,
        "arb_margin_pct":    float(best["margin_pct"]),
        "arb_book":          best["this_book"],
        "arb_partner_book":  best["partner_book"],
        "arb_partner_price": int(best["partner_price"]) if best["partner_price"] is not None else None,
        "arb_partner_line":  best["partner_line"],
        "arb_book_count":    book_count,
    }


def find_all_arbs_in_snapshot(
    flat_snapshot: dict,
    markets: Iterable[str] = ("h2h", "spreads", "totals"),
    min_margin_pct: float = DEFAULT_MIN_ARB_MARGIN_PCT,
) -> list[dict]:
    """
    Scan a flat snapshot for every arb across the requested markets.
    Returns a list of arb records (one per side that participates).
    """
    out = []
    for market in markets:
        market_dict = flat_snapshot.get(market) or {}
        for side in market_dict.keys():
            r = arb_features_for_side(flat_snapshot, market, side, min_margin_pct)
            if r.get("is_arb_side"):
                out.append({
                    "event_id":      flat_snapshot.get("event_id"),
                    "sport_key":     flat_snapshot.get("sport_key"),
                    "commence_time": flat_snapshot.get("commence_time"),
                    "home_team":     flat_snapshot.get("home_team"),
                    "away_team":     flat_snapshot.get("away_team"),
                    "market":        market,
                    "side":          side,
                    **r,
                })
    return out
