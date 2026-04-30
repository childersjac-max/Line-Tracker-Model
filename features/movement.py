# features/movement.py

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from configs.config import PUBLIC_BOOKS, SHARP_BOOKS
from utils.odds_math import american_to_implied_prob, line_move_in_prob, no_vig_prob_for_side


def _parse_ts(ts_str):
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except Exception:
        return None


def filter_snaps_by_window(snapshots, hours_back=None):
    if hours_back is None or not snapshots:
        return snapshots
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)
    return [s for s in snapshots if (_parse_ts(s.get("timestamp", "")) or cutoff) >= cutoff]


def get_opener(snapshots):
    for s in snapshots:
        if s.get("h2h") or s.get("spreads") or s.get("totals"):
            return s
    return snapshots[0] if snapshots else None


def get_latest(snapshots):
    return snapshots[-1] if snapshots else None


def _get_price(snap, market, side, book):
    if market == "h2h":
        return snap.get("h2h", {}).get(side, {}).get(book)
    if market == "spreads":
        entry = snap.get("spreads", {}).get(side, {}).get(book, {})
        return entry.get("price") if isinstance(entry, dict) else None
    if market == "totals":
        entry = snap.get("totals", {}).get(side, {}).get(book, {})
        return entry.get("price") if isinstance(entry, dict) else None
    return None


def _get_line(snap, market, side, book="pinnacle"):
    if market == "spreads":
        entry = snap.get("spreads", {}).get(side, {}).get(book, {})
        return entry.get("line") if isinstance(entry, dict) else None
    if market == "totals":
        entry = snap.get("totals", {}).get(side, {}).get(book, {})
        return entry.get("line") if isinstance(entry, dict) else None
    return None


def prob_move(snaps_a, snaps_b, market, side, book):
    s = get_opener(snaps_a) if snaps_a else None
    e = get_latest(snaps_b) if snaps_b else None
    if s is None or e is None:
        return None
    p_start = _get_price(s, market, side, book)
    p_end   = _get_price(e, market, side, book)
    if p_start is None or p_end is None:
        return None
    return line_move_in_prob(p_start, p_end)


def line_move(snaps_a, snaps_b, market, side, book="pinnacle"):
    s = get_opener(snaps_a) if snaps_a else None
    e = get_latest(snaps_b) if snaps_b else None
    if s is None or e is None:
        return None
    l_start = _get_line(s, market, side, book)
    l_end   = _get_line(e, market, side, book)
    if l_start is None or l_end is None:
        return None
    return float(l_end - l_start)


def move_speed(snapshots, market, side, book="pinnacle"):
    if not snapshots or len(snapshots) < 2:
        return 0.0
    first_price = last_price = first_ts = last_ts = None
    for s in snapshots:
        p = _get_price(s, market, side, book)
        ts = _parse_ts(s.get("timestamp", ""))
        if p is not None and ts is not None:
            if first_price is None:
                first_price, first_ts = p, ts
            last_price, last_ts = p, ts
    if first_price is None or last_price is None or first_ts == last_ts:
        return 0.0
    hours = (last_ts - first_ts).total_seconds() / 3600.0
    if hours < 0.01:
        return 0.0
    return line_move_in_prob(first_price, last_price) / hours


def sharp_pub_divergence(snap, market, side):
    pub   = [_get_price(snap, market, side, b) for b in PUBLIC_BOOKS]
    sharp = [_get_price(snap, market, side, b) for b in SHARP_BOOKS]
    pub   = [p for p in pub if p is not None]
    sharp = [p for p in sharp if p is not None]
    if not pub or not sharp:
        return None
    return float(max(pub) - max(sharp))


def cross_book_std(snap, market, side):
    prices = [_get_price(snap, market, side, b) for b in PUBLIC_BOOKS + SHARP_BOOKS]
    prices = [p for p in prices if p is not None]
    if len(prices) < 3:
        return None
    return float(np.std([american_to_implied_prob(p) for p in prices]))


def num_direction_changes(snapshots, market, side, book="pinnacle"):
    prices = [_get_price(s, market, side, book) for s in snapshots]
    prices = [p for p in prices if p is not None]
    if len(prices) < 3:
        return 0
    changes, prev_dir = 0, 0
    for i in range(1, len(prices)):
        d = prices[i] - prices[i-1]
        cur_dir = 1 if d > 0 else (-1 if d < 0 else 0)
        if cur_dir != 0 and prev_dir != 0 and cur_dir != prev_dir:
            changes += 1
        if cur_dir != 0:
            prev_dir = cur_dir
    return changes


def extract_features(record):
    snaps  = record.get("snapshots", [])
    market = record["market"]
    side   = record["side"]
    if not snaps:
        return None

    snaps_6h = filter_snaps_by_window(snaps, hours_back=6)
    opener   = get_opener(snaps)
    latest   = get_latest(snaps)
    if opener is None or latest is None:
        return None

    splits = latest.get("splits", {})
    htg    = latest.get("hours_to_game")

    pin_move_full = prob_move(snaps, snaps, market, side, "pinnacle")
    pin_move_6h   = prob_move(snaps_6h, snaps_6h, market, side, "pinnacle") if snaps_6h else None
    pub_moves     = [prob_move(snaps, snaps, market, side, b) for b in PUBLIC_BOOKS]
    pub_moves     = [m for m in pub_moves if m is not None]
    pub_move_avg  = float(np.mean(pub_moves)) if pub_moves else 0.0
    pub_move_std  = float(np.std(pub_moves)) if len(pub_moves) > 1 else 0.0
    line_move_full = line_move(snaps, snaps, market, side)
    line_move_6h   = line_move(snaps_6h, snaps_6h, market, side) if snaps_6h else None
    div_open   = sharp_pub_divergence(opener, market, side)
    div_latest = sharp_pub_divergence(latest, market, side)
    div_change = ((div_latest or 0) - (div_open or 0)) if div_open is not None and div_latest is not None else None
    std_open   = cross_book_std(opener, market, side)
    std_latest = cross_book_std(latest, market, side)
    speed_pin  = move_speed(snaps, market, side, "pinnacle")
    n_rev      = num_direction_changes(snaps, market, side, "pinnacle")

    mkey   = {"h2h": "moneyline", "spreads": "spread", "totals": "total"}.get(market, market)
    msplit = splits.get(mkey, {})
    sharp_money_pct   = msplit.get("sharp_money_pct")
    sharp_tickets_pct = msplit.get("sharp_tickets_pct")
    pub_money_pct     = msplit.get("public_money_pct")
    pub_tickets_pct   = msplit.get("public_tickets_pct")
    magnitude_pts     = msplit.get("magnitude_pts")
    money_vs_tickets  = (sharp_money_pct or 50) - (sharp_tickets_pct or 50)

    p_open = _get_price(opener, market, side, "pinnacle")
    p_late = _get_price(latest, market, side, "pinnacle")
    sig_sharp, sig_rlm, sig_fade = 0, 0, 0
    if p_open is not None and p_late is not None:
        pin_move_pts = abs(p_late - p_open)
        pub_avg_pts  = float(np.mean([abs((_get_price(latest, market, side, b) or p_open) - p_open) for b in PUBLIC_BOOKS]))
        sig_sharp = int(pin_move_pts >= 4.0 and pub_avg_pts <= 2.0)
        public_on = (pub_tickets_pct or 50) >= 65.0
        sig_rlm   = int(public_on and (p_late - p_open) >= 0.5)
        sig_fade  = int((pub_tickets_pct or 50) >= 70.0 and pin_move_pts <= 1.0 and (pub_money_pct or 50) < 50)

    n_snaps = len(snaps)
    hours_tracked = 0.0
    if n_snaps >= 2:
        t0 = _parse_ts(snaps[0].get("timestamp", ""))
        t1 = _parse_ts(snaps[-1].get("timestamp", ""))
        if t0 and t1:
            hours_tracked = (t1 - t0).total_seconds() / 3600.0

    pin_latest_price = _get_price(latest, market, side, "pinnacle")
    # NOTE: pin_implied_prob is the WITH-VIG Pinnacle prob, kept for backward compatibility only.
    # Use pin_no_vig_prob (below) as the fair-probability benchmark.
    pin_implied_prob = american_to_implied_prob(pin_latest_price) if pin_latest_price else 0.5

    # ── No-vig (devigged) probabilities — Tier 1 fix #1 ─────────────────────
    # The fair price comparison must strip Pinnacle's ~2-3% vig, otherwise every
    # edge is systematically understated.
    pin_no_vig_prob_open  = no_vig_prob_for_side(opener, market, side, "pinnacle")
    pin_no_vig_prob_close = no_vig_prob_for_side(latest, market, side, "pinnacle")
    pin_no_vig_prob       = (pin_no_vig_prob_close
                              if pin_no_vig_prob_close is not None
                              else (pin_no_vig_prob_open
                                    if pin_no_vig_prob_open is not None
                                    else pin_implied_prob))

    # ── CLV (closing line value) features — Tier 1 fix #6 ───────────────────
    # Positive clv_signed means the no-vig fair line moved TOWARD this side
    # between open and close. Across many bets, CLV > 0 is the single best
    # predictor of long-run profitability.
    if pin_no_vig_prob_open is not None and pin_no_vig_prob_close is not None:
        clv_signed = float(pin_no_vig_prob_close - pin_no_vig_prob_open)
    else:
        clv_signed = 0.0
    clv_abs = abs(clv_signed)
    has_clv = int(pin_no_vig_prob_open is not None and pin_no_vig_prob_close is not None)

    pub_prices = [_get_price(latest, market, side, b) for b in PUBLIC_BOOKS]
    pub_prices = [p for p in pub_prices if p is not None]
    best_pub_price = max(pub_prices) if pub_prices else None
    best_pub_book  = PUBLIC_BOOKS[pub_prices.index(best_pub_price)] if pub_prices else None

# ── Injury features ──────────────────────────────────────────────
    snap_injuries = latest.get("injuries", {})
    has_major_injury  = float(snap_injuries.get("has_major_injury",  0))
    home_injury_score = float(snap_injuries.get("home_injury_score", 0))
    away_injury_score = float(snap_injuries.get("away_injury_score", 0))
    injury_asymmetry  = abs(home_injury_score - away_injury_score)
    
    return {
        "event_id": record["event_id"], "sport_key": record["sport_key"],
        "commence_time": record.get("commence_time"),
        "home_team": record["home_team"], "away_team": record["away_team"],
        "market": market, "side": side, "is_home": record.get("is_home"),
        "line": record.get("line"), "best_pub_price": best_pub_price,
        "best_pub_book": best_pub_book,
        "pin_implied_prob": pin_implied_prob,
        "pin_no_vig_prob":       pin_no_vig_prob,
        "pin_no_vig_prob_open":  pin_no_vig_prob_open if pin_no_vig_prob_open is not None else pin_no_vig_prob,
        "pin_no_vig_prob_close": pin_no_vig_prob_close if pin_no_vig_prob_close is not None else pin_no_vig_prob,
        "clv_signed": clv_signed,
        "clv_abs":    clv_abs,
        "has_clv":    has_clv,
        "pin_move_full": pin_move_full or 0.0, "pin_move_6h": pin_move_6h or 0.0,
        "pub_move_avg": pub_move_avg, "pub_move_std": pub_move_std,
        "line_move_full": line_move_full or 0.0, "line_move_6h": line_move_6h or 0.0,
        "div_open": div_open or 0.0, "div_latest": div_latest or 0.0,
        "div_change": div_change or 0.0,
        "cross_book_std_open": std_open or 0.0, "cross_book_std_latest": std_latest or 0.0,
        "cross_book_std_change": ((std_latest or 0) - (std_open or 0)),
        "pin_move_speed": speed_pin, "n_reversals": n_rev,
        "n_snaps": n_snaps, "n_snaps_6h": len(snaps_6h),
        "hours_tracked": hours_tracked, "hours_to_game": htg or 0.0,
        "sharp_money_pct": sharp_money_pct or 50.0,
        "sharp_tickets_pct": sharp_tickets_pct or 50.0,
        "pub_money_pct": pub_money_pct or 50.0,
        "pub_tickets_pct": pub_tickets_pct or 50.0,
        "money_vs_tickets": money_vs_tickets, "magnitude_pts": magnitude_pts or 0.0,
        "sig_sharp": sig_sharp, "sig_rlm": sig_rlm, "sig_fade": sig_fade,
        "n_signals": sig_sharp + sig_rlm + sig_fade,
        "outcome": record.get("outcome"),
        "has_major_injury":   has_major_injury,
        "home_injury_score":  home_injury_score,
        "away_injury_score":  away_injury_score,
        "injury_asymmetry":   injury_asymmetry,
    }


FEATURE_COLS = [
    "pin_move_full", "pin_move_6h", "pub_move_avg", "pub_move_std",
    "line_move_full", "line_move_6h", "div_open", "div_latest", "div_change",
    "cross_book_std_open", "cross_book_std_latest", "cross_book_std_change",
    "pin_move_speed", "n_reversals", "n_snaps", "n_snaps_6h",
    "hours_tracked", "hours_to_game", "sharp_money_pct", "sharp_tickets_pct",
    "pub_money_pct", "pub_tickets_pct", "money_vs_tickets", "magnitude_pts",
    "sig_sharp", "sig_rlm", "sig_fade", "n_signals",
    # No-vig fair-prob & CLV (Tier 1 fixes #1 and #6)
    "pin_no_vig_prob", "pin_no_vig_prob_open", "pin_no_vig_prob_close",
    "clv_signed", "clv_abs", "has_clv",
    # Legacy with-vig prob retained (predictive on its own; harmless to keep)
    "pin_implied_prob", "is_home",
    "has_major_injury", "home_injury_score", "away_injury_score", "injury_asymmetry",
]


def build_feature_dataframe(records):
    rows = [extract_features(r) for r in records]
    rows = [r for r in rows if r is not None]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df
