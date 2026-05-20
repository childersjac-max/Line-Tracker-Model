# data/paper_log.py
# Walk-forward paper-trading log: each predict run appends picks; grade-paper
# fills closing line, true CLV, and W/L from outcomes + line_history.

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from configs.config import (
    CLOSING_SNAP_MAX_HOURS_BEFORE_START,
    OUTPUT_DIR,
    PAPER_LOG_FILE,
    SPORTS,
)
from data.line_tracker import load_history
from data.results import load_outcomes
from features.movement import get_closing_snapshot
from utils.odds_math import american_to_decimal, compute_clv_vs_close, no_vig_prob_for_side

logger = logging.getLogger(__name__)

PAPER_LOG_COLUMNS = [
    "log_id",
    "slate_generated_at",
    "status",
    "sport",
    "sport_key",
    "market",
    "side",
    "line",
    "event_id",
    "is_home",
    "matchup",
    "commence_time",
    "american_odds",
    "book",
    "model_prob",
    "sized_prob",
    "fair_prob",
    "edge_pct",
    "edge_pct_raw",
    "shrinkage_alpha",
    "ev_pct",
    "bet_pct",
    "bet_usd",
    "confidence",
    "signals",
    "trained_on",
    "closing_fair_prob",
    "closing_american_odds",
    "clv_prob_pts",
    "clv_pct",
    "outcome",
    "pnl",
    "graded_at",
]


def _log_path():
    return Path(PAPER_LOG_FILE)


def _pick_key(row):
    line = row.get("line")
    if line is None or (isinstance(line, float) and pd.isna(line)):
        line = ""
    return f"{row.get('event_id')}|{row.get('market')}|{row.get('side')}|{line}"


def _load_log():
    p = _log_path()
    if not p.exists():
        return pd.DataFrame(columns=PAPER_LOG_COLUMNS)
    df = pd.read_csv(p)
    for col in PAPER_LOG_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[PAPER_LOG_COLUMNS]


def _save_log(df):
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    df.to_csv(_log_path(), index=False)


def append_slate(slate_df, slate_generated_at=None):
    """
    Append picks from a predict run. Dedupes by event|market|side|line — first
    entry wins (same policy as bpr-model tracker log).
    """
    if slate_df is None or slate_df.empty:
        logger.info("Paper log: empty slate, nothing to append.")
        return 0

    ts = slate_generated_at or datetime.now(timezone.utc).isoformat()
    existing = _load_log()
    seen = set(existing.apply(_pick_key, axis=1).tolist()) if not existing.empty else set()

    rows = []
    for _, r in slate_df.iterrows():
        key = _pick_key(r)
        if key in seen:
            continue
        seen.add(key)
        log_id = f"{r.get('event_id', '')}_{r.get('market', '')}_{len(rows)}_{ts[:10]}"
        rows.append({
            "log_id": log_id,
            "slate_generated_at": ts,
            "status": "pending",
            "sport": r.get("sport"),
            "sport_key": r.get("sport_key"),
            "market": r.get("market"),
            "side": r.get("side"),
            "line": r.get("line"),
            "event_id": r.get("event_id"),
            "is_home": r.get("is_home"),
            "matchup": r.get("matchup"),
            "commence_time": r.get("game_time") or "",
            "american_odds": r.get("american_odds"),
            "book": r.get("book"),
            "model_prob": r.get("model_prob"),
            "sized_prob": r.get("sized_prob"),
            "fair_prob": r.get("fair_prob"),
            "edge_pct": r.get("edge_pct"),
            "edge_pct_raw": r.get("edge_pct_raw"),
            "shrinkage_alpha": r.get("shrinkage_alpha"),
            "ev_pct": r.get("ev_pct"),
            "bet_pct": r.get("bet_pct"),
            "bet_usd": r.get("bet_usd"),
            "confidence": r.get("confidence"),
            "signals": r.get("signals"),
            "trained_on": r.get("trained_on"),
            "closing_fair_prob": None,
            "closing_american_odds": None,
            "clv_prob_pts": None,
            "clv_pct": None,
            "outcome": None,
            "pnl": None,
            "graded_at": None,
        })

    if not rows:
        logger.info("Paper log: all slate rows already logged.")
        return 0

    updated = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    _save_log(updated)
    logger.info("Paper log: appended %d new pick(s) → %s", len(rows), _log_path())
    return len(rows)


def _closing_prices_for_side(snap, market, side):
    """Pinnacle closing price + no-vig fair for one side."""
    if not snap:
        return None, None
    fair = no_vig_prob_for_side(snap, market, side, "pinnacle")
    if market == "h2h":
        price = (snap.get("h2h") or {}).get(side, {}).get("pinnacle")
    elif market == "spreads":
        entry = (snap.get("spreads") or {}).get(side, {}).get("pinnacle", {})
        price = entry.get("price") if isinstance(entry, dict) else None
    elif market == "totals":
        entry = (snap.get("totals") or {}).get(side, {}).get("pinnacle", {})
        price = entry.get("price") if isinstance(entry, dict) else None
    else:
        price = None
    return fair, price


def _resolve_outcome(event_id, market, side, is_home, line, outcomes):
    if market == "h2h":
        key = f"{event_id}_home_ml" if is_home else f"{event_id}_away_ml"
        o = outcomes.get(key)
        return int(o) if o is not None else None
    if market == "spreads":
        hs = outcomes.get(f"{event_id}_home_score")
        aw = outcomes.get(f"{event_id}_away_score")
        if hs is None or aw is None or line is None:
            return None
        try:
            line = float(line)
        except (TypeError, ValueError):
            return None
        margin = (hs - aw) if is_home else (aw - hs)
        adjusted = margin + line
        if adjusted == 0:
            return None
        return 1 if adjusted > 0 else 0
    if market == "totals":
        total_pts = outcomes.get(f"{event_id}_total")
        if total_pts is None or line is None:
            return None
        try:
            line = float(line)
        except (TypeError, ValueError):
            return None
        if total_pts == line:
            return None
        if side == "Over":
            return 1 if total_pts > line else 0
        if side == "Under":
            return 1 if total_pts < line else 0
    return None


def grade_pending_picks():
    """
    Grade pending paper picks: closing line from line_history, W/L from outcomes.json.
    CLV uses bet-time fair_prob vs closing no-vig Pinnacle (bpr-model convention).
    """
    df = _load_log()
    if df.empty:
        logger.info("Paper log: no entries to grade.")
        return {"graded": 0, "still_pending": 0}

    outcomes = load_outcomes()
    now = datetime.now(timezone.utc).isoformat()
    graded = 0

    for idx, row in df[df["status"] == "pending"].iterrows():
        eid = row.get("event_id")
        market = row.get("market")
        side = row.get("side")
        if not eid or not market or not side:
            continue

        hist = load_history(eid)
        if not hist:
            continue

        snaps = hist.get("snapshots") or []
        close_snap = get_closing_snapshot(
            snaps, max_hours_before_start=CLOSING_SNAP_MAX_HOURS_BEFORE_START
        )
        close_fair, close_price = _closing_prices_for_side(close_snap, market, side)

        bet_fair = row.get("fair_prob")
        clv_pts, clv_pct = compute_clv_vs_close(bet_fair, close_fair)

        is_home = row.get("is_home")
        if pd.isna(is_home):
            is_home = None
        else:
            is_home = bool(int(is_home)) if str(is_home) in ("0", "1", "0.0", "1.0") else bool(is_home)

        line = row.get("line")
        if pd.isna(line):
            line = None

        outcome = _resolve_outcome(eid, market, side, is_home, line, outcomes)
        if outcome is None:
            continue

        dec = american_to_decimal(row.get("american_odds"))
        bet_usd = float(row.get("bet_usd") or 0)
        pnl = bet_usd * (dec - 1.0) if outcome == 1 else -bet_usd

        df.at[idx, "status"] = "graded"
        df.at[idx, "closing_fair_prob"] = close_fair
        df.at[idx, "closing_american_odds"] = close_price
        df.at[idx, "clv_prob_pts"] = clv_pts
        df.at[idx, "clv_pct"] = clv_pct
        df.at[idx, "outcome"] = outcome
        df.at[idx, "pnl"] = round(pnl, 2)
        df.at[idx, "graded_at"] = now
        graded += 1

    _save_log(df)
    pending = int((df["status"] == "pending").sum())
    logger.info("Paper log: graded %d pick(s); %d still pending.", graded, pending)
    return {"graded": graded, "still_pending": pending}


def paper_log_summary():
    """Aggregate metrics for graded paper picks."""
    df = _load_log()
    graded = df[df["status"] == "graded"] if not df.empty else df
    if graded.empty:
        return {}
    total_wagered = graded["bet_usd"].astype(float).sum()
    total_pnl = graded["pnl"].astype(float).sum()
    clv = graded["clv_prob_pts"].dropna().astype(float)
    return {
        "n_logged": int(len(df)),
        "n_graded": int(len(graded)),
        "n_pending": int((df["status"] == "pending").sum()),
        "hit_rate": float(graded["outcome"].astype(float).mean()),
        "roi_pct": float(total_pnl / total_wagered * 100) if total_wagered > 0 else 0.0,
        "total_pnl": float(total_pnl),
        "clv_mean_pts": float(clv.mean()) if len(clv) else None,
        "clv_positive_pct": float((clv > 0).mean() * 100) if len(clv) else None,
    }
