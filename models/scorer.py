# models/scorer.py

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from configs.config import MARKETS, SPORTS, PROB_SHRINKAGE_ALPHA, MIN_MODEL_PROB
from utils.kelly import size_bet, apply_portfolio_cap, confidence_label
from utils.scoring import evaluate_bet_row
from features.movement import build_feature_dataframe
from data.labeler import label_histories
from data.line_tracker import load_all_histories
from models.model import load_all_models

logger = logging.getLogger(__name__)

ODDS_DATA_SOURCE = os.environ.get("ODDS_DATA_SOURCE", "the_odds_api").strip().lower()


def _hours_to_game(commence_time_str):
    try:
        commence = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        return (commence - now).total_seconds() / 3600.0
    except Exception:
        return None


def _fetch_oj_arb_lookup(sport_keys):
    """
    Fetch OddsJam's pre-computed arbitrage feed for all sports.
    Returns dict keyed by (event_id, market, side) -> arb_info.
    Only called when ODDS_DATA_SOURCE=oddsjam.
    Falls back gracefully to {} if the endpoint is unavailable or returns no data.
    """
    lookup = {}
    try:
        from data.sources import get_source
        src = get_source("oddsjam")
        for sport_key in sport_keys:
            arbs = src.fetch_arbitrage_opportunities(sport_key=sport_key)
            for arb in arbs:
                eid    = arb.get("event_id")
                market = arb.get("market")
                margin = float(arb.get("margin_pct") or 0.0)
                legs   = arb.get("legs") or []
                for leg in legs:
                    side = leg.get("side")
                    if not side or not eid or not market:
                        continue
                    partner_legs = [l for l in legs if l.get("side") != side]
                    partner = partner_legs[0] if partner_legs else {}
                    key = (eid, market, side)
                    # Keep highest-margin arb for this (event, market, side)
                    if key not in lookup or margin > lookup[key]["arb_margin_pct"]:
                        lookup[key] = {
                            "is_arb_side":       1,
                            "arb_margin_pct":    round(margin, 3),
                            "arb_book":          leg.get("book"),
                            "arb_partner_book":  partner.get("book"),
                            "arb_partner_price": partner.get("price"),
                            "arb_partner_line":  partner.get("line"),
                            "arb_book_count":    len(legs),
                        }
    except Exception as e:
        logger.warning("OddsJam arb feed fetch skipped (non-fatal): %s", e)
    return lookup


def score_all(bankroll=10000.0, min_signals=0, prob_shrinkage_alpha=None, min_model_prob=None):
    """
    Score the current slate.

    prob_shrinkage_alpha: blend factor between model_prob and fair (no-vig) prob.
        sized_prob = alpha * model_prob + (1 - alpha) * fair_prob
        Defaults to PROB_SHRINKAGE_ALPHA from config.
    min_model_prob: minimum raw model probability required to bet.
        Defaults to MIN_MODEL_PROB from config.
    """
    alpha    = PROB_SHRINKAGE_ALPHA if prob_shrinkage_alpha is None else prob_shrinkage_alpha
    min_prob = MIN_MODEL_PROB       if min_model_prob       is None else min_model_prob

    histories = load_all_histories()
    if not histories:
        logger.info("No line histories found.")
        return pd.DataFrame()

    # ── Filter to only UPCOMING games ────────────────────────────────
    upcoming = []
    for hist in histories:
        htg = _hours_to_game(hist.get("commence_time", ""))
        if htg is not None and htg > 0:
            upcoming.append(hist)

    if not upcoming:
        logger.info("No upcoming games found in line history.")
        return pd.DataFrame()

    logger.info("Scoring %d upcoming games (filtered from %d total)", len(upcoming), len(histories))

    records = label_histories(upcoming, outcomes={})
    if not records:
        return pd.DataFrame()

    feat_df = build_feature_dataframe(records)
    if feat_df.empty:
        return pd.DataFrame()

    # Lookup for game metadata
    hist_map = {h["event_id"]: h for h in upcoming}

    all_bets = []
    synthetic_models_used = []

    for sport_key in SPORTS:
        models = load_all_models(sport_key, MARKETS)
        if not models:
            continue
        sport_df = feat_df[feat_df["sport_key"] == sport_key].copy()
        if sport_df.empty:
            continue

        for market, model in models.items():
            mdf = sport_df[sport_df["market"] == market].copy()
            if mdf.empty:
                continue

            trained_on = getattr(model, "trained_on", None) or "unknown"
            if trained_on == "synthetic":
                synthetic_models_used.append(f"{sport_key}/{market}")

            probs = model.predict_proba(mdf)

            for i, (_, row) in enumerate(mdf.iterrows()):
                if min_signals > 0 and row.get("n_signals", 0) < min_signals:
                    continue
                model_prob = float(probs[i])
                ev = evaluate_bet_row(
                    model_prob, row, alpha=alpha, min_model_prob=min_prob,
                )
                if ev is None:
                    continue

                fair_prob = ev["fair_prob"]
                sized_prob = ev["sized_prob"]
                edge_shrunk = ev["edge_shrunk"]
                edge_raw = ev["edge_raw"]
                best_odds = ev["american_odds"]

                bet_pct, bet_usd = size_bet(sized_prob, best_odds, bankroll)
                if bet_pct == 0:
                    continue

                event_id = row.get("event_id", "")
                hist     = hist_map.get(event_id, {})
                home     = hist.get("home_team", "")
                away     = hist.get("away_team", "")
                commence = hist.get("commence_time", "")
                htg      = _hours_to_game(commence)

                # Format game time in ET
                game_time = ""
                try:
                    dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
                    dt_et = dt - timedelta(hours=4)
                    game_time = dt_et.strftime("%a %b %-d · %-I:%M %p ET")
                except Exception:
                    game_time = commence[:10] if commence else ""

                signals = []
                if row.get("sig_sharp"): signals.append("SHARP_MONEY")
                if row.get("sig_rlm"):   signals.append("REVERSE_LINE_MOVEMENT")
                if row.get("sig_fade"):  signals.append("PUBLIC_FADE")
                if row.get("is_arb_side"):
                    margin_lbl = float(row.get("arb_margin_pct") or 0.0)
                    signals.append(f"ARBITRAGE({margin_lbl:.2f}%)")

                all_bets.append({
                    "event_id":    event_id,
                    "sport":       SPORTS.get(sport_key, sport_key),
                    "sport_key":   sport_key,
                    "market":      market,
                    "side":        row.get("side"),
                    "is_home":     row.get("is_home"),
                    "line":        row.get("line"),
                    "book":        row.get("best_pub_book"),
                    "home_team":   home,
                    "away_team":   away,
                    "matchup":     f"{away} @ {home}" if away and home else "",
                    "game_time":   game_time,
                    "hours_to_game": round(htg, 1) if htg else None,
                    "american_odds":       best_odds,
                    "model_prob":          round(model_prob,  4),
                    "sized_prob":          round(sized_prob,  4),
                    "fair_prob":           round(fair_prob,   4),
                    "edge_pct":            ev["edge_pct"],
                    "edge_pct_raw":        ev["edge_pct_raw"],
                    "shrinkage_alpha":     ev["shrinkage_alpha"],
                    "ev_pct":              ev["ev_pct"],
                    "bet_pct":             round(bet_pct, 4),
                    "bet_usd":             round(bet_usd, 2),
                    "confidence":          confidence_label(edge_shrunk, bet_pct),
                    "signals":             ", ".join(signals) if signals else "CLV_MODEL",
                    "n_signals":           row.get("n_signals", 0),
                    "pin_move_full":       row.get("pin_move_full", 0),
                    "money_vs_tickets":    row.get("money_vs_tickets", 0),
                    "clv_signed_train":    row.get("clv_signed", 0),
                    "trained_on":          trained_on,
                    "american_odds_display": f"+{int(best_odds)}" if best_odds > 0 else str(int(best_odds)),
                    # Arbitrage columns (local detection — overridden below by OddsJam feed)
                    "is_arb_side":       int(row.get("is_arb_side", 0) or 0),
                    "arb_margin_pct":    round(float(row.get("arb_margin_pct", 0) or 0), 3),
                    "arb_book_count":    int(row.get("arb_book_count", 0) or 0),
                    "arb_book":          row.get("arb_book"),
                    "arb_partner_book":  row.get("arb_partner_book"),
                    "arb_partner_price": row.get("arb_partner_price"),
                    "arb_partner_line":  row.get("arb_partner_line"),
                })

    if synthetic_models_used:
        logger.warning(
            "WARNING: %d model(s) trained on SYNTHETIC outcomes: %s. "
            "Tagged trained_on=synthetic.",
            len(synthetic_models_used), ", ".join(sorted(set(synthetic_models_used))),
        )

    if not all_bets:
        return pd.DataFrame()

    df = pd.DataFrame(all_bets)

    # ── DEDUPLICATION ─────────────────────────────────────────────────
    df = (
        df.sort_values("edge_pct", ascending=False)
          .drop_duplicates(subset=["event_id", "market", "side"], keep="first")
    )

    totals_mask = df["market"] == "totals"
    if totals_mask.any():
        totals_dedup = (
            df[totals_mask].sort_values("edge_pct", ascending=False)
                           .drop_duplicates(subset=["event_id", "market"], keep="first")
        )
        df = pd.concat([df[~totals_mask], totals_dedup], ignore_index=True)

    spreads_mask = df["market"] == "spreads"
    if spreads_mask.any():
        spreads_dedup = (
            df[spreads_mask].sort_values("edge_pct", ascending=False)
                            .drop_duplicates(subset=["event_id", "market"], keep="first")
        )
        df = pd.concat([df[~spreads_mask], spreads_dedup], ignore_index=True)

    # ── PORTFOLIO CAP ─────────────────────────────────────────────────
    bets_list = df.to_dict("records")
    bets_list = apply_portfolio_cap(bets_list, bankroll)
    df = pd.DataFrame(bets_list)

    # ── ODDJAM ARB ENRICHMENT ─────────────────────────────────────────
    # When using OddsJam, fetch their pre-computed arb feed and overlay
    # it onto the locally-detected arbs.  OddsJam's results take priority
    # (their feed is pre-screened and more accurate); local detection
    # remains the fallback for any bet not covered by the OddsJam feed.
    if ODDS_DATA_SOURCE == "oddsjam":
        logger.info("Fetching OddsJam arbitrage feed...")
        oj_lookup = _fetch_oj_arb_lookup(list(SPORTS.keys()))
        if oj_lookup:
            logger.info("OddsJam arb feed: %d arb leg(s) found", len(oj_lookup))
            def _enrich_row(row):
                key = (row.get("event_id"), row.get("market"), row.get("side"))
                oj  = oj_lookup.get(key)
                if oj:
                    for k, v in oj.items():
                        row[k] = v
                    # Re-stamp signal with OddsJam's margin
                    sigs = [s for s in (row.get("signals") or "").split(", ")
                            if s and not s.startswith("ARBITRAGE")]
                    sigs.append(f"ARBITRAGE({float(oj['arb_margin_pct']):.2f}%)")
                    row["signals"] = ", ".join(sigs)
                return row
            df = df.apply(_enrich_row, axis=1)
        else:
            logger.info("OddsJam arb feed: no active arbitrage opportunities found")

    # ── INJURY ANNOTATION ─────────────────────────────────────────────
    try:
        from features.injury import annotate_slate_with_injuries, apply_injury_signals
        df = annotate_slate_with_injuries(df)
        df = apply_injury_signals(df)
    except Exception as e:
        logger.warning("  [injury] Annotation skipped (non-fatal): %s", e)

    # ── SORT: confidence → hours to game → edge ───────────────────────
    conf_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    df["_co"]  = df["confidence"].map(conf_order).fillna(3)
    df["_htg"] = pd.to_numeric(df.get("hours_to_game"), errors="coerce").fillna(999)
    df = (
        df.sort_values(["_co", "_htg", "edge_pct"], ascending=[True, True, False])
          .drop(columns=["_co", "_htg"])
          .reset_index(drop=True)
    )

    return df
