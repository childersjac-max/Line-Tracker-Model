# models/scorer.py

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from configs.config import MIN_EDGE_TO_BET, MARKETS, SPORTS
from utils.odds_math import ev_pct
from utils.kelly import size_bet, apply_portfolio_cap, confidence_label
from features.movement import build_feature_dataframe
from data.labeler import label_histories
from data.line_tracker import load_all_histories
from models.model import load_all_models


def _hours_to_game(commence_time_str):
    try:
        commence = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        return (commence - now).total_seconds() / 3600.0
    except Exception:
        return None


def score_all(bankroll=10000.0, min_signals=0):
    histories = load_all_histories()
    if not histories:
        print("No line histories found.")
        return pd.DataFrame()

    # ── Filter to only UPCOMING games ────────────────────────────────
    # Drop any event that has already started or completed
    upcoming = []
    for hist in histories:
        commence = hist.get("commence_time", "")
        htg = _hours_to_game(commence)
        if htg is not None and htg > 0:
            upcoming.append(hist)

    if not upcoming:
        print("No upcoming games found in line history.")
        return pd.DataFrame()

    print(f"Scoring {len(upcoming)} upcoming games (filtered from {len(histories)} total)")

    records = label_histories(upcoming, outcomes={})
    if not records:
        return pd.DataFrame()

    feat_df = build_feature_dataframe(records)
    if feat_df.empty:
        return pd.DataFrame()

    # Attach commence_time and team names from histories
    hist_map = {h["event_id"]: h for h in upcoming}

    all_bets = []

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

            probs = model.predict_proba(mdf)

            for i, (_, row) in enumerate(mdf.iterrows()):
                if min_signals > 0 and row.get("n_signals", 0) < min_signals:
                    continue
                model_prob = float(probs[i])
                fair_prob  = row.get("pin_implied_prob", 0.5)
                edge       = model_prob - fair_prob
                best_odds  = row.get("best_pub_price")

                if edge < MIN_EDGE_TO_BET or best_odds is None or np.isnan(best_odds):
                    continue

                bet_pct, bet_usd = size_bet(model_prob, best_odds, bankroll)
                if bet_pct == 0:
                    continue

                event_id = row.get("event_id", "")
                hist     = hist_map.get(event_id, {})
                home     = hist.get("home_team", "")
                away     = hist.get("away_team", "")
                commence = hist.get("commence_time", "")
                htg      = _hours_to_game(commence)

                # Format game time
                game_time = ""
                try:
                    dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
                    # Convert to ET (UTC-4 in summer)
                    from datetime import timedelta
                    dt_et = dt - timedelta(hours=4)
                    game_time = dt_et.strftime("%a %b %-d · %-I:%M %p ET")
                except Exception:
                    game_time = commence[:10] if commence else ""

                signals = []
                if row.get("sig_sharp"): signals.append("SHARP_MONEY")
                if row.get("sig_rlm"):   signals.append("REVERSE_LINE_MOVEMENT")
                if row.get("sig_fade"):  signals.append("PUBLIC_FADE")

                all_bets.append({
                    "event_id":         event_id,
                    "sport":            SPORTS.get(sport_key, sport_key),
                    "sport_key":        sport_key,
                    "market":           market,
                    "side":             row.get("side"),
                    "home_team":        home,
                    "away_team":        away,
                    "matchup":          f"{away} @ {home}",
                    "game_time":        game_time,
                    "hours_to_game":    round(htg, 1) if htg else None,
                    "is_home":          row.get("is_home"),
                    "line":             row.get("line"),
                    "book":             row.get("best_pub_book"),
                    "american_odds":    best_odds,
                    "model_prob":       round(model_prob, 4),
                    "fair_prob":        round(fair_prob, 4),
                    "edge_pct":         round(edge * 100, 2),
                    "ev_pct":           round(ev_pct(model_prob, best_odds), 2),
                    "bet_pct":          round(bet_pct, 4),
                    "bet_usd":          round(bet_usd, 2),
                    "confidence":       confidence_label(edge, bet_pct),
                    "signals":          ", ".join(signals) if signals else "CLV_MODEL",
                    "n_signals":        row.get("n_signals", 0),
                    "pin_move_full":    row.get("pin_move_full", 0),
                    "money_vs_tickets": row.get("money_vs_tickets", 0),
                    "american_odds_display": f"+{int(best_odds)}" if best_odds > 0 else str(int(best_odds)),
                })

    if not all_bets:
        return pd.DataFrame()

    df = pd.DataFrame(all_bets)

    # ── Deduplication: one bet per game per market ────────────────────
    df = (
        df.sort_values("edge_pct", ascending=False)
          .drop_duplicates(subset=["event_id", "market", "side"], keep="first")
    )

    # Totals: keep only best side (Over OR Under) per game
    totals_mask = df["market"] == "totals"
    if totals_mask.any():
        totals_dedup = (
            df[totals_mask]
            .sort_values("edge_pct", ascending=False)
            .drop_duplicates(subset=["event_id", "market"], keep="first")
        )
        df = pd.concat([df[~totals_mask], totals_dedup], ignore_index=True)

    # Spreads: keep only best side per game
    spreads_mask = df["market"] == "spreads"
    if spreads_mask.any():
        spreads_dedup = (
            df[spreads_mask]
            .sort_values("edge_pct", ascending=False)
            .drop_duplicates(subset=["event_id", "market"], keep="first")
        )
        df = pd.concat([df[~spreads_mask], spreads_dedup], ignore_index=True)

    # ── Portfolio cap ─────────────────────────────────────────────────
    bets_list = df.to_dict("records")
    bets_list = apply_portfolio_cap(bets_list, bankroll)
    df = pd.DataFrame(bets_list)

    # ── Sort: confidence first, then by game time, then edge ──────────
    conf_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    df["_co"] = df["confidence"].map(conf_order).fillna(3)
    df["_htg"] = pd.to_numeric(df["hours_to_game"], errors="coerce").fillna(999)
    df = (
        df.sort_values(["_co", "_htg", "edge_pct"], ascending=[True, True, False])
          .drop(columns=["_co", "_htg"])
          .reset_index(drop=True)
    )

    return df
