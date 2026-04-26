# models/scorer.py

import pandas as pd
import numpy as np
from configs.config import MIN_EDGE_TO_BET, MARKETS, SPORTS
from utils.odds_math import ev_pct
from utils.kelly import size_bet, apply_portfolio_cap, confidence_label
from features.movement import build_feature_dataframe
from data.labeler import label_histories
from data.line_tracker import load_all_histories
from models.model import load_all_models


def score_all(bankroll=10000.0, min_signals=0):
    histories = load_all_histories()
    if not histories:
        print("No line histories found.")
        return pd.DataFrame()

    records = label_histories(histories, outcomes={})
    if not records:
        return pd.DataFrame()

    feat_df = build_feature_dataframe(records)
    if feat_df.empty:
        return pd.DataFrame()

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

                signals = []
                if row.get("sig_sharp"): signals.append("SHARP_MONEY")
                if row.get("sig_rlm"):   signals.append("REVERSE_LINE_MOVEMENT")
                if row.get("sig_fade"):  signals.append("PUBLIC_FADE")

                all_bets.append({
                    "event_id":    row.get("event_id"),
                    "sport":       SPORTS.get(sport_key, sport_key),
                    "sport_key":   sport_key,
                    "market":      market,
                    "side":        row.get("side"),
                    "is_home":     row.get("is_home"),
                    "line":        row.get("line"),
                    "book":        row.get("best_pub_book"),
                    "american_odds": best_odds,
                    "model_prob":  round(model_prob, 4),
                    "fair_prob":   round(fair_prob, 4),
                    "edge_pct":    round(edge * 100, 2),
                    "ev_pct":      round(ev_pct(model_prob, best_odds), 2),
                    "bet_pct":     round(bet_pct, 4),
                    "bet_usd":     round(bet_usd, 2),
                    "confidence":  confidence_label(edge, bet_pct),
                    "signals":     ", ".join(signals) if signals else "CLV_MODEL",
                    "n_signals":   row.get("n_signals", 0),
                    "pin_move_full":      row.get("pin_move_full", 0),
                    "money_vs_tickets":   row.get("money_vs_tickets", 0),
                    "american_odds_display": f"+{int(best_odds)}" if best_odds > 0 else str(int(best_odds)),
                })

    if not all_bets:
        return pd.DataFrame()

    all_bets = apply_portfolio_cap(all_bets, bankroll)
    df = pd.DataFrame(all_bets)
    conf_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    df["_co"] = df["confidence"].map(conf_order).fillna(3)
    return df.sort_values(["_co", "edge_pct"], ascending=[True, False]).drop(columns=["_co"]).reset_index(drop=True)
