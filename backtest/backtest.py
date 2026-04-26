# backtest/backtest.py

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from configs.config import SPORTS, MARKETS
from data.labeler import label_histories
from data.line_tracker import load_all_histories
from data.results import load_outcomes
from features.movement import build_feature_dataframe
from models.model import load_all_models
from utils.odds_math import american_to_decimal
from utils.kelly import size_bet

logger = logging.getLogger(__name__)


def run_backtest(bankroll=10000.0, min_edge=0.025, sport_filter=None, market_filter=None):
    histories = load_all_histories()
    outcomes  = load_outcomes()
    if not histories:
        return pd.DataFrame(), {}
    records = label_histories(histories, outcomes)
    labeled = [r for r in records if r.get("outcome") is not None]
    if not labeled:
        return pd.DataFrame(), {}
    feat_df = build_feature_dataframe(labeled)
    if feat_df.empty:
        return pd.DataFrame(), {}
    outcome_map = {r["event_id"] + "_" + r["market"] + "_" + str(r["side"]): r["outcome"] for r in labeled if r.get("outcome") is not None}
    for r in labeled:
        key = r["event_id"] + "_" + r["market"] + "_" + str(r["side"])
        mask = (feat_df["event_id"] == r["event_id"]) & (feat_df["market"] == r["market"]) & (feat_df["side"] == r["side"])
        feat_df.loc[mask, "outcome"] = r["outcome"]

    bet_records = []
    sports_to_run  = [sport_filter] if sport_filter else list(SPORTS.keys())
    markets_to_run = [market_filter] if market_filter else MARKETS

    for sport_key in sports_to_run:
        models = load_all_models(sport_key, markets_to_run)
        if not models:
            continue
        sport_df = feat_df[feat_df["sport_key"] == sport_key].copy()
        if sport_df.empty:
            continue
        for market, model in models.items():
            mdf = sport_df[(sport_df["market"] == market) & (sport_df["outcome"].notna())].copy()
            if mdf.empty:
                continue
            probs = model.predict_proba(mdf)
            for i, (_, row) in enumerate(mdf.iterrows()):
                model_prob = float(probs[i])
                fair_prob  = row.get("pin_implied_prob", 0.5)
                edge = model_prob - fair_prob
                if edge < min_edge:
                    continue
                best_odds = row.get("best_pub_price")
                if best_odds is None or np.isnan(best_odds):
                    continue
                bet_pct, bet_usd = size_bet(model_prob, best_odds, bankroll)
                if bet_pct == 0:
                    continue
                outcome = row.get("outcome")
                if outcome is None:
                    continue
                dec = american_to_decimal(best_odds)
                pnl = bet_usd * (dec - 1.0) if outcome == 1 else -bet_usd
                signals = []
                if row.get("sig_sharp"): signals.append("SHARP_MONEY")
                if row.get("sig_rlm"):   signals.append("REVERSE_LINE_MOVEMENT")
                if row.get("sig_fade"):  signals.append("PUBLIC_FADE")
                bet_records.append({
                    "sport_key": sport_key, "sport": SPORTS.get(sport_key, sport_key),
                    "market": market, "side": row.get("side"),
                    "american_odds": best_odds, "model_prob": model_prob,
                    "fair_prob": fair_prob, "edge": edge,
                    "bet_pct": bet_pct, "bet_usd": bet_usd,
                    "outcome": outcome, "pnl": pnl, "clv": edge,
                    "signals": ", ".join(signals) if signals else "CLV_MODEL",
                    "sig_sharp": row.get("sig_sharp", 0),
                    "sig_rlm": row.get("sig_rlm", 0),
                    "sig_fade": row.get("sig_fade", 0),
                    "n_signals": row.get("n_signals", 0),
                })

    if not bet_records:
        return pd.DataFrame(), {}
    df = pd.DataFrame(bet_records)
    return df, compute_metrics(df, bankroll)


def compute_metrics(df, starting_bankroll=10000.0):
    if df.empty:
        return {}
    total_wagered = df["bet_usd"].sum()
    total_pnl     = df["pnl"].sum()
    roi           = (total_pnl / total_wagered * 100) if total_wagered > 0 else 0
    pnl_s         = df["pnl"]
    sharpe        = (pnl_s.mean() / pnl_s.std() * np.sqrt(252)) if pnl_s.std() > 0 else 0
    cumulative    = np.array([starting_bankroll] + list(starting_bankroll + df["pnl"].cumsum()))
    rolling_max   = np.maximum.accumulate(cumulative)
    max_dd        = float(((cumulative - rolling_max) / rolling_max * 100).min())
    wins   = df.loc[df["pnl"] > 0, "pnl"].sum()
    losses = abs(df.loc[df["pnl"] < 0, "pnl"].sum())
    pf     = wins / losses if losses > 0 else float("inf")
    by_market = {}
    for mkt, grp in df.groupby("market"):
        w = grp["bet_usd"].sum()
        by_market[mkt] = {"n_bets": int(len(grp)), "hit_rate": float(grp["outcome"].mean()),
                          "roi_pct": float(grp["pnl"].sum() / w * 100) if w > 0 else 0,
                          "clv_mean": float(grp["clv"].mean())}
    by_signal = {}
    for sig in ["SHARP_MONEY", "REVERSE_LINE_MOVEMENT", "PUBLIC_FADE", "CLV_MODEL"]:
        grp = df[df["signals"].str.contains(sig, na=False)]
        if grp.empty:
            continue
        w = grp["bet_usd"].sum()
        by_signal[sig] = {"n_bets": int(len(grp)), "hit_rate": float(grp["outcome"].mean()),
                          "roi_pct": float(grp["pnl"].sum() / w * 100) if w > 0 else 0}
    multi_grp = df[df["n_signals"] >= 2]
    multi_w   = multi_grp["bet_usd"].sum()
    return {
        "n_bets": int(len(df)), "hit_rate": float(df["outcome"].mean()),
        "roi_pct": float(roi), "total_pnl": float(total_pnl),
        "total_wagered": float(total_wagered), "clv_mean": float(df["clv"].mean()),
        "clv_positive_pct": float((df["clv"] > 0).mean() * 100),
        "sharpe": float(sharpe), "profit_factor": float(pf),
        "max_drawdown_pct": float(max_dd), "by_market": by_market,
        "by_signal": by_signal,
        "multi_signal": {"n_bets": int(len(multi_grp)),
                         "hit_rate": float(multi_grp["outcome"].mean()) if not multi_grp.empty else 0,
                         "roi_pct": float(multi_grp["pnl"].sum() / multi_w * 100) if multi_w > 0 else 0},
        "monthly_roi": {},
    }
