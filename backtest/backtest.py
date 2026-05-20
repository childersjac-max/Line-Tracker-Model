# backtest/backtest.py
# Historical simulation using the SAME filters as live score_all().

import logging
import numpy as np
import pandas as pd
from configs.config import (
    MARKETS,
    MIN_EDGE_TO_BET,
    MIN_MODEL_PROB,
    PROB_SHRINKAGE_ALPHA,
    SPORTS,
)
from data.labeler import label_histories
from data.line_tracker import load_all_histories
from data.results import load_outcomes
from features.movement import build_feature_dataframe
from models.model import load_all_models
from utils.kelly import size_bet
from utils.odds_math import american_to_decimal, compute_clv_vs_close
from utils.scoring import evaluate_bet_row, resolve_fair_prob

logger = logging.getLogger(__name__)


def run_backtest(
    bankroll=10000.0,
    min_edge=None,
    min_model_prob=None,
    prob_shrinkage_alpha=None,
    sport_filter=None,
    market_filter=None,
):
    """
    Simulate bets on labeled history using the same logic as score_all():
    no-vig fair prob, probability shrinkage, MIN_EDGE_TO_BET, MIN_MODEL_PROB,
    sized_prob for Kelly sizing, true CLV vs pin_no_vig_prob_close.
    """
    min_edge = MIN_EDGE_TO_BET if min_edge is None else min_edge
    min_prob = MIN_MODEL_PROB if min_model_prob is None else min_model_prob
    alpha = PROB_SHRINKAGE_ALPHA if prob_shrinkage_alpha is None else prob_shrinkage_alpha

    histories = load_all_histories()
    outcomes = load_outcomes()
    if not histories:
        return pd.DataFrame(), {}

    records = label_histories(histories, outcomes)
    labeled = [r for r in records if r.get("outcome") is not None]
    if not labeled:
        return pd.DataFrame(), {}

    feat_df = build_feature_dataframe(labeled)
    if feat_df.empty:
        return pd.DataFrame(), {}

    for r in labeled:
        mask = (
            (feat_df["event_id"] == r["event_id"])
            & (feat_df["market"] == r["market"])
            & (feat_df["side"] == r["side"])
        )
        feat_df.loc[mask, "outcome"] = r["outcome"]

    bet_records = []
    sports_to_run = [sport_filter] if sport_filter else list(SPORTS.keys())
    markets_to_run = [market_filter] if market_filter else MARKETS

    for sport_key in sports_to_run:
        models = load_all_models(sport_key, markets_to_run)
        if not models:
            continue
        sport_df = feat_df[feat_df["sport_key"] == sport_key].copy()
        if sport_df.empty:
            continue

        for market, model in models.items():
            mdf = sport_df[
                (sport_df["market"] == market) & (sport_df["outcome"].notna())
            ].copy()
            if mdf.empty:
                continue

            probs = model.predict_proba(mdf)
            for i, (_, row) in enumerate(mdf.iterrows()):
                model_prob = float(probs[i])
                ev = evaluate_bet_row(
                    model_prob,
                    row,
                    alpha=alpha,
                    min_edge=min_edge,
                    min_model_prob=min_prob,
                )
                if ev is None:
                    continue

                bet_pct, bet_usd = size_bet(ev["sized_prob"], ev["american_odds"], bankroll)
                if bet_pct == 0:
                    continue

                outcome = int(row.get("outcome"))
                dec = american_to_decimal(ev["american_odds"])
                pnl = bet_usd * (dec - 1.0) if outcome == 1 else -bet_usd

                close_fair = row.get("pin_no_vig_prob_close")
                if close_fair is None or (isinstance(close_fair, float) and np.isnan(close_fair)):
                    close_fair = resolve_fair_prob(row)
                clv_pts, clv_pct = compute_clv_vs_close(ev["fair_prob"], close_fair)

                signals = []
                if row.get("sig_sharp"):
                    signals.append("SHARP_MONEY")
                if row.get("sig_rlm"):
                    signals.append("REVERSE_LINE_MOVEMENT")
                if row.get("sig_fade"):
                    signals.append("PUBLIC_FADE")

                bet_records.append({
                    "sport_key": sport_key,
                    "sport": SPORTS.get(sport_key, sport_key),
                    "market": market,
                    "side": row.get("side"),
                    "event_id": row.get("event_id"),
                    "american_odds": ev["american_odds"],
                    "model_prob": ev["model_prob"],
                    "sized_prob": ev["sized_prob"],
                    "fair_prob": ev["fair_prob"],
                    "closing_fair_prob": close_fair,
                    "edge_pct": ev["edge_pct"],
                    "edge_pct_raw": ev["edge_pct_raw"],
                    "shrinkage_alpha": alpha,
                    "ev_pct": ev["ev_pct"],
                    "bet_pct": bet_pct,
                    "bet_usd": bet_usd,
                    "outcome": outcome,
                    "pnl": pnl,
                    "clv_prob_pts": clv_pts,
                    "clv_pct": clv_pct,
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
    total_pnl = df["pnl"].sum()
    roi = (total_pnl / total_wagered * 100) if total_wagered > 0 else 0
    pnl_s = df["pnl"]
    sharpe = (pnl_s.mean() / pnl_s.std() * np.sqrt(252)) if pnl_s.std() > 0 else 0
    cumulative = np.array([starting_bankroll] + list(starting_bankroll + df["pnl"].cumsum()))
    rolling_max = np.maximum.accumulate(cumulative)
    max_dd = float(((cumulative - rolling_max) / rolling_max * 100).min())
    wins = df.loc[df["pnl"] > 0, "pnl"].sum()
    losses = abs(df.loc[df["pnl"] < 0, "pnl"].sum())
    pf = wins / losses if losses > 0 else float("inf")

    clv = df["clv_prob_pts"].dropna()
    by_market = {}
    for mkt, grp in df.groupby("market"):
        w = grp["bet_usd"].sum()
        clv_m = grp["clv_prob_pts"].dropna()
        by_market[mkt] = {
            "n_bets": int(len(grp)),
            "hit_rate": float(grp["outcome"].mean()),
            "roi_pct": float(grp["pnl"].sum() / w * 100) if w > 0 else 0,
            "clv_mean_pts": float(clv_m.mean()) if len(clv_m) else None,
        }

    by_signal = {}
    for sig in ["SHARP_MONEY", "REVERSE_LINE_MOVEMENT", "PUBLIC_FADE", "CLV_MODEL"]:
        grp = df[df["signals"].str.contains(sig, na=False)]
        if grp.empty:
            continue
        w = grp["bet_usd"].sum()
        by_signal[sig] = {
            "n_bets": int(len(grp)),
            "hit_rate": float(grp["outcome"].mean()),
            "roi_pct": float(grp["pnl"].sum() / w * 100) if w > 0 else 0,
        }

    multi_grp = df[df["n_signals"] >= 2]
    multi_w = multi_grp["bet_usd"].sum()

    return {
        "scoring_parity": "matches score_all (no-vig fair, shrinkage, MIN_EDGE_TO_BET)",
        "shrinkage_alpha": float(df["shrinkage_alpha"].iloc[0]) if "shrinkage_alpha" in df.columns else None,
        "n_bets": int(len(df)),
        "hit_rate": float(df["outcome"].mean()),
        "roi_pct": float(roi),
        "total_pnl": float(total_pnl),
        "total_wagered": float(total_wagered),
        "clv_mean_pts": float(clv.mean()) if len(clv) else None,
        "clv_positive_pct": float((clv > 0).mean() * 100) if len(clv) else None,
        "sharpe": float(sharpe),
        "profit_factor": float(pf),
        "max_drawdown_pct": float(max_dd),
        "by_market": by_market,
        "by_signal": by_signal,
        "multi_signal": {
            "n_bets": int(len(multi_grp)),
            "hit_rate": float(multi_grp["outcome"].mean()) if not multi_grp.empty else 0,
            "roi_pct": float(multi_grp["pnl"].sum() / multi_w * 100) if multi_w > 0 else 0,
        },
    }
