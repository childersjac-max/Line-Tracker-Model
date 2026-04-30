# models/scorer.py

import logging
import pandas as pd
import numpy as np
from configs.config import (
    MIN_EDGE_TO_BET, MARKETS, SPORTS,
    PROB_SHRINKAGE_ALPHA, MIN_MODEL_PROB,
)
from utils.odds_math import ev_pct
from utils.kelly import size_bet, apply_portfolio_cap, confidence_label
from features.movement import build_feature_dataframe
from data.labeler import label_histories
from data.line_tracker import load_all_histories
from models.model import load_all_models

logger = logging.getLogger(__name__)


def score_all(bankroll=10000.0, min_signals=0, prob_shrinkage_alpha=None, min_model_prob=None):
    """
    Score the current slate.

    prob_shrinkage_alpha: blend factor between model_prob and fair (no-vig) prob.
        sized_prob = α * model_prob + (1 - α) * fair_prob
        Defaults to PROB_SHRINKAGE_ALPHA from config.
    min_model_prob: minimum raw model probability required to bet
        (avoids long-shot mathematical edges). Defaults to MIN_MODEL_PROB.
    """
    alpha    = PROB_SHRINKAGE_ALPHA if prob_shrinkage_alpha is None else prob_shrinkage_alpha
    min_prob = MIN_MODEL_PROB       if min_model_prob       is None else min_model_prob

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

            # ── Tier 1 fix #21: synthetic-training guard ────────────────────
            trained_on = getattr(model, "trained_on", None) or "unknown"
            if trained_on == "synthetic":
                synthetic_models_used.append(f"{sport_key}/{market}")

            probs = model.predict_proba(mdf)

            for i, (_, row) in enumerate(mdf.iterrows()):
                if min_signals > 0 and row.get("n_signals", 0) < min_signals:
                    continue
                model_prob = float(probs[i])

                # ── Tier 1 fix #1: use NO-VIG Pinnacle prob as fair, NEVER
                # the with-vig implied prob.
                fair_prob = row.get("pin_no_vig_prob")
                if fair_prob is None or (isinstance(fair_prob, float) and np.isnan(fair_prob)):
                    # Last-resort fallback (should be rare): with-vig prob.
                    fair_prob = row.get("pin_implied_prob", 0.5)
                fair_prob = float(fair_prob)

                # ── Tier 1 fix #16: shrink model prob toward fair prob
                # before sizing. Hugely reduces drawdown when the model is
                # overconfident, at the cost of slightly fewer bets cleared.
                sized_prob = alpha * model_prob + (1 - alpha) * fair_prob

                # Edge measured on the SHRUNK prob (this is what we're actually
                # betting). Raw edge is reported separately for diagnostics.
                edge_shrunk = sized_prob - fair_prob
                edge_raw    = model_prob  - fair_prob

                best_odds = row.get("best_pub_price")

                if (edge_shrunk < MIN_EDGE_TO_BET
                        or sized_prob  < min_prob
                        or best_odds is None
                        or (isinstance(best_odds, float) and np.isnan(best_odds))):
                    continue

                bet_pct, bet_usd = size_bet(sized_prob, best_odds, bankroll)
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
                    "model_prob":      round(model_prob,  4),
                    "sized_prob":      round(sized_prob,  4),
                    "fair_prob":       round(fair_prob,   4),
                    "edge_pct":        round(edge_shrunk * 100, 2),
                    "edge_pct_raw":    round(edge_raw    * 100, 2),
                    "shrinkage_alpha": alpha,
                    "ev_pct":          round(ev_pct(sized_prob, best_odds), 2),
                    "bet_pct":         round(bet_pct, 4),
                    "bet_usd":         round(bet_usd, 2),
                    "confidence":      confidence_label(edge_shrunk, bet_pct),
                    "signals":         ", ".join(signals) if signals else "CLV_MODEL",
                    "n_signals":       row.get("n_signals", 0),
                    "pin_move_full":    row.get("pin_move_full", 0),
                    "money_vs_tickets": row.get("money_vs_tickets", 0),
                    "clv_signed_train": row.get("clv_signed", 0),
                    "trained_on":       trained_on,    # Tier 1 fix #21 — visible in CSV
                    "american_odds_display": f"+{int(best_odds)}" if best_odds > 0 else str(int(best_odds)),
                })

    if synthetic_models_used:
        logger.warning(
            "⚠️  %d model(s) trained on SYNTHETIC outcomes were used to score this slate: %s. "
            "Recommendations are tagged trained_on=synthetic. Do NOT bet these as real edges.",
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
        totals_df    = df[totals_mask].copy()
        other_df     = df[~totals_mask].copy()
        totals_dedup = (
            totals_df.sort_values("edge_pct", ascending=False)
                     .drop_duplicates(subset=["event_id", "market"], keep="first")
        )
        df = pd.concat([other_df, totals_dedup], ignore_index=True)

    spreads_mask = df["market"] == "spreads"
    if spreads_mask.any():
        spreads_df    = df[spreads_mask].copy()
        other_df      = df[~spreads_mask].copy()
        spreads_dedup = (
            spreads_df.sort_values("edge_pct", ascending=False)
                      .drop_duplicates(subset=["event_id", "market"], keep="first")
        )
        df = pd.concat([other_df, spreads_dedup], ignore_index=True)

    # ── PORTFOLIO CAP ─────────────────────────────────────────────────
    bets_list = df.to_dict("records")
    bets_list = apply_portfolio_cap(bets_list, bankroll)
    df = pd.DataFrame(bets_list)

    # ── SORT ──────────────────────────────────────────────────────────
    conf_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    df["_co"] = df["confidence"].map(conf_order).fillna(3)
    df = (
        df.sort_values(["_co", "edge_pct"], ascending=[True, False])
          .drop(columns=["_co"])
          .reset_index(drop=True)
    )

    return df
