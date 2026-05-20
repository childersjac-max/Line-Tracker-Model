# utils/scoring.py
# Shared bet-decision math used by live scoring (score_all) and backtest.
# Matches bpr-model conventions: same-book de-vig fair prob, half-Kelly stake input.

import numpy as np
from configs.config import (
    MIN_EDGE_TO_BET,
    MIN_MODEL_PROB,
    PROB_SHRINKAGE_ALPHA,
)
from utils.odds_math import ev_pct


def resolve_fair_prob(row):
    """No-vig Pinnacle fair prob; fall back to with-vig implied."""
    fair = row.get("pin_no_vig_prob")
    if fair is None or (isinstance(fair, float) and np.isnan(fair)):
        fair = row.get("pin_implied_prob", 0.5)
    return float(fair)


def compute_sized_prob(model_prob, fair_prob, alpha=None):
    alpha = PROB_SHRINKAGE_ALPHA if alpha is None else alpha
    return alpha * float(model_prob) + (1.0 - alpha) * float(fair_prob)


def evaluate_bet_row(
    model_prob,
    row,
    *,
    alpha=None,
    min_edge=None,
    min_model_prob=None,
):
    """
    Apply the same filters as score_all() for one feature row.
    Returns dict with sized_prob, edges, ev_pct, pass_filter — or None if no bet.
    """
    min_edge = MIN_EDGE_TO_BET if min_edge is None else min_edge
    min_prob = MIN_MODEL_PROB if min_model_prob is None else min_model_prob
    alpha = PROB_SHRINKAGE_ALPHA if alpha is None else alpha

    fair_prob = resolve_fair_prob(row)
    sized_prob = compute_sized_prob(model_prob, fair_prob, alpha)
    edge_shrunk = sized_prob - fair_prob
    edge_raw = float(model_prob) - fair_prob

    best_odds = row.get("best_pub_price")
    if best_odds is None or (isinstance(best_odds, float) and np.isnan(best_odds)):
        return None
    if edge_shrunk < min_edge or sized_prob < min_prob:
        return None

    return {
        "model_prob": float(model_prob),
        "fair_prob": fair_prob,
        "sized_prob": sized_prob,
        "edge_shrunk": edge_shrunk,
        "edge_raw": edge_raw,
        "edge_pct": round(edge_shrunk * 100, 2),
        "edge_pct_raw": round(edge_raw * 100, 2),
        "shrinkage_alpha": alpha,
        "ev_pct": round(ev_pct(sized_prob, best_odds), 2),
        "american_odds": best_odds,
        "best_pub_book": row.get("best_pub_book"),
    }
