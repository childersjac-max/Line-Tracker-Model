# utils/kelly.py

import numpy as np
from configs.config import KELLY_FRACTION, MAX_BET_PCT, MIN_BET_PCT, MAX_TOTAL_EXPOSURE_PCT

def kelly_fraction(model_prob, american_odds):
    from utils.odds_math import american_to_decimal
    b = american_to_decimal(american_odds) - 1.0
    p, q = model_prob, 1.0 - model_prob
    raw = (b * p - q) / b
    return max(raw, 0.0)

def size_bet(model_prob, american_odds, bankroll):
    raw = kelly_fraction(model_prob, american_odds)
    sized = raw * KELLY_FRACTION
    if sized < MIN_BET_PCT:
        return 0.0, 0.0
    sized = min(sized, MAX_BET_PCT)
    return sized, sized * bankroll

def apply_portfolio_cap(bets, bankroll):
    total = sum(b.get("bet_pct", 0) for b in bets)
    if total > MAX_TOTAL_EXPOSURE_PCT:
        scale = MAX_TOTAL_EXPOSURE_PCT / total
        for b in bets:
            b["bet_pct"] = b["bet_pct"] * scale
            b["bet_usd"] = b["bet_pct"] * bankroll
    return bets

def confidence_label(edge, kelly_pct):
    if edge >= 0.07 and kelly_pct >= 0.025:
        return "HIGH"
    if edge >= 0.04 and kelly_pct >= 0.012:
        return "MEDIUM"
    return "LOW"

def simulate_growth(bets, starting_bankroll=10000.0, n_sims=1000, seed=42):
    from utils.odds_math import american_to_decimal
    rng = np.random.default_rng(seed)
    ruin = starting_bankroll * 0.10
    finals = []
    for _ in range(n_sims):
        br = starting_bankroll
        for bet in bets:
            if br <= ruin:
                break
            stake = br * bet["bet_pct"]
            dec = american_to_decimal(bet["american_odds"])
            br += stake * (dec - 1.0) if rng.random() < bet["model_prob"] else -stake
        finals.append(max(br, ruin))
    a = np.array(finals)
    return {
        "median": float(np.median(a)),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
        "ruin_pct": float(np.mean(a <= ruin) * 100),
        "growth_x": float(np.median(a) / starting_bankroll),
    }
