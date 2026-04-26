# utils/odds_math.py

def american_to_decimal(american):
    if american >= 100:
        return (american / 100) + 1.0
    return (100 / abs(american)) + 1.0

def american_to_implied_prob(american):
    if american >= 100:
        return 100.0 / (american + 100.0)
    return abs(american) / (abs(american) + 100.0)

def implied_prob_to_american(prob):
    if prob <= 0 or prob >= 1:
        return None
    if prob < 0.5:
        return round((100.0 / prob) - 100.0, 1)
    return round(-100.0 * prob / (1.0 - prob), 1)

def no_vig_prob(american_a, american_b):
    pa = american_to_implied_prob(american_a)
    pb = american_to_implied_prob(american_b)
    total = pa + pb
    return pa / total, pb / total

def clv_edge(model_prob, no_vig_market_prob):
    return model_prob - no_vig_market_prob

def ev_pct(model_prob, american_odds):
    dec = american_to_decimal(american_odds)
    return (model_prob * (dec - 1.0) - (1.0 - model_prob)) * 100.0

def line_move_in_prob(old_american, new_american):
    return american_to_implied_prob(new_american) - american_to_implied_prob(old_american)
