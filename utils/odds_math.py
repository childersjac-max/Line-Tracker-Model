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


def no_vig_prob_for_side(snap, market, side, book="pinnacle"):
    """
    Compute the no-vig (devigged) probability for one side of a 2-way market
    using prices from a single book (Pinnacle by default).

    Returns None if the snapshot does not have both sides at that book.
    This is what should be used as the "fair" probability for edge calculation.
    """
    if not snap:
        return None

    if market == "h2h":
        h2h = snap.get("h2h", {}) or {}
        teams = list(h2h.keys())
        if len(teams) != 2 or side not in teams:
            return None
        pa = h2h[teams[0]].get(book) if isinstance(h2h[teams[0]], dict) else None
        pb = h2h[teams[1]].get(book) if isinstance(h2h[teams[1]], dict) else None
        if pa is None or pb is None:
            return None
        prob_a, prob_b = no_vig_prob(pa, pb)
        return prob_a if side == teams[0] else prob_b

    if market == "totals":
        totals = snap.get("totals", {}) or {}
        ov = totals.get("Over",  {}).get(book, {}) if isinstance(totals.get("Over"),  dict) else {}
        un = totals.get("Under", {}).get(book, {}) if isinstance(totals.get("Under"), dict) else {}
        po = ov.get("price") if isinstance(ov, dict) else None
        pu = un.get("price") if isinstance(un, dict) else None
        if po is None or pu is None:
            return None
        p_over, p_under = no_vig_prob(po, pu)
        return p_over if side == "Over" else p_under

    if market == "spreads":
        spreads = snap.get("spreads", {}) or {}
        teams = list(spreads.keys())
        if len(teams) != 2 or side not in teams:
            return None
        ea = spreads[teams[0]].get(book, {}) if isinstance(spreads[teams[0]], dict) else {}
        eb = spreads[teams[1]].get(book, {}) if isinstance(spreads[teams[1]], dict) else {}
        pa = ea.get("price") if isinstance(ea, dict) else None
        pb = eb.get("price") if isinstance(eb, dict) else None
        if pa is None or pb is None:
            return None
        prob_a, prob_b = no_vig_prob(pa, pb)
        return prob_a if side == teams[0] else prob_b

    return None
