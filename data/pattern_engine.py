# data/pattern_engine.py
# =====================================================================
# PATTERN DISCOVERY ENGINE
# =====================================================================
# Analyzes all historical line history + outcomes to find combinations
# of ticket%, money%, line value, and movement that signal EV advantages.
#
# For each (sport, market) it discovers patterns like:
#   "When spread is +5.5 to +7.5 AND tickets shift from 40% to 60%
#    AND money stays below 45% → EV = +14.2% (23 occurrences)"
#
# Output: patterns.json saved to pipeline_output/
# =====================================================================

import json
import itertools
import numpy as np
from pathlib import Path
from collections import defaultdict

from data.line_tracker import load_all_histories
from data.results import load_outcomes

OUTPUT_FILE = Path("./pipeline_output/patterns.json")

# ── Bucketing definitions ─────────────────────────────────────────────

TICKET_BUCKETS = [
    (0,  30,  "0-30%"),
    (30, 45,  "30-45%"),
    (45, 55,  "45-55%"),
    (55, 65,  "55-65%"),
    (65, 75,  "65-75%"),
    (75, 100, "75%+"),
]

MONEY_BUCKETS = [
    (0,  30,  "0-30%"),
    (30, 45,  "30-45%"),
    (45, 55,  "45-55%"),
    (55, 65,  "55-65%"),
    (65, 75,  "65-75%"),
    (75, 100, "75%+"),
]

SPREAD_BUCKETS = [
    (-99, -7.5, "Heavy fav (< -7.5)"),
    (-7.5, -3.5, "Fav (-7.5 to -3.5)"),
    (-3.5, -0.5, "Slight fav (-3.5 to -0.5)"),
    (-0.5, 0.5,  "Pick em"),
    (0.5, 3.5,   "Slight dog (+0.5 to +3.5)"),
    (3.5, 7.5,   "Dog (+3.5 to +7.5)"),
    (7.5, 99,    "Big dog (+7.5)"),
]

TOTAL_BUCKETS = [
    (0,   38,  "Under 38"),
    (38,  42,  "38-42"),
    (42,  46,  "42-46"),
    (46,  50,  "46-50"),
    (50,  54,  "50-54"),
    (54,  58,  "54-58"),
    (58,  999, "58+"),
]

ML_BUCKETS = [
    (-99999, -300, "Heavy fav (< -300)"),
    (-300,  -150,  "Fav (-300 to -150)"),
    (-150,  -110,  "Slight fav (-150 to -110)"),
    (-110,   110,  "Pick em"),
    (110,    200,  "Slight dog (+110 to +200)"),
    (200,    400,  "Dog (+200 to +400)"),
    (400,  99999,  "Big dog (+400)"),
]

MOVE_BUCKETS = [
    (-99, -5,  "Big move away (< -5pts)"),
    (-5,  -2,  "Move away (-5 to -2pts)"),
    (-2,   2,  "Flat (-2 to +2pts)"),
    (2,    5,  "Move toward (+2 to +5pts)"),
    (5,   99,  "Big move toward (+5pts)"),
]


def bucket(value, buckets):
    """Return the label for whichever bucket value falls in."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    for lo, hi, label in buckets:
        if lo <= value < hi:
            return label
    return buckets[-1][2]


def american_to_ev(model_prob, american_odds):
    """EV% given model win probability and american odds."""
    if american_odds >= 100:
        dec = (american_odds / 100) + 1.0
    else:
        dec = (100 / abs(american_odds)) + 1.0
    return (model_prob * (dec - 1.0) - (1.0 - model_prob)) * 100.0


def implied_prob(american_odds):
    if american_odds >= 100:
        return 100.0 / (american_odds + 100.0)
    return abs(american_odds) / (abs(american_odds) + 100.0)


def no_vig_prob(american_a, american_b):
    pa = implied_prob(american_a)
    pb = implied_prob(american_b)
    total = pa + pb
    return (pa / total, pb / total) if total > 0 else (0.5, 0.5)


# ── Feature extraction per snapshot ──────────────────────────────────

def extract_snapshot_features(hist, outcome):
    """
    Extract pattern features from a line history record.
    Returns list of feature dicts (one per market side).
    """
    snaps = hist.get("snapshots", [])
    if len(snaps) < 2:
        return []

    sport_key  = hist.get("sport_key", "")
    home_team  = hist.get("home_team", "")
    away_team  = hist.get("away_team", "")
    event_id   = hist.get("event_id", "")

    opener = snaps[0]
    latest = snaps[-1]

    # Splits from latest snapshot
    splits = latest.get("splits", {})

    def get_splits(market_key):
        """Get splits for moneyline/spread/total."""
        m = splits.get(market_key, {})
        return {
            "sharp_tickets": m.get("sharp_tickets_pct"),
            "sharp_money":   m.get("sharp_money_pct"),
            "pub_tickets":   m.get("public_tickets_pct"),
            "pub_money":     m.get("public_money_pct"),
            "magnitude":     m.get("magnitude_pts"),
            "money_vs_tick": m.get("money_vs_tickets"),
            "sharp_side":    m.get("sharp_side"),
        }

    def price_move(market, side, book="pinnacle"):
        """Change in american odds from opener to latest."""
        def get_price(snap):
            if market == "h2h":
                return snap.get("h2h", {}).get(side, {}).get(book)
            elif market == "spreads":
                e = snap.get("spreads", {}).get(side, {}).get(book, {})
                return e.get("price") if isinstance(e, dict) else None
            elif market == "totals":
                e = snap.get("totals", {}).get(side, {}).get(book, {})
                return e.get("price") if isinstance(e, dict) else None
        p_open = get_price(opener)
        p_late = get_price(latest)
        if p_open is None or p_late is None:
            return None
        return p_late - p_open

    def get_line(market, side, book="pinnacle"):
        """Get current line value (spread or total)."""
        if market == "spreads":
            e = latest.get("spreads", {}).get(side, {}).get(book, {})
            return e.get("line") if isinstance(e, dict) else None
        elif market == "totals":
            e = latest.get("totals", {}).get(side, {}).get(book, {})
            return e.get("line") if isinstance(e, dict) else None
        return None

    def get_ml(side):
        """Get current moneyline."""
        return latest.get("h2h", {}).get(side, {}).get("pinnacle") or \
               latest.get("h2h", {}).get(side, {}).get("draftkings")

    def get_best_pub_price(market, side):
        pub_books = ["draftkings", "fanduel", "betmgm", "bovada", "williamhill_us", "bet365"]
        if market == "h2h":
            prices = [latest.get("h2h", {}).get(side, {}).get(b) for b in pub_books]
        elif market == "spreads":
            prices = [latest.get("spreads", {}).get(side, {}).get(b, {}).get("price") for b in pub_books]
        elif market == "totals":
            prices = [latest.get("totals", {}).get(side, {}).get(b, {}).get("price") for b in pub_books]
        else:
            return None
        prices = [p for p in prices if p is not None]
        return max(prices) if prices else None

    records = []

    # ── Moneyline ─────────────────────────────────────────────────────
    h2h = latest.get("h2h", {})
    teams = list(h2h.keys())
    if len(teams) == 2:
        ml_splits = get_splits("moneyline")
        for team in teams:
            is_home = (team == home_team)
            other   = [t for t in teams if t != team][0]
            ml      = get_ml(team)
            ml_other = get_ml(other)
            move    = price_move("h2h", team)
            best_pub = get_best_pub_price("h2h", team)

            if ml and ml_other:
                fair, _ = no_vig_prob(ml, ml_other)
            else:
                fair = 0.5

            outcome_val = outcome.get(f"{event_id}_home_ml" if is_home else f"{event_id}_away_ml")

            records.append({
                "sport_key":   sport_key,
                "market":      "moneyline",
                "side":        "home" if is_home else "away",
                "team":        team,
                "line_value":  ml,
                "line_bucket": bucket(ml, ML_BUCKETS),
                "line_move":   move,
                "move_bucket": bucket(move, MOVE_BUCKETS),
                "fair_prob":   fair,
                "best_pub":    best_pub,
                "sharp_tickets": ml_splits.get("sharp_tickets"),
                "sharp_money":   ml_splits.get("sharp_money"),
                "pub_tickets":   ml_splits.get("pub_tickets"),
                "pub_money":     ml_splits.get("pub_money"),
                "magnitude":     ml_splits.get("magnitude"),
                "money_vs_tick": ml_splits.get("money_vs_tick"),
                "sharp_side":    ml_splits.get("sharp_side"),
                "is_sharp_side": (ml_splits.get("sharp_side") == ("home" if is_home else "away")),
                "outcome":     outcome_val,
                "n_snaps":     len(snaps),
            })

    # ── Spreads ───────────────────────────────────────────────────────
    spreads = latest.get("spreads", {})
    spread_teams = list(spreads.keys())
    if len(spread_teams) == 2:
        sp_splits = get_splits("spread")
        for team in spread_teams:
            is_home = (team == home_team)
            other   = [t for t in spread_teams if t != team][0]
            line    = get_line("spreads", team)
            move    = price_move("spreads", team)
            best_pub = get_best_pub_price("spreads", team)

            pin_price = spreads.get(team, {}).get("pinnacle", {})
            pin_other = spreads.get(other, {}).get("pinnacle", {})
            fair = 0.5
            if isinstance(pin_price, dict) and isinstance(pin_other, dict):
                pp = pin_price.get("price")
                po = pin_other.get("price")
                if pp and po:
                    fair, _ = no_vig_prob(pp, po)

            hs = outcome.get(f"{event_id}_home_score")
            as_ = outcome.get(f"{event_id}_away_score")
            spread_outcome = None
            if hs is not None and as_ is not None and line is not None:
                margin = (hs - as_) if is_home else (as_ - hs)
                spread_outcome = 1 if margin + line > 0 else 0

            records.append({
                "sport_key":   sport_key,
                "market":      "spread",
                "side":        "home" if is_home else "away",
                "team":        team,
                "line_value":  line,
                "line_bucket": bucket(line, SPREAD_BUCKETS),
                "line_move":   move,
                "move_bucket": bucket(move, MOVE_BUCKETS),
                "fair_prob":   fair,
                "best_pub":    best_pub,
                "sharp_tickets": sp_splits.get("sharp_tickets"),
                "sharp_money":   sp_splits.get("sharp_money"),
                "pub_tickets":   sp_splits.get("pub_tickets"),
                "pub_money":     sp_splits.get("pub_money"),
                "magnitude":     sp_splits.get("magnitude"),
                "money_vs_tick": sp_splits.get("money_vs_tick"),
                "sharp_side":    sp_splits.get("sharp_side"),
                "is_sharp_side": (sp_splits.get("sharp_side") == ("home" if is_home else "away")),
                "outcome":     spread_outcome,
                "n_snaps":     len(snaps),
            })

    # ── Totals ────────────────────────────────────────────────────────
    totals = latest.get("totals", {})
    if "Over" in totals or "Under" in totals:
        tot_splits = get_splits("total")
        for side in ["Over", "Under"]:
            other = "Under" if side == "Over" else "Over"
            line  = get_line("totals", side)
            move  = price_move("totals", side)
            best_pub = get_best_pub_price("totals", side)

            pin_ov = totals.get("Over", {}).get("pinnacle", {})
            pin_un = totals.get("Under", {}).get("pinnacle", {})
            fair = 0.5
            if isinstance(pin_ov, dict) and isinstance(pin_un, dict):
                po = pin_ov.get("price")
                pu = pin_un.get("price")
                if po and pu:
                    p_over, _ = no_vig_prob(po, pu)
                    fair = p_over if side == "Over" else 1 - p_over

            total_pts = outcome.get(f"{event_id}_total")
            tot_outcome = None
            if total_pts is not None and line is not None:
                tot_outcome = 1 if (side == "Over" and total_pts > line) or \
                                   (side == "Under" and total_pts < line) else 0

            records.append({
                "sport_key":   sport_key,
                "market":      "total",
                "side":        side.lower(),
                "team":        side,
                "line_value":  line,
                "line_bucket": bucket(line, TOTAL_BUCKETS),
                "line_move":   move,
                "move_bucket": bucket(move, MOVE_BUCKETS),
                "fair_prob":   fair,
                "best_pub":    best_pub,
                "sharp_tickets": tot_splits.get("sharp_tickets"),
                "sharp_money":   tot_splits.get("sharp_money"),
                "pub_tickets":   tot_splits.get("pub_tickets"),
                "pub_money":     tot_splits.get("pub_money"),
                "magnitude":     tot_splits.get("magnitude"),
                "money_vs_tick": tot_splits.get("money_vs_tick"),
                "sharp_side":    tot_splits.get("sharp_side"),
                "is_sharp_side": (tot_splits.get("sharp_side") == side.lower()),
                "outcome":     tot_outcome,
                "n_snaps":     len(snaps),
            })

    return records


# ── Pattern aggregation ───────────────────────────────────────────────

def compute_ev(records):
    """
    Given a list of records with outcomes, compute:
    hit_rate, ev_pct, sample_size, avg_fair_prob
    """
    labeled = [r for r in records if r.get("outcome") is not None]
    if len(labeled) < 3:
        return None
    hits = sum(r["outcome"] for r in labeled)
    n    = len(labeled)
    hit_rate = hits / n

    # EV using best pub price when available
    evs = []
    for r in labeled:
        pub = r.get("best_pub")
        if pub:
            evs.append(american_to_ev(hit_rate, pub))
    ev = float(np.mean(evs)) if evs else 0.0

    fair_probs = [r["fair_prob"] for r in labeled if r.get("fair_prob")]
    avg_fair   = float(np.mean(fair_probs)) if fair_probs else 0.5

    return {
        "hit_rate":     round(hit_rate * 100, 1),
        "ev_pct":       round(ev, 2),
        "n":            n,
        "n_wins":       int(hits),
        "avg_fair_prob": round(avg_fair * 100, 1),
    }


def discover_patterns(min_samples=5):
    """
    Main entry point. Discovers all profitable/unprofitable patterns
    across every combination of ticket%, money%, line, and movement.

    Returns dict of patterns grouped by sport+market.
    """
    histories = load_all_histories()
    outcomes  = load_outcomes()

    if not histories:
        return {}

    # Extract all features
    all_records = []
    for hist in histories:
        feats = extract_snapshot_features(hist, outcomes)
        all_records.extend(feats)

    print(f"  Total records extracted: {len(all_records)}")

    # Group by sport + market
    grouped = defaultdict(list)
    for r in all_records:
        key = f"{r['sport_key']}|{r['market']}"
        grouped[key].append(r)

    output = {}

    for group_key, records in grouped.items():
        sport_key, market = group_key.split("|")
        patterns = []

        # ── Pattern 1: Ticket% bucket only ───────────────────────────
        tick_groups = defaultdict(list)
        for r in records:
            tb = bucket(r.get("pub_tickets"), TICKET_BUCKETS)
            if tb:
                tick_groups[tb].append(r)
        for tb, recs in tick_groups.items():
            stats = compute_ev(recs)
            if stats and stats["n"] >= min_samples:
                patterns.append({
                    "type":        "ticket_pct",
                    "description": f"Public tickets: {tb}",
                    "filters":     {"ticket_bucket": tb},
                    **stats,
                })

        # ── Pattern 2: Money% bucket only ────────────────────────────
        money_groups = defaultdict(list)
        for r in records:
            mb = bucket(r.get("pub_money"), MONEY_BUCKETS)
            if mb:
                money_groups[mb].append(r)
        for mb, recs in money_groups.items():
            stats = compute_ev(recs)
            if stats and stats["n"] >= min_samples:
                patterns.append({
                    "type":        "money_pct",
                    "description": f"Public money: {mb}",
                    "filters":     {"money_bucket": mb},
                    **stats,
                })

        # ── Pattern 3: Ticket% + Money% combo ────────────────────────
        tm_groups = defaultdict(list)
        for r in records:
            tb = bucket(r.get("pub_tickets"), TICKET_BUCKETS)
            mb = bucket(r.get("pub_money"),   MONEY_BUCKETS)
            if tb and mb:
                tm_groups[(tb, mb)].append(r)
        for (tb, mb), recs in tm_groups.items():
            stats = compute_ev(recs)
            if stats and stats["n"] >= min_samples:
                patterns.append({
                    "type":        "ticket_money_combo",
                    "description": f"Tickets: {tb} + Money: {mb}",
                    "filters":     {"ticket_bucket": tb, "money_bucket": mb},
                    **stats,
                })

        # ── Pattern 4: Line value + Ticket% + Money% ─────────────────
        ltm_groups = defaultdict(list)
        for r in records:
            tb = bucket(r.get("pub_tickets"), TICKET_BUCKETS)
            mb = bucket(r.get("pub_money"),   MONEY_BUCKETS)
            lb = r.get("line_bucket")
            if tb and mb and lb:
                ltm_groups[(lb, tb, mb)].append(r)
        for (lb, tb, mb), recs in ltm_groups.items():
            stats = compute_ev(recs)
            if stats and stats["n"] >= min_samples:
                patterns.append({
                    "type":        "line_ticket_money",
                    "description": f"Line: {lb} | Tickets: {tb} | Money: {mb}",
                    "filters":     {"line_bucket": lb, "ticket_bucket": tb, "money_bucket": mb},
                    **stats,
                })

        # ── Pattern 5: Line movement + Ticket% + Money% ──────────────
        mtm_groups = defaultdict(list)
        for r in records:
            tb  = bucket(r.get("pub_tickets"), TICKET_BUCKETS)
            mb  = bucket(r.get("pub_money"),   MONEY_BUCKETS)
            mov = r.get("move_bucket")
            if tb and mb and mov:
                mtm_groups[(mov, tb, mb)].append(r)
        for (mov, tb, mb), recs in mtm_groups.items():
            stats = compute_ev(recs)
            if stats and stats["n"] >= min_samples:
                patterns.append({
                    "type":        "move_ticket_money",
                    "description": f"Move: {mov} | Tickets: {tb} | Money: {mb}",
                    "filters":     {"move_bucket": mov, "ticket_bucket": tb, "money_bucket": mb},
                    **stats,
                })

        # ── Pattern 6: Full combo (line + move + tickets + money) ─────
        full_groups = defaultdict(list)
        for r in records:
            lb  = r.get("line_bucket")
            mov = r.get("move_bucket")
            tb  = bucket(r.get("pub_tickets"), TICKET_BUCKETS)
            mb  = bucket(r.get("pub_money"),   MONEY_BUCKETS)
            if lb and mov and tb and mb:
                full_groups[(lb, mov, tb, mb)].append(r)
        for (lb, mov, tb, mb), recs in full_groups.items():
            stats = compute_ev(recs)
            if stats and stats["n"] >= min_samples:
                patterns.append({
                    "type":        "full_combo",
                    "description": f"Line: {lb} | Move: {mov} | Tickets: {tb} | Money: {mb}",
                    "filters":     {
                        "line_bucket":   lb,
                        "move_bucket":   mov,
                        "ticket_bucket": tb,
                        "money_bucket":  mb,
                    },
                    **stats,
                })

        # ── Pattern 7: Sharp side flag + line + tickets ───────────────
        sharp_groups = defaultdict(list)
        for r in records:
            lb    = r.get("line_bucket")
            sharp = r.get("is_sharp_side")
            tb    = bucket(r.get("pub_tickets"), TICKET_BUCKETS)
            if lb and sharp is not None and tb:
                sharp_groups[(lb, str(sharp), tb)].append(r)
        for (lb, sharp, tb), recs in sharp_groups.items():
            stats = compute_ev(recs)
            if stats and stats["n"] >= min_samples:
                sharp_label = "Sharp side" if sharp == "True" else "Public side"
                patterns.append({
                    "type":        "sharp_side_pattern",
                    "description": f"{sharp_label} | Line: {lb} | Tickets: {tb}",
                    "filters":     {
                        "line_bucket":   lb,
                        "is_sharp_side": sharp,
                        "ticket_bucket": tb,
                    },
                    **stats,
                })

        # Sort by absolute EV (most interesting patterns first)
        patterns.sort(key=lambda x: abs(x.get("ev_pct", 0)), reverse=True)

        output[group_key] = {
            "sport_key": sport_key,
            "market":    market,
            "n_records": len(records),
            "n_labeled": len([r for r in records if r.get("outcome") is not None]),
            "patterns":  patterns[:100],  # Top 100 per group
        }

    return output


def run_pattern_discovery():
    """Run discovery and save to pipeline_output/patterns.json."""
    print("\n=== PATTERN DISCOVERY ===")
    Path("./pipeline_output").mkdir(exist_ok=True)

    patterns = discover_patterns(min_samples=5)

    total_patterns = sum(len(v["patterns"]) for v in patterns.values())
    print(f"  Discovered {total_patterns} patterns across {len(patterns)} sport/market groups")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(patterns, f, indent=2, default=str)

    print(f"  Saved → {OUTPUT_FILE}")
    return patterns


if __name__ == "__main__":
    run_pattern_discovery()
