# data/labeler.py

import json
from pathlib import Path
import pandas as pd
from configs.config import OUTCOMES_FILE
from data.line_tracker import load_all_histories


def load_outcomes():
    p = Path(OUTCOMES_FILE)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def label_histories(histories=None, outcomes=None):
    if histories is None:
        histories = load_all_histories()
    if outcomes is None:
        outcomes = load_outcomes()
    records = []
    for hist in histories:
        eid       = hist["event_id"]
        sport_key = hist["sport_key"]
        home      = hist["home_team"]
        away      = hist["away_team"]
        commence  = hist["commence_time"]
        snaps     = hist["snapshots"]
        if not snaps:
            continue
        last = snaps[-1]
        for team in last.get("h2h", {}):
            is_home = (team == home)
            outcome_key = f"{eid}_home_ml" if is_home else f"{eid}_away_ml"
            records.append({
                "event_id": eid, "sport_key": sport_key,
                "home_team": home, "away_team": away,
                "commence_time": commence, "market": "h2h",
                "side": team, "is_home": is_home, "line": None,
                "snapshots": snaps, "outcome": outcomes.get(outcome_key),
            })
        spreads = last.get("spreads", {})
        for team in spreads:
            is_home = (team == home)
            pin = spreads[team].get("pinnacle", {})
            line = pin.get("line") if isinstance(pin, dict) else None
            hs = outcomes.get(f"{eid}_home_score")
            as_ = outcomes.get(f"{eid}_away_score")
            spread_outcome = None
            if hs is not None and as_ is not None and line is not None:
                margin = (hs - as_) if is_home else (as_ - hs)
                spread_outcome = 1 if margin + line > 0 else 0
            records.append({
                "event_id": eid, "sport_key": sport_key,
                "home_team": home, "away_team": away,
                "commence_time": commence, "market": "spreads",
                "side": team, "is_home": is_home, "line": line,
                "snapshots": snaps, "outcome": spread_outcome,
            })
        for side in ["Over", "Under"]:
            totals = last.get("totals", {})
            if side not in totals:
                continue
            pin = totals[side].get("pinnacle", {})
            line = pin.get("line") if isinstance(pin, dict) else None
            total_pts = outcomes.get(f"{eid}_total")
            total_outcome = None
            if total_pts is not None and line is not None:
                total_outcome = 1 if (side == "Over" and total_pts > line) or (side == "Under" and total_pts < line) else 0
            records.append({
                "event_id": eid, "sport_key": sport_key,
                "home_team": home, "away_team": away,
                "commence_time": commence, "market": "totals",
                "side": side, "is_home": None, "line": line,
                "snapshots": snaps, "outcome": total_outcome,
            })
    return records


def synthetic_outcomes(records, noise=0.05, seed=42):
    import numpy as np
    from utils.odds_math import no_vig_prob
    rng = np.random.default_rng(seed)
    labeled = []
    for rec in records:
        snaps = rec.get("snapshots", [])
        if not snaps:
            continue
        last = snaps[-1]
        fair_prob = 0.5
        try:
            if rec["market"] == "h2h":
                h2h = last.get("h2h", {})
                teams = list(h2h.keys())
                if len(teams) == 2:
                    pa = h2h[teams[0]].get("pinnacle")
                    pb = h2h[teams[1]].get("pinnacle")
                    if pa and pb:
                        probs = no_vig_prob(pa, pb)
                        fair_prob = probs[0] if rec["side"] == teams[0] else probs[1]
            elif rec["market"] == "totals":
                totals = last.get("totals", {})
                ov = totals.get("Over", {}).get("pinnacle", {})
                un = totals.get("Under", {}).get("pinnacle", {})
                if isinstance(ov, dict) and isinstance(un, dict):
                    pa, pb = ov.get("price"), un.get("price")
                    if pa and pb:
                        p_over, _ = no_vig_prob(pa, pb)
                        fair_prob = p_over if rec["side"] == "Over" else 1 - p_over
        except Exception:
            pass
        rec = dict(rec)
        rec["outcome"] = int(rng.random() < float(np.clip(fair_prob + rng.normal(0, noise), 0.05, 0.95)))
        labeled.append(rec)
    return labeled
