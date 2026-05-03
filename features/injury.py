# features/injury.py
# =====================================================================
# INJURY ANNOTATION FOR BET SLATE
# =====================================================================
# Called after score_all() produces the raw bet DataFrame.
# Reads the injury cache (written by the scrape step) and adds:
#
#   home_injury_score  — severity for the home team (0=none, 0.5=Q, 1.0=Out/Doubtful)
#   away_injury_score  — severity for the away team
#   has_major_injury   — max(home,away) capped at 1.0 (game-level flag)
#   injured_players    — comma-sep list of key injured players with status
#   side_injury_score  — injury score for the SIDE we're betting (negative signal)
#   opp_injury_score   — injury score for the OPPONENT  (positive signal for our side)
#
# Signal tags added to the 'signals' column:
#   INJURY_RISK  — key player Out/Doubtful on the team we're betting
#   OPP_INJURED  — key player Out/Doubtful on the opposing team (edge booster)
#
# NOTE: Injuries are NOT fed into XGBoost (fixed feature set would break
#       inference on existing models). They annotate the output slate only.
#       Include injury columns in future retraining feature sets.
# =====================================================================

import pandas as pd
from data.injuries import get_game_injury_flag, load_injury_cache


def annotate_slate_with_injuries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add injury columns to the bet slate DataFrame.

    Required columns: home_team, away_team, sport_key, is_home
    Returns a new DataFrame with injury columns appended.
    """
    if df.empty:
        return df

    df = df.copy()

    # Null-fill defaults so the CSV always has these columns
    injury_defaults = {
        "home_injury_score": 0.0,
        "away_injury_score": 0.0,
        "has_major_injury":  0.0,
        "injured_players":   "",
        "side_injury_score": 0.0,
        "opp_injury_score":  0.0,
    }
    for col, default in injury_defaults.items():
        df[col] = default

    # Load the cached injury data written by the scrape step
    cache = load_injury_cache()
    injuries_by_sport = cache.get("injuries", {})

    if not injuries_by_sport:
        print("  [injury] Cache empty — ESPN data unavailable; slate unannotated.")
        return df

    enriched = 0
    for idx, row in df.iterrows():
        flags = get_game_injury_flag(
            home_team=str(row.get("home_team", "")),
            away_team=str(row.get("away_team", "")),
            sport_key=str(row.get("sport_key", "")),
            injuries_by_sport=injuries_by_sport,
        )

        home_score = float(flags.get("home_injury_score", 0.0))
        away_score = float(flags.get("away_injury_score", 0.0))
        is_home    = int(row.get("is_home", 0))

        side_score = home_score if is_home else away_score
        opp_score  = away_score if is_home else home_score

        df.at[idx, "home_injury_score"] = round(home_score, 2)
        df.at[idx, "away_injury_score"] = round(away_score, 2)
        df.at[idx, "has_major_injury"]  = float(flags.get("has_major_injury", 0.0))
        df.at[idx, "injured_players"]   = ", ".join(flags.get("injured_players", []))
        df.at[idx, "side_injury_score"] = round(side_score, 2)
        df.at[idx, "opp_injury_score"]  = round(opp_score, 2)

        if home_score > 0 or away_score > 0:
            enriched += 1

    print(f"  [injury] Annotated {enriched} bets with injury data "
          f"({(df['has_major_injury'] > 0).sum()} with key injuries)")
    return df


def apply_injury_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append INJURY_RISK / OPP_INJURED tags to the 'signals' column.

    INJURY_RISK  — our side has a key player Out or Doubtful (severity >= 0.5)
    OPP_INJURED  — opponent has a key player Out or Doubtful (severity >= 1.0)
                   This is a positive edge booster for our side.
    """
    if df.empty or "side_injury_score" not in df.columns:
        return df

    df = df.copy()

    def _update(row):
        sigs       = str(row.get("signals", "CLV_MODEL"))
        side_score = float(row.get("side_injury_score", 0.0))
        opp_score  = float(row.get("opp_injury_score",  0.0))

        sig_list = [s.strip() for s in sigs.split(",") if s.strip()]

        # Side injury — key player missing from the team we're betting on
        if side_score >= 0.5 and "INJURY_RISK" not in sig_list:
            sig_list.append("INJURY_RISK")

        # Opponent injury — key player missing from the team we're betting against
        if opp_score >= 1.0 and "OPP_INJURED" not in sig_list:
            sig_list.append("OPP_INJURED")

        return ", ".join(sig_list)

    df["signals"] = df.apply(_update, axis=1)
    return df
