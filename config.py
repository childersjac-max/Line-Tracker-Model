# configs/config.py

SPORTS = {
    "americanfootball_nfl":   "NFL",
    "basketball_nba":         "NBA",
    "baseball_mlb":           "MLB",
    "icehockey_nhl":          "NHL",
    "americanfootball_ncaaf": "CFB",
    "basketball_ncaab":       "CBB",
}

MARKETS = ["h2h", "spreads", "totals"]

PROP_MARKETS = {
    "basketball_nba": ["player_points", "player_rebounds", "player_assists",
                       "player_threes", "player_points_rebounds_assists"],
    "baseball_mlb":   ["batter_hits", "batter_total_bases", "batter_rbis", "pitcher_strikeouts"],
    "americanfootball_nfl": ["player_pass_yds", "player_rush_yds",
                              "player_reception_yds", "player_receptions"],
}

PUBLIC_BOOKS = ["draftkings", "fanduel", "betmgm", "bovada", "williamhill_us", "bet365"]
SHARP_BOOKS  = ["pinnacle", "circa", "bookmaker"]
ALL_BOOKS    = PUBLIC_BOOKS + SHARP_BOOKS

SHARP_MONEY_MIN_MOVE_PTS   = 4.0
SHARP_MONEY_MAX_PUB_MOVE   = 2.0
RLM_MIN_PUBLIC_PCT         = 65.0
RLM_MIN_LINE_MOVE_AGAINST  = 0.5
FADE_MIN_PUBLIC_TICKETS    = 70.0
FADE_MAX_SHARP_MOVE        = 1.0
LATE_MOVE_HOURS_THRESHOLD  = 6.0
EARLY_OPEN_HOURS_THRESHOLD = 48.0

MIN_EDGE_TO_BET            = 0.005
MIN_SAMPLES_TO_TRAIN       = 50          # raised from 10 — 10 is far too few for stable training
MIN_REAL_OUTCOMES          = 20
MIN_MODEL_PROB             = 0.30        # don't bet on long-shot mathematical edges

KELLY_FRACTION             = 0.25
MAX_BET_PCT                = 0.04
MIN_BET_PCT                = 0.005
MAX_TOTAL_EXPOSURE_PCT     = 0.20

# Probability shrinkage toward fair (no-vig) prob before Kelly sizing.
# 1.0 = trust the model fully (original behavior)
# 0.0 = always bet the fair prob (zero edge, zero bets)
# 0.5 = compromise that dramatically reduces drawdown when the model is overconfident
PROB_SHRINKAGE_ALPHA       = 0.5

# Time-series CV settings
TIMESERIES_CV_SPLITS       = 5
CALIBRATION_TAIL_FRACTION  = 0.20        # fraction of (chronological) tail used for prefit calibration

XGB_PARAMS = {
    "objective":        "binary:logistic",
    "eval_metric":      "logloss",
    "n_estimators":     400,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "random_state":     42,
    "n_jobs":           -1,
}

XGB_OVERRIDES = {
    "americanfootball_nfl":   {"max_depth": 3, "n_estimators": 250, "min_child_weight": 8},
    "basketball_nba":         {"max_depth": 5, "n_estimators": 500, "min_child_weight": 4},
    "baseball_mlb":           {"max_depth": 5, "n_estimators": 600, "min_child_weight": 4},
    "icehockey_nhl":          {"max_depth": 4, "n_estimators": 350, "min_child_weight": 6},
    "americanfootball_ncaaf": {"max_depth": 3, "n_estimators": 250, "min_child_weight": 10},
    "basketball_ncaab":       {"max_depth": 4, "n_estimators": 350, "min_child_weight": 8},
}

DATA_DIR         = "./jlab_data"
LINE_HISTORY_DIR = "./line_history"
OUTCOMES_FILE    = "./outcomes.json"
MODELS_DIR       = "./saved_models"
OUTPUT_DIR       = "./pipeline_output"
