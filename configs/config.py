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
SHAR
