
import os
import json
import time
import re
from datetime import datetime, timezone
from pathlib import Path
import requests
from bs4 import BeautifulSoup

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "YOUR_KEY_HERE")
OUTPUT_DIR = Path("./jlab_data")
OUTPUT_DIR.mkdir(exist_ok=True)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
HTTP_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

PUBLIC_BOOKS = ["draftkings", "fanduel", "betmgm", "bovada", "williamhill_us", "bet365"]
SHARP_BOOKS = ["pinnacle", "circa", "bookmaker"]

SPORTS = {
    "americanfootball_nfl":   {"vsin": "NFL", "label": "NFL"},
    "basketball_nba":         {"vsin": "NBA", "label": "NBA"},
    "baseball_mlb":           {"vsin": "MLB", "label": "MLB"},
    "icehockey_nhl":          {"vsin": "NHL", "label": "NHL"},
    "americanfootball_ncaaf": {"vsin": "CFB", "label": "CFB"},
    "basketball_ncaab":       {"vsin": "CBB", "label": "CBB"},
}

PROP_MARKETS = {
    "basketball_nba": [
        "player_points", "player_rebounds", "player_assists",
        "player_threes", "player_points_rebounds_assists",
    ],
    "baseball_mlb": [
        "batter_hits", "batter_total_bases", "batter_rbis", "pitcher_strikeouts",
    ],
    "americanfootball_nfl": [
        "player_pass_yds", "player_rush_yds",
        "player_reception_yds", "player_receptions",
    ],
}

VSIN_PROPS_CONFIG = {
    "basketball_nba": {
        "base": "https://data.vsin.com/nba/player-props/",
        "stats": [
            ("points", "Points"), ("rebounds", "Rebounds"), ("assists", "Assists"),
            ("threespointersmade", "3PT Made"), ("pointsreboundsassists", "PRA"),
        ],
    },
    "baseball_mlb": {
        "base": "https://data.vsin.com/mlb/player-props/",
        "stats": [
            ("hits", "Hits"), ("totalbases", "Total Bases"),
            ("rbi", "RBI"), ("strikeouts", "Strikeouts (P)"),
        ],
    },
    "americanfootball_nfl": {
        "base": "https://data.vsin.com/nfl/player-props/",
        "stats": [
            ("passingyards", "Pass Yds"), ("rushingyards", "Rush Yds"),
            ("receivingyards", "Rec Yds"), ("receivingreceptions", "Receptions"),
        ],
    },
}


def fetch_odds(sport_key, markets=None):
    if markets is None:
        markets = ["h2h", "spreads", "totals"]
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "regions": "us,eu",
        "markets": ",".join(markets),
        "oddsFormat": "american",
        "apiKey": ODDS_API_KEY,
    }
    r = requests.get(url, params=params, timeout=15)
    if r.status_code != 200:
        print(f"  [odds] {sport_key} -> HTTP {r.status_code}: {r.text[:200]}")
        return []
    return r.json()


def fetch_player_props_for_event(sport_key, event_id):
    if sport_key not in PROP_MARKETS:
        return {}
    markets = ",".join(PROP_MARKETS[sport_key])
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/events/{event_id}/odds"
    params = {
        "regions": "us",
        "markets": markets,
        "oddsFormat": "american",
        "apiKey": ODDS_API_KEY,
    }
    r = requests.get(url, params=params, timeout=15)
    if r.status_code != 200:
        return {}
    return r.json()


def parse_props_payload(payload):
    rows = []
    for bk in payload.get("bookmakers", []):
        for market in bk.get("markets", []):
            for outcome in market.get("outcomes", []):
                rows.append({
                    "player": outcome.get("description"),
                    "market": market["key"],
                    "side": outcome.get("name"),
                    "line": outcome.get("point"),
                    "price_american": outcome.get("price"),
                    "book_key": bk["key"],
                    "book_title": bk["title"],
                })
    return rows


def fetch_vsin_splits(vsin_sport, source="DK"):
    url = f"https://data.vsin.com/betting-splits/?source={source}&sport={vsin_sport}"
    r = requests.get(url, headers=HTTP_HEADERS, timeout=15)
    if r.status_code != 200:
        print(f"  [vsin splits] {vsin_sport} -> HTTP {r.status_code}")
        return {"events": []}
    return parse_vsin_splits(r.text)


def parse_vsin_splits(html):
    soup = BeautifulSoup(html, "html.parser")
    events = []
    for tbl in soup.find_all("table"):
        rows = tbl.find_all("tr")
        i = 0
        while i < len(rows) - 1:
            row_a = rows[i]
            row_h = rows[i + 1]
            cells_a = [c.get_text(strip=True) for c in row_a.find_all("td")]
            cells_h = [c.get_text(strip=True) for c in row_h.find_all("td")]
            if len(cells_a) < 8 or len(cells_h) < 8:
                i += 1
                continue
            away = cells_a[0]
            home = cells_h[0]
            if not away or away.lower() in ("team", "away", "home"):
                i += 1
                continue
            if not re.search(r"[A-Za-z]", away):
                i += 1
                continue
            try:
                event = {
                    "awayTeam": away,
                    "homeTeam": home,
                    "spread": {
                        "lineAway":    _parse_num(cells_a[1]),
                        "moneyAway":   _parse_pct(cells_a[2]),
                        "ticketsAway": _parse_pct(cells_a[3]),
                        "lineHome":    _parse_num(cells_h[1]),
                        "moneyHome":   _parse_pct(cells_h[2]),
                        "ticketsHome": _parse_pct(cells_h[3]),
                    },
                    "total": {
                        "line":         _parse_num(cells_a[4]) or _parse_num(cells_h[4]),
                        "moneyOver":    _parse_pct(cells_a[5]),
                        "ticketsOver":  _parse_pct(cells_a[6]),
                        "moneyUnder":   _parse_pct(cells_h[5]),
                        "ticketsUnder": _parse_pct(cells_h[6]),
                    },
                    "moneyline": {
                        "moneyAway":   _parse_pct(cells_a[7] if len(cells_a) > 7 else ""),
                        "ticketsAway": _parse_pct(cells_a[8] if len(cells_a) > 8 else ""),
                        "moneyHome":   _parse_pct(cells_h[7] if len(cells_h) > 7 else ""),
                        "ticketsHome": _parse_pct(cells_h[8] if len(cells_h) > 8 else ""),
                    },
                }
                events.append(event)
            except (IndexError, ValueError):
                pass
            i += 2
    return {"events": events}


def _parse_pct(s):
    if not s:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else None


def _parse_num(s):
    if not s:
        return None
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else None


def fetch_vsin_props_history(sport_key):
    cfg = VSIN_PROPS_CONFIG.get(sport_key)
    if not cfg:
        return []
    out = []
    for slug, label in cfg["stats"]:
        url = f"{cfg['base']}?stat={slug}&range=cs&situation=all&siteid=all"
        try:
            r = requests.get(url, headers=HTTP_HEADERS, timeout=15)
            if r.status_code != 200:
                continue
            out.extend(parse_vsin_props(r.text, label, sport_key))
            time.sleep(0.5)
        except Exception as e:
            print(f"  [vsin props] {label}: {e}")
    return out


def parse_vsin_props(html, stat_label, sport_key):
    soup = BeautifulSoup(html, "html.parser")
    rows_out = []
    for row in soup.find_all("tr"):
        cells = [td.get_text(strip=True).replace("\xa0", " ") for td in row.find_all("td")]
        if len(cells) < 16:
            continue
        player = cells[0]
        if not player or "player" in player.lower() or "prop" in player.lower():
            continue
        try:
            dk_line = float(cells[2]) if cells[2] else None
            dk_odds = int(cells[3]) if cells[3] else None
        except ValueError:
            continue
        if dk_line is None:
            continue
        rows_out.append({
            "player": player, "game": cells[1], "stat": stat_label,
            "sport_key": sport_key, "dkLine": dk_line, "dkOdds": dk_odds,
            "seasonRecord": cells[6], "seasonRoi": cells[8],
            "gamesPlayed": _parse_num(cells[9]), "seasonAvg": _parse_num(cells[10]),
            "lowValue": _parse_num(cells[11]), "highValue": _parse_num(cells[12]),
            "histRecord": cells[13], "plusMinus": cells[14],
            "hitPct": _parse_pct(cells[15]), "histRoi": cells[16],
        })
    return rows_out


def extract_book_odds(event):
    out = {
        "event_id":    event.get("id"),
        "sport_key":   event.get("sport_key"),
        "sport_title": event.get("sport_title"),
        "commence_time": event.get("commence_time"),
        "home_team":   event.get("home_team"),
        "away_team":   event.get("away_team"),
        "h2h": {}, "spreads": {}, "totals": {},
    }
    for bk in event.get("bookmakers", []):
        bk_key = bk["key"]
        for market in bk.get("markets", []):
            mkey = market["key"]
            for o in market.get("outcomes", []):
                name  = o.get("name")
                price = o.get("price")
                point = o.get("point")
                if mkey == "h2h":
                    out["h2h"].setdefault(name, {})[bk_key] = price
                elif mkey == "spreads":
                    out["spreads"].setdefault(name, {})[bk_key] = {"line": point, "price": price}
                elif mkey == "totals":
                    out["totals"].setdefault(name, {})[bk_key] = {"line": point, "price": price}
    return out


def derive_pin_pub_signals(flat_event):
    sig = {"h2h": {}, "spreads": {}, "totals": {}}

    def best_of(prices_dict, books):
        cands = [(b, prices_dict[b]) for b in books if b in prices_dict]
        if not cands:
            return None
        cands.sort(key=lambda x: x[1] if isinstance(x[1], (int, float)) else -9999, reverse=True)
        return {"book": cands[0][0], "price": cands[0][1]}

    def first_of(prices_dict, books):
        for b in books:
            if b in prices_dict:
                return {"book": b, "price": prices_dict[b]}
        return None

    for team, prices in flat_event.get("h2h", {}).items():
        pub   = best_of(prices, PUBLIC_BOOKS)
        sharp = first_of(prices, SHARP_BOOKS)
        if pub and sharp:
            sig["h2h"][team] = {"public": pub, "sharp": sharp, "diff_points": pub["price"] - sharp["price"]}

    for team, books in flat_event.get("spreads", {}).items():
        pub_books   = {b: v["price"] for b, v in books.items() if b in PUBLIC_BOOKS}
        sharp_books = {b: v["price"] for b, v in books.items() if b in SHARP_BOOKS}
        pub   = best_of(pub_books, PUBLIC_BOOKS)
        sharp = first_of(sharp_books, SHARP_BOOKS)
        if pub and sharp:
            sig["spreads"][team] = {
                "public": {**pub, "line": books[pub["book"]]["line"]},
                "sharp":  {**sharp, "line": books[sharp["book"]]["line"]},
                "diff_points": pub["price"] - sharp["price"],
            }

    for side, books in flat_event.get("totals", {}).items():
        pub_books   = {b: v["price"] for b, v in books.items() if b in PUBLIC_BOOKS}
        sharp_books = {b: v["price"] for b, v in books.items() if b in SHARP_BOOKS}
        pub   = best_of(pub_books, PUBLIC_BOOKS)
        sharp = first_of(sharp_books, SHARP_BOOKS)
        if pub and sharp:
            sig["totals"][side] = {
                "public": {**pub, "line": books[pub["book"]]["line"]},
                "sharp":  {**sharp, "line": books[sharp["book"]]["line"]},
                "diff_points": pub["price"] - sharp["price"],
            }
    return sig


def derive_splits_features(splits_event):
    out = {}
    for market_name, key_pairs in [
        ("moneyline", [("ticketsAway", "moneyAway"), ("ticketsHome", "moneyHome")]),
        ("spread",    [("ticketsAway", "moneyAway"), ("ticketsHome", "moneyHome")]),
        ("total",     [("ticketsOver", "moneyOver"),  ("ticketsUnder", "moneyUnder")]),
    ]:
        m  = splits_event.get(market_name) or {}
        t1 = m.get(key_pairs[0][0])
        n1 = m.get(key_pairs[0][1])
        t2 = m.get(key_pairs[1][0])
        n2 = m.get(key_pairs[1][1])
        if None in (t1, n1, t2, n2):
            continue
        diff1     = n1 - t1
        diff2     = n2 - t2
        magnitude = max(abs(diff1), abs(diff2))
        if diff1 > diff2:
            sharp_label  = "away" if "Away" in key_pairs[0][0] else "over"
            public_label = "home" if "Home" in key_pairs[1][0] else "under"
        else:
            sharp_label  = "home" if "Home" in key_pairs[1][0] else "under"
            public_label = "away" if "Away" in key_pairs[0][0] else "over"
        out[market_name] = {
            "sharp_side":        sharp_label,
            "public_side":       public_label,
            "magnitude_pts":     round(magnitude, 1),
            "sharp_money_pct":   round(n1 if diff1 > diff2 else n2, 1),
            "sharp_tickets_pct": round(t1 if diff1 > diff2 else t2, 1),
            "public_money_pct":  round(n2 if diff1 > diff2 else n1, 1),
            "public_tickets_pct": round(t2 if diff1 > diff2 else t1, 1),
        }
    return out


def run():
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    print(f"\n=== J LAB Data Pull — {timestamp} ===\n")

    if ODDS_API_KEY in (None, "", "YOUR_KEY_HERE"):
        print("Set ODDS_API_KEY env var first.")
        return

    combined = {"timestamp_utc": timestamp, "sports": {}}

    for sport_key, meta in SPORTS.items():
        print(f"--- {meta['label']} ({sport_key}) ---")
        sport_block = {"events": [], "splits": [], "props_history": [], "props_live": []}

        print(f"  [1/4] Fetching cross-book odds...")
        events = fetch_odds(sport_key)
        for ev in events:
            flat = extract_book_odds(ev)
            flat["pin_vs_pub"] = derive_pin_pub_signals(flat)
            sport_block["events"].append(flat)
        print(f"        Got {len(sport_block['events'])} events")

        print(f"  [2/4] Fetching VSiN splits...")
        splits = fetch_vsin_splits(meta["vsin"])
        for ev in splits.get("events", []):
            ev["features"] = derive_splits_features(ev)
            sport_block["splits"].append(ev)
        print(f"        Got {len(sport_block['splits'])} splits events")

        if sport_key in VSIN_PROPS_CONFIG:
            print(f"  [3/4] Fetching VSiN prop history...")
            props_hist = fetch_vsin_props_history(sport_key)
            sport_block["props_history"] = props_hist
            print(f"        Got {len(props_hist)} prop rows")
        else:
            print(f"  [3/4] (no VSiN props for {meta['label']})")

        if sport_key in PROP_MARKETS and events:
            print(f"  [4/4] Fetching live prop pricing...")
            for ev in events[:20]:
                payload = fetch_player_props_for_event(sport_key, ev["id"])
                if payload:
                    rows = parse_props_payload(payload)
                    for row in rows:
                        row["event_id"]  = ev["id"]
                        row["away_team"] = ev["away_team"]
                        row["home_team"] = ev["home_team"]
                    sport_block["props_live"].extend(rows)
                time.sleep(0.3)
            print(f"        Got {len(sport_block['props_live'])} live prop rows")

        combined["sports"][sport_key] = sport_block
        time.sleep(1.0)

    out_path = OUTPUT_DIR / f"jlab_data_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"\nWrote combined file: {out_path}")

    for sport_key, block in combined["sports"].items():
        for data_type in ("events", "splits", "props_history", "props_live"):
            data = block.get(data_type) or []
            if not data:
                continue
            fp = OUTPUT_DIR / f"{sport_key}_{data_type}_{timestamp}.json"
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
    print(f"Wrote per-sport breakdowns to {OUTPUT_DIR}\n")


if __name__ == "__main__":
    run()
