# data/line_tracker.py

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from configs.config import LINE_HISTORY_DIR, DATA_DIR


def _hours_to_game(commence_time_str):
    try:
        commence = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        return (commence - now).total_seconds() / 3600.0
    except Exception:
        return None


def _history_path(event_id):
    Path(LINE_HISTORY_DIR).mkdir(parents=True, exist_ok=True)
    return Path(LINE_HISTORY_DIR) / f"{event_id}.json"


def load_history(event_id):
    p = _history_path(event_id)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def save_history(history):
    p = _history_path(history["event_id"])
    with open(p, "w") as f:
        json.dump(history, f, indent=2)


def _snapshot_from_event(event, splits_map, timestamp):
    htg = _hours_to_game(event.get("commence_time", ""))
    snap = {
        "timestamp":     timestamp,
        "hours_to_game": htg,
        "h2h":     event.get("h2h", {}),
        "spreads": event.get("spreads", {}),
        "totals":  event.get("totals", {}),
        "splits":  {},
    }
    key = f"{event.get('away_team','').upper()}|{event.get('home_team','').upper()}"
    if key in splits_map:
        snap["splits"] = splits_map[key]
    return snap


def _build_splits_map(splits_list):
    m = {}
    for ev in splits_list:
        away = ev.get("awayTeam", "").upper()
        home = ev.get("homeTeam", "").upper()
        m[f"{away}|{home}"] = ev.get("features", {})
    return m


def update_line_history(combined_snapshot):
    timestamp = combined_snapshot.get("timestamp_utc", datetime.now(timezone.utc).isoformat())
    stats = {"updated": 0, "new": 0, "skipped": 0}
    for sport_key, block in combined_snapshot.get("sports", {}).items():
        events = block.get("events", [])
        splits_map = _build_splits_map(block.get("splits", []))
        for event in events:
            event_id = event.get("event_id")
            if not event_id:
                stats["skipped"] += 1
                continue
            htg = _hours_to_game(event.get("commence_time", ""))
            if htg is not None and htg < -1.0:
                stats["skipped"] += 1
                continue
            snap = _snapshot_from_event(event, splits_map, timestamp)
            history = load_history(event_id)
            if history is None:
                history = {
                    "event_id":      event_id,
                    "sport_key":     sport_key,
                    "home_team":     event.get("home_team"),
                    "away_team":     event.get("away_team"),
                    "commence_time": event.get("commence_time"),
                    "snapshots":     [snap],
                }
                stats["new"] += 1
            else:
                history["snapshots"].append(snap)
                stats["updated"] += 1
            save_history(history)
    return stats


def load_all_histories():
    p = Path(LINE_HISTORY_DIR)
    if not p.exists():
        return []
    histories = []
    for f in p.glob("*.json"):
        try:
            with open(f) as fh:
                histories.append(json.load(fh))
        except Exception:
            pass
    return histories


def load_latest_combined():
    data_dir = Path(DATA_DIR)
    files = sorted(data_dir.glob("jlab_data_*.json"))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)
