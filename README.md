# Line-Tracker-Model

ML pipeline that tracks **multi-book line movement**, trains per-sport/market classifiers, and publishes a daily **+EV bet slate** for the Line-Tracker dashboard app.

**Goal:** Find sides where the model’s win probability (after calibration and shrinkage) exceeds a **no-vig Pinnacle** fair price, then size with fractional Kelly.

Related UI: [bpr-model](https://github.com/childersjac-max/bpr-model) (J LAB) — live +EV locks, alt ladders, and CLV tracking in the browser. This repo is the **batch ML backend**; bpr-model is the **interactive pricing tool**. Shared ideas: same-book de-vig, true CLV vs close, half-Kelly discipline.

---

## Quick start

```bash
pip install -r requirements.txt

# 1. Scrape odds + enrich injuries/splits
python pipeline.py --mode scrape

# 2. Append snapshots to line_history/
python pipeline.py --mode track

# 3. Pull final scores into outcomes.json
export ODDS_API_KEY=your_key
python pipeline.py --mode results

# 4. Train models (needs ≥50 samples per sport/market; ≥20 real outcomes or uses synthetic bootstrap)
python train.py --sport all --market all

# 5. Generate slate + append paper-trading log
python pipeline.py --mode predict --bankroll 10000 --min-signals 0

# 6. Grade paper picks (closing CLV + W/L) — also runs after `results`
python pipeline.py --mode grade-paper

# 7. Backtest (same math as live predict)
python pipeline.py --mode backtest --bankroll 10000
```

---

## Environment variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `ODDS_API_KEY` | Yes (default source) | [The Odds API](https://the-odds-api.com/) — odds, scores, results |
| `ODDSJAM_API_KEY` | If using OddsJam | Alternative odds + arb feed |
| `ODDS_DATA_SOURCE` | No | `the_odds_api` (default) or `oddsjam` |
| `HISTORICAL_SPORTS` | No | Comma-separated sport keys for historical pull |
| `DAILY_HOURS` | No | Hours between historical snapshots (default 6) |
| `HISTORICAL_PULL_SLEEP` | No | Sleep seconds between historical API calls |

Set in shell locally or in GitHub Actions → Settings → Secrets.

---

## Pipeline modes

| Mode | Command | What it does |
|------|---------|--------------|
| `scrape` | `pipeline.py --mode scrape` | Run `scraper.py`; refresh injury/splits caches; enrich histories |
| `track` | `pipeline.py --mode track` | Merge latest `jlab_data/` into `line_history/{event_id}.json` |
| `results` | `pipeline.py --mode results` | Fetch completed scores → `outcomes.json`; auto-run `grade-paper` |
| `historical` | `pipeline.py --mode historical --days 30` | Bulk historical odds pull (`data/historical.py`) |
| `predict` | `pipeline.py --mode predict` | Score upcoming games → `pipeline_output/bet_slate_latest.csv` + paper log |
| `grade-paper` | `pipeline.py --mode grade-paper` | Grade pending paper picks (CLV + W/L) → `paper_trading_log.csv` |
| `backtest` | `pipeline.py --mode backtest` | Simulate on labeled history (aligned with live scoring) |
| `patterns` | `pipeline.py --mode patterns` | Pattern mining → `pipeline_output/patterns.json` |

### Training (separate entry)

```bash
python train.py --sport basketball_nba --market spreads
python train.py --sport all --market all
```

Models save to `saved_models/{sport_key}__{market}.joblib` + `.json` metadata.

---

## Outputs

| File | Description |
|------|-------------|
| `pipeline_output/bet_slate_latest.csv` | Today’s recommended bets (consumed by dashboard) |
| `pipeline_output/paper_trading_log.csv` | Walk-forward log of every slate pick + closing CLV + graded P/L |
| `pipeline_output/paper_log_summary.json` | Aggregate stats after `grade-paper` |
| `pipeline_output/backtest_results.csv` | Historical simulation bets |
| `pipeline_output/backtest_metrics.json` | ROI, CLV, breakdowns by market/signal |

---

## How to read the bet slate

Each row is one **actionable side** on an upcoming game.

| Column | Meaning |
|--------|---------|
| `model_prob` | Calibrated XGBoost P(win) |
| `fair_prob` | No-vig Pinnacle benchmark (same-book de-vig) |
| `sized_prob` | `α·model_prob + (1−α)·fair_prob` with `α=0.5` |
| `edge_pct` | `(sized_prob − fair_prob) × 100` — **bet filter** (default ≥ 0.5%) |
| `edge_pct_raw` | `(model_prob − fair_prob) × 100` |
| `ev_pct` | Expected value % at `american_odds` / `book` |
| `bet_usd` | Quarter-Kelly stake for `--bankroll` |
| `confidence` | HIGH / MEDIUM / LOW from edge + stake size |
| `signals` | `SHARP_MONEY`, `REVERSE_LINE_MOVEMENT`, `PUBLIC_FADE`, `ARBITRAGE`, or `CLV_MODEL` |
| `trained_on` | `real` = trained on actual results; `synthetic` = bootstrap only — **avoid betting** |

**Do not bet** rows with `trained_on=synthetic` or far-future games with no line history (`pin_move_full=0`).

---

## Paper-trading log & true CLV

Every `predict` run **appends** new picks to `paper_trading_log.csv` (deduped by `event_id|market|side|line`).

After games finish, `grade-paper` (or `results`) fills:

- `closing_fair_prob` — no-vig Pinnacle from the last pre-game snapshot (≤30 min before tip)
- `clv_prob_pts` / `clv_pct` — bet-time `fair_prob` vs closing fair ([bpr-model CLV convention](https://github.com/childersjac-max/bpr-model))
- `outcome` / `pnl` — graded from `outcomes.json`

Positive average `clv_prob_pts` over 50+ picks is the best sign the model finds real edge, independent of short-term W/L variance.

---

## Backtest vs live

As of this version, **backtest uses the same rules as `score_all()`**:

- No-vig `fair_prob`
- `PROB_SHRINKAGE_ALPHA` (0.5)
- `MIN_EDGE_TO_BET` (0.5%)
- `MIN_MODEL_PROB` (30%)
- Kelly sizing on `sized_prob`
- True CLV: bet-time fair vs `pin_no_vig_prob_close`

---

## Repo layout

```
pipeline.py          # CLI orchestrator
train.py             # Train XGBoost per sport × market
scraper.py           # Live odds pull → jlab_data/
configs/config.py    # Sports, thresholds, paths
data/                # line_tracker, labeler, results, paper_log, sources/
features/            # movement, arbitrage, injury
models/              # LineMovementModel, scorer
backtest/            # Historical simulation (parity with scorer)
utils/               # odds_math, kelly, scoring (shared live/backtest)
line_history/        # Per-event snapshot JSON
saved_models/        # Trained models
pipeline_output/     # Slates, logs, backtest artifacts
```

---

## GitHub Actions

| Workflow | Schedule | Action |
|----------|----------|--------|
| `scrape.yml` | Every 30 min | scrape → track |
| `train.yml` | Every 2 h | results → train |
| `predict.yml` | Hourly :45 | predict → commit slate |
| `daily_pipeline.yml` | 19:00 UTC | Full daily run |
| `backtest.yml` | Manual | Backtest artifacts |

---

## Data sources

- **The Odds API** — odds + scores (required for `results` / grading)
- **OddsJam** — optional (`ODDS_DATA_SOURCE=oddsjam`)
- **Action Network** — splits (`data/splits.py`)
- **ESPN** — injuries (`data/injuries.py`)
- **VSiN** — splits in standalone scraper (pipeline enrich prefers Action Network)

---

## License / credits

Built for J LAB line-tracking. Betting math aligned with [bpr-model v3.5](https://github.com/childersjac-max/bpr-model) where noted (same-book de-vig, true CLV, Kelly).
