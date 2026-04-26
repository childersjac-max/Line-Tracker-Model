# pipeline.py

import argparse
import json
import logging
import sys
from pathlib import Path
from configs.config import OUTPUT_DIR, SPORTS, MARKETS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def mode_scrape():
    try:
        import scraper
        scraper.run()
    except ImportError:
        logger.error("scraper.py not found.")
        sys.exit(1)


def mode_track():
    from data.line_tracker import load_latest_combined, update_line_history
    combined = load_latest_combined()
    if combined is None:
        logger.error("No scraped data found. Run --mode scrape first.")
        sys.exit(1)
    stats = update_line_history(combined)
    logger.info(f"New: {stats['new']} | Updated: {stats['updated']} | Skipped: {stats['skipped']}")


def mode_results(days_from=3):
    from data.results import fetch_and_store_outcomes
    fetch_and_store_outcomes(days_from=days_from)


def mode_predict(bankroll, min_signals):
    from models.scorer import score_all
    from utils.kelly import simulate_growth
    slate = score_all(bankroll=bankroll, min_signals=min_signals)
    if slate.empty:
        logger.info("No edges found today.")
        return
    out_path = Path(OUTPUT_DIR) / "bet_slate_latest.csv"
    slate.to_csv(out_path, index=False)
    logger.info(f"\n{len(slate)} bets found. Saved to {out_path}")
    print(slate[["sport","market","side","american_odds_display","edge_pct","bet_usd","confidence","signals"]].to_string(index=False))


def mode_backtest(bankroll, sport_filter, market_filter):
    from backtest.backtest import run_backtest, compute_metrics
    df, metrics = run_backtest(bankroll=bankroll, sport_filter=sport_filter, market_filter=market_filter)
    if df.empty:
        logger.warning("No backtest records.")
        return
    df.to_csv(Path(OUTPUT_DIR) / "backtest_results.csv", index=False)
    with open(Path(OUTPUT_DIR) / "backtest_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"ROI: {metrics.get('roi_pct',0):+.2f}% | Bets: {metrics.get('n_bets',0)} | Saved to {OUTPUT_DIR}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",        required=True, choices=["scrape","track","results","predict","backtest"])
    p.add_argument("--bankroll",    type=float, default=10000.0)
    p.add_argument("--min-signals", type=int,   default=0)
    p.add_argument("--sport",       default=None, choices=list(SPORTS.keys()))
    p.add_argument("--market",      default=None, choices=MARKETS)
    p.add_argument("--days",        type=int,   default=3)
    args = p.parse_args()

    if   args.mode == "scrape":   mode_scrape()
    elif args.mode == "track":    mode_track()
    elif args.mode == "results":  mode_results(days_from=args.days)
    elif args.mode == "predict":  mode_predict(args.bankroll, args.min_signals)
    elif args.mode == "backtest": mode_backtest(args.bankroll, args.sport, args.market)


if __name__ == "__main__":
    main()
