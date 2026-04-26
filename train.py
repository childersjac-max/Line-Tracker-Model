# train.py

import argparse
import logging
import sys
import pandas as pd
from configs.config import SPORTS, MARKETS, MIN_SAMPLES_TO_TRAIN, MIN_REAL_OUTCOMES
from data.line_tracker import load_all_histories
from data.labeler import label_histories, synthetic_outcomes
from data.results import load_outcomes
from features.movement import build_feature_dataframe
from models.model import LineMovementModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sport",       default="all", choices=list(SPORTS.keys()) + ["all"])
    p.add_argument("--market",      default="all", choices=MARKETS + ["all"])
    p.add_argument("--min-samples", type=int, default=MIN_SAMPLES_TO_TRAIN)
    p.add_argument("--synthetic",   action="store_true")
    args = p.parse_args()

    histories = load_all_histories()
    if not histories:
        logger.error("No line histories found. Run the scraper first.")
        sys.exit(1)
    logger.info(f"Loaded {len(histories)} event histories.")

    outcomes = load_outcomes()
    n_real   = len([k for k in outcomes if k.endswith("_home_ml")])
    logger.info(f"Real outcomes: {n_real}")

    if args.synthetic or n_real < MIN_REAL_OUTCOMES:
        logger.warning("Using SYNTHETIC outcomes for bootstrap.")
        records = synthetic_outcomes(label_histories(histories, outcomes={}))
    else:
        records = label_histories(histories, outcomes)

    feat_df = build_feature_dataframe(records)
    if feat_df.empty:
        logger.error("No features extracted.")
        sys.exit(1)
    logger.info(f"Feature matrix: {feat_df.shape}")

    sports_to_run  = list(SPORTS.keys()) if args.sport  == "all" else [args.sport]
    markets_to_run = MARKETS             if args.market == "all" else [args.market]
    results = {}

    for sport_key in sports_to_run:
        for market in markets_to_run:
            df = feat_df[(feat_df["sport_key"] == sport_key) & (feat_df["market"] == market) & (feat_df["outcome"].notna())].copy()
            if len(df) < args.min_samples:
                logger.warning(f"  [{sport_key}/{market}] Only {len(df)} samples, skipping.")
                continue
            n_pos = int(df["outcome"].sum())
            if n_pos < 10 or (len(df) - n_pos) < 10:
                continue
            try:
                model   = LineMovementModel(sport_key, market)
                metrics = model.train(df, n_splits=min(5, max(3, len(df) // 40)))
                model.save()
                results[f"{sport_key}/{market}"] = metrics
            except Exception as e:
                logger.error(f"  [{sport_key}/{market}] Error: {e}")

    logger.info("\n=== TRAINING SUMMARY ===")
    for key, m in results.items():
        logger.info(f"  {key:35} AUC {m['auc_mean']:.4f} | n={m['n_samples']}")


if __name__ == "__main__":
    main()
