def mode_predict(bankroll, min_signals):
    from models.scorer import score_all
    from utils.kelly import simulate_growth

    slate = score_all(bankroll=bankroll, min_signals=min_signals)

    # Always write the file even if empty
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out_path = Path(OUTPUT_DIR) / "bet_slate_latest.csv"

    if slate.empty:
        # Write empty CSV with headers so the app knows it loaded
        import csv
        headers = ["sport","market","side","american_odds","american_odds_display",
                   "model_prob","fair_prob","edge_pct","ev_pct","bet_pct","bet_usd",
                   "confidence","signals","n_signals","pin_move_full","money_vs_tickets",
                   "book","line","is_home","event_id","sport_key"]
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
        logger.info("No edges found — wrote empty slate file.")
    else:
        slate.to_csv(out_path, index=False)
        logger.info(f"\n{len(slate)} bets found. Saved to {out_path}")
        print(slate[["sport","market","side","american_odds_display",
                      "edge_pct","bet_usd","confidence","signals"]].to_string(index=False))

    # Always commit
    logger.info(f"Slate written to {out_path}")
