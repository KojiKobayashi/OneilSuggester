#!/usr/bin/env python3
"""Daily batch runner for OneilSuggester.

Usage::

    python batch/run_daily.py [--config CONFIG] [--output OUTPUT]

The script:
1. Loads the stock list from ``data/stock_list.csv``.
2. Fetches the last *lookback_days* of OHLCV data for each ticker.
3. Scores each ticker using pattern detectors.
4. Writes the top-N results to ``docs/data/latest.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import yaml

# Ensure the project root is on the Python path so ``src`` can be imported
# when the script is executed directly (e.g. ``python batch/run_daily.py``).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.fetcher import fetch_ohlcv  # noqa: E402
from src.scoring import score_ticker  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Default paths ──────────────────────────────────────────────────────────────
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "config.yaml")
DEFAULT_STOCK_LIST = os.path.join(PROJECT_ROOT, "data", "stock_list.csv")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "docs", "data", "latest.json")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_stock_list(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str)


def run(config_path: str, output_path: str) -> None:
    cfg = load_config(config_path)
    top_n: int = int(cfg.get("top_n", 20))
    lookback_days: int = int(cfg.get("lookback_days", 180))

    stocks = load_stock_list(DEFAULT_STOCK_LIST)
    logger.info("Loaded %d tickers", len(stocks))

    results: list[dict] = []

    for _, row in stocks.iterrows():
        code: str = str(row["code"]).strip()
        name: str = str(row["name"]).strip()
        logger.info("Processing %s (%s)", code, name)

        df = fetch_ohlcv(code, period_days=lookback_days)
        if df is None:
            logger.warning("Skipping %s – no data", code)
            continue

        result = score_ticker(code, name, df)
        if result is not None:
            results.append(result)
            logger.info("  → %s score=%.4f signals=%s", result["type"], result["score"], result["signals"])
        else:
            logger.info("  → no pattern detected")

    # Sort descending by score and keep top N
    results.sort(key=lambda r: r["score"], reverse=True)
    top_results = results[:top_n]

    # Attach metadata
    output = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "items": top_results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)

    logger.info("Wrote %d results to %s", len(top_results), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="OneilSuggester daily batch")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to config.yaml")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to output JSON")
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
