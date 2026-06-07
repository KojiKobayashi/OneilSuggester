#!/usr/bin/env python3
"""Daily batch runner for OneilSuggester.

Usage::

    python batch/run_daily.py [--config CONFIG] [--output-dir OUTPUT_DIR]

The script:
1. Loads the stock list from ``data/stock_list.csv``.
2. Fetches the last *lookback_days* of OHLCV data for each ticker.
3. Scores each ticker using pattern detectors.
4. Writes the top-N long and top-N short results to
   ``docs/data/YYYY-MM-DD.json`` (JST date).
5. Updates ``docs/data/index.json`` with the list of available dates.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

# Ensure the project root is on the Python path so ``src`` can be imported
# when the script is executed directly (e.g. ``python batch/run_daily.py``).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.fetcher import fetch_ohlcv  # noqa: E402
from src.scoring import score_ticker_all  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Default paths ──────────────────────────────────────────────────────────────
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "config.yaml")
DEFAULT_STOCK_LIST = os.path.join(PROJECT_ROOT, "data", "stock_list.csv")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "docs", "data")
INDEX_FILENAME = "index.json"
MAX_DATES_IN_INDEX = 30  # keep at most this many dates in the index

_JST = ZoneInfo("Asia/Tokyo")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_stock_list(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str)


def update_index(output_dir: str, date_str: str) -> None:
    """Add *date_str* to the index file, keeping it sorted descending."""
    index_path = os.path.join(output_dir, INDEX_FILENAME)
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as fh:
            index_data: dict = json.load(fh)
    else:
        index_data = {"dates": []}

    dates: list[str] = index_data.get("dates", [])
    if date_str not in dates:
        dates.append(date_str)

    # Sort descending (newest first) and cap length
    dates.sort(reverse=True)
    dates = dates[:MAX_DATES_IN_INDEX]
    index_data["dates"] = dates

    with open(index_path, "w", encoding="utf-8") as fh:
        json.dump(index_data, fh, ensure_ascii=False, indent=2)

    logger.info("Updated index: %d dates", len(dates))


def run(config_path: str, output_dir: str) -> None:
    cfg = load_config(config_path)
    top_n: int = int(cfg.get("top_n", 20))
    lookback_days: int = int(cfg.get("lookback_days", 180))

    stocks = load_stock_list(DEFAULT_STOCK_LIST)
    logger.info("Loaded %d tickers", len(stocks))

    long_results: list[dict] = []
    short_results: list[dict] = []

    for _, row in stocks.iterrows():
        code: str = str(row["code"]).strip()
        name: str = str(row["name"]).strip()
        logger.info("Processing %s (%s)", code, name)

        df = fetch_ohlcv(code, period_days=lookback_days)
        if df is None:
            logger.warning("Skipping %s – no data", code)
            continue

        results = score_ticker_all(code, name, df)
        if results:
            for result in results:
                if result["type"] == "long":
                    long_results.append(result)
                else:
                    short_results.append(result)
                logger.info(
                    "  → %s score=%.4f signals=%s",
                    result["type"],
                    result["score"],
                    result["signals"],
                )
        else:
            logger.info("  → no pattern detected")

    # Sort each list descending by score and keep top N
    long_results.sort(key=lambda r: r["score"], reverse=True)
    short_results.sort(key=lambda r: r["score"], reverse=True)
    top_long = long_results[:top_n]
    top_short = short_results[:top_n]
    top_results = top_long + top_short

    # Determine output file name from today's JST date
    now_jst = datetime.now(timezone.utc).astimezone(_JST)
    date_str = now_jst.strftime("%Y-%m-%d")
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    output = {
        "generated_at": generated_at,
        "date": date_str,
        "items": top_results,
    }

    os.makedirs(output_dir, exist_ok=True)

    # Write date-named file
    dated_path = os.path.join(output_dir, f"{date_str}.json")
    with open(dated_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)
    logger.info(
        "Wrote %d long + %d short results to %s",
        len(top_long),
        len(top_short),
        dated_path,
    )

    # Update the index
    update_index(output_dir, date_str)


def main() -> None:
    parser = argparse.ArgumentParser(description="OneilSuggester daily batch")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to config.yaml")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write JSON output files",
    )
    args = parser.parse_args()
    run(args.config, args.output_dir)


if __name__ == "__main__":
    main()
