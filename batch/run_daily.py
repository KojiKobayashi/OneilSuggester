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

from src.fetcher import fetch_fundamentals, fetch_ohlcv  # noqa: E402
from src.indicators import add_moving_averages  # noqa: E402
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
LEGACY_LATEST_FILENAME = "latest.json"
MAX_DATES_IN_INDEX = 30  # keep at most this many dates in the index

_JST = ZoneInfo("Asia/Tokyo")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_stock_list(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str)


def compute_market_trend(df: pd.DataFrame | None) -> bool:
    """Return True when the market index is in an uptrend."""
    if df is None or df.empty:
        return True
    enriched = add_moving_averages(df)
    valid = enriched.dropna(subset=["ma25", "ma75"])
    if len(valid) < 5:
        return True
    latest = valid.iloc[-1]
    return bool(
        latest["Close"] > latest["ma25"] > latest["ma75"]
        and valid["ma25"].iloc[-1] >= valid["ma25"].iloc[-5]
    )


def passes_liquidity_filters(
    df: pd.DataFrame,
    min_price: float,
    min_avg_dollar_volume: float,
) -> bool:
    """Return True when the ticker is liquid enough for ranking."""
    if df.empty:
        return False
    recent = df.iloc[-20:]
    last_close = float(recent["Close"].iloc[-1])
    avg_dollar_volume = float((recent["Close"] * recent["Volume"]).mean())
    return last_close >= min_price and avg_dollar_volume >= min_avg_dollar_volume


def compute_momentum_score(df: pd.DataFrame) -> float:
    """Return a weighted multi-horizon momentum score."""

    def price_return(period: int) -> float | None:
        if len(df) <= period:
            return None
        start = float(df["Close"].iloc[-period - 1])
        end = float(df["Close"].iloc[-1])
        if start <= 0:
            return None
        return (end - start) / start

    weighted_returns = []
    for period, weight in ((63, 0.4), (126, 0.3), (252, 0.3)):
        ret = price_return(period)
        if ret is not None:
            weighted_returns.append((ret, weight))

    if not weighted_returns:
        return 0.0

    total_weight = sum(weight for _, weight in weighted_returns)
    return float(sum(ret * weight for ret, weight in weighted_returns) / total_weight)


def assign_relative_strength_scores(universe_rows: list[dict]) -> None:
    """Annotate each universe row with a 0-1 relative-strength score."""
    if not universe_rows:
        return
    momentum_series = pd.Series(
        [row["momentum_score"] for row in universe_rows],
        index=[row["code"] for row in universe_rows],
        dtype=float,
    )
    ranks = momentum_series.rank(method="average", pct=True)
    for row in universe_rows:
        row["rs_score"] = float(ranks[row["code"]])


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


def remove_legacy_latest_output(output_dir: str) -> None:
    """Delete the legacy latest.json file so outputs stay date-based."""
    legacy_latest_path = os.path.join(output_dir, LEGACY_LATEST_FILENAME)
    if not os.path.exists(legacy_latest_path):
        return

    os.remove(legacy_latest_path)
    logger.info("Removed legacy output file: %s", legacy_latest_path)


def run(config_path: str, output_dir: str) -> None:
    cfg = load_config(config_path)
    top_n: int = int(cfg.get("top_n", 20))
    lookback_days: int = int(cfg.get("lookback_days", 400))
    market_index: str = str(cfg.get("market_index", "^N225"))
    min_price: float = float(cfg.get("min_price", 300))
    min_avg_dollar_volume: float = float(cfg.get("min_avg_dollar_volume", 200_000_000))
    include_canslim_fundamentals: bool = bool(
        cfg.get("include_canslim_fundamentals", False)
    )

    stocks = load_stock_list(DEFAULT_STOCK_LIST)
    logger.info("Loaded %d tickers", len(stocks))

    japan_long_results: list[dict] = []
    canslim_long_results: list[dict] = []
    short_results: list[dict] = []
    universe_rows: list[dict] = []

    market_df = fetch_ohlcv(market_index, period_days=lookback_days)
    market_in_uptrend = compute_market_trend(market_df)
    logger.info("Market trend for %s: %s", market_index, market_in_uptrend)

    for _, row in stocks.iterrows():
        code: str = str(row["code"]).strip()
        name: str = str(row["name"]).strip()
        logger.info("Processing %s (%s)", code, name)

        df = fetch_ohlcv(code, period_days=lookback_days)
        if df is None:
            logger.warning("Skipping %s – no data", code)
            continue

        if not passes_liquidity_filters(df, min_price, min_avg_dollar_volume):
            logger.info("Skipping %s – failed liquidity filter", code)
            continue

        universe_rows.append(
            {
                "code": code,
                "name": name,
                "df": df,
                "momentum_score": compute_momentum_score(df),
            }
        )

    assign_relative_strength_scores(universe_rows)

    for row in universe_rows:
        fundamentals = (
            fetch_fundamentals(row["code"]) if include_canslim_fundamentals else None
        )
        results = score_ticker_all(
            row["code"],
            row["name"],
            row["df"],
            market_in_uptrend=market_in_uptrend,
            rs_score=row.get("rs_score"),
            fundamentals=fundamentals,
        )
        if results:
            for result in results:
                if result["strategy"] == "japan_long":
                    japan_long_results.append(result)
                elif result["strategy"] == "canslim_long":
                    canslim_long_results.append(result)
                else:
                    short_results.append(result)
                logger.info(
                    "  → %s/%s score=%.4f signals=%s",
                    result["type"],
                    result["strategy"],
                    result["score"],
                    result["signals"],
                )
        else:
            logger.info("  → no pattern detected for %s", row["code"])

    # Sort each list descending by score and keep top N
    japan_long_results.sort(key=lambda r: r["score"], reverse=True)
    canslim_long_results.sort(key=lambda r: r["score"], reverse=True)
    short_results.sort(key=lambda r: r["score"], reverse=True)
    top_japan_long = japan_long_results[:top_n]
    top_canslim_long = canslim_long_results[:top_n]
    top_short = short_results[:top_n]
    top_results = top_japan_long + top_canslim_long + top_short

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
        "Wrote %d japan-long + %d canslim-long + %d short results to %s",
        len(top_japan_long),
        len(top_canslim_long),
        len(top_short),
        dated_path,
    )

    remove_legacy_latest_output(output_dir)

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
