#!/usr/bin/env python3
"""Update ``data/stock_list.csv`` from JPX listed-securities data.

This script downloads the latest list of TSE-listed securities from the JPX
(Japan Exchange Group) website, filters for **Prime market** stocks, and
writes the result to ``data/stock_list.csv``.

Usage::

    python batch/update_stock_list.py [--min-market-cap MIN_MARKET_CAP_BILLION]

Requirements:
    pip install requests pandas xlrd

The JPX provides a fresh Excel file every business day at:
https://www.jpx.co.jp/markets/statistics-equities/misc/01.html

The downloaded file contains all TSE-listed securities with columns such as
コード (code), 銘柄名 (name), 市場・商品区分 (market segment), etc.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import urllib.request

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "stock_list.csv")

# JPX provides the listed securities file at this URL.
JPX_DATA_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
)

# Market segment name used in the JPX file for the Prime market
PRIME_MARKET_SEGMENT = "プライム（内国株式）"


def fetch_jpx_excel(url: str) -> bytes:
    logger.info("Downloading JPX stock list from %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def parse_jpx_excel(data: bytes) -> pd.DataFrame:
    """Parse the JPX Excel file and return a filtered DataFrame."""
    df = pd.read_excel(io.BytesIO(data), dtype=str)
    logger.info("Downloaded %d rows from JPX", len(df))

    # The column names differ slightly across file versions; find them flexibly.
    col_map: dict[str, str] = {}
    for col in df.columns:
        col_s = str(col).strip()
        if "コード" in col_s:
            col_map["code"] = col
        elif "銘柄名" in col_s:
            col_map["name"] = col
        elif "市場" in col_s or "商品区分" in col_s:
            col_map["market"] = col

    missing = [k for k in ("code", "name", "market") if k not in col_map]
    if missing:
        raise ValueError(
            f"Could not find required columns {missing} in JPX file. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.rename(columns={v: k for k, v in col_map.items()})
    df = df[["code", "name", "market"]].copy()
    df["code"] = df["code"].str.strip()
    df["name"] = df["name"].str.strip()
    df["market"] = df["market"].str.strip()

    # Filter for Prime market equities
    prime = df[df["market"] == PRIME_MARKET_SEGMENT].copy()
    logger.info("Prime market stocks: %d", len(prime))

    # Build Yahoo Finance ticker symbol (e.g. "7203" → "7203.T")
    prime = prime[prime["code"].str.match(r"^\d{4}$", na=False)].copy()
    prime["code"] = prime["code"] + ".T"

    return prime[["code", "name"]].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update data/stock_list.csv from JPX Prime market data"
    )
    parser.add_argument(
        "--url",
        default=JPX_DATA_URL,
        help="URL of the JPX listed-securities Excel file",
    )
    args = parser.parse_args()

    try:
        raw = fetch_jpx_excel(args.url)
    except Exception as exc:
        logger.error("Failed to download JPX data: %s", exc)
        sys.exit(1)

    try:
        df = parse_jpx_excel(raw)
    except Exception as exc:
        logger.error("Failed to parse JPX data: %s", exc)
        sys.exit(1)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    logger.info("Saved %d Prime market stocks to %s", len(df), OUTPUT_PATH)


if __name__ == "__main__":
    main()
