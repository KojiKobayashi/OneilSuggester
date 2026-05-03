"""Fetches historical OHLCV data from Yahoo Finance using yfinance."""

import logging
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_ohlcv(code: str, period_days: int = 180) -> Optional[pd.DataFrame]:
    """Download *period_days* of daily OHLCV data for *code*.

    Args:
        code: Yahoo Finance ticker symbol (e.g. ``"7203.T"``).
        period_days: Number of calendar days of history to retrieve.

    Returns:
        A :class:`~pandas.DataFrame` with columns
        ``["Open", "Close", "High", "Low", "Volume"]`` indexed by date,
        or ``None`` when data cannot be retrieved.
    """
    try:
        ticker = yf.Ticker(code)
        df = ticker.history(period=f"{period_days}d", auto_adjust=True)
        if df is None or df.empty:
            logger.warning("No data returned for %s", code)
            return None
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[["Open", "Close", "High", "Low", "Volume"]].copy()
        df.dropna(subset=["Close", "Volume"], inplace=True)
        if len(df) < 20:
            logger.warning("Insufficient rows (%d) for %s", len(df), code)
            return None
        return df
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to fetch %s: %s", code, exc)
        return None
