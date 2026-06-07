"""Fetches historical OHLCV data and optional fundamentals from Yahoo Finance."""

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


def _extract_growth(frame: pd.DataFrame, aliases: list[str]) -> Optional[float]:
    if frame is None or frame.empty or frame.shape[1] < 2:
        return None
    for alias in aliases:
        if alias in frame.index:
            series = pd.to_numeric(frame.loc[alias], errors="coerce").dropna()
            if len(series) < 2:
                return None
            latest = float(series.iloc[0])
            previous = float(series.iloc[1])
            if previous == 0:
                return None
            return (latest - previous) / abs(previous)
    return None


def fetch_fundamentals(code: str) -> dict:
    """Return optional growth metrics for CANSLIM-lite scoring."""
    try:
        ticker = yf.Ticker(code)
        quarterly = getattr(ticker, "quarterly_income_stmt", pd.DataFrame())
        annual = getattr(ticker, "income_stmt", pd.DataFrame())
        if quarterly is None or quarterly.empty:
            quarterly = getattr(ticker, "quarterly_financials", pd.DataFrame())
        if annual is None or annual.empty:
            annual = getattr(ticker, "financials", pd.DataFrame())
        return {
            "quarterly_revenue_growth": _extract_growth(
                quarterly,
                ["Total Revenue", "Operating Revenue", "Revenue"],
            ),
            "annual_revenue_growth": _extract_growth(
                annual,
                ["Total Revenue", "Operating Revenue", "Revenue"],
            ),
            "quarterly_eps_growth": _extract_growth(
                quarterly,
                ["Diluted EPS", "Basic EPS", "Reported EPS"],
            ),
            "annual_eps_growth": _extract_growth(
                annual,
                ["Diluted EPS", "Basic EPS", "Reported EPS"],
            ),
        }
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Failed to fetch fundamentals for %s: %s", code, exc)
        return {}
