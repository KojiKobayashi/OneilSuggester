"""Entry-point for pattern detection and scoring.

This module wires together the technical indicator calculations and the
individual pattern detectors to produce a single scored result per ticker.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from src import indicators
from src.patterns import cup_with_handle, short_sell

logger = logging.getLogger(__name__)


def score_ticker(
    code: str,
    name: str,
    df: pd.DataFrame,
) -> Optional[dict]:
    """Run all pattern detectors against *df* and return the best result.

    Args:
        code: Ticker symbol (e.g. ``"7203.T"``).
        name: Human-readable company name.
        df: Raw OHLCV DataFrame as returned by :func:`src.fetcher.fetch_ohlcv`.

    Returns:
        A dict compatible with the output JSON schema, or ``None`` when no
        pattern is detected.
    """
    # ── Enrich with indicators ─────────────────────────────────────────────────
    df = indicators.add_moving_averages(df)
    df = indicators.add_volume_ma(df)

    results: list[dict] = []

    # ── Cup-with-handle (long) ─────────────────────────────────────────────────
    try:
        cwh = cup_with_handle.detect(df)
        if cwh is not None:
            results.append(
                {
                    "code": code,
                    "name": name,
                    "type": "long",
                    "score": cwh["score"],
                    "signals": cwh["signals"],
                }
            )
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("CWH detection failed for %s: %s", code, exc)

    # ── Short-sell (short) ────────────────────────────────────────────────────
    try:
        short = short_sell.detect(df)
        if short is not None:
            results.append(
                {
                    "code": code,
                    "name": name,
                    "type": "short",
                    "score": short["score"],
                    "signals": short["signals"],
                }
            )
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Short-sell detection failed for %s: %s", code, exc)

    if not results:
        return None

    # Return the highest-scoring result for this ticker
    return max(results, key=lambda r: r["score"])


def score_ticker_all(
    code: str,
    name: str,
    df: pd.DataFrame,
) -> list[dict]:
    """Run all pattern detectors against *df* and return every detected result.

    Unlike :func:`score_ticker`, this function returns **all** detected
    patterns for the ticker (at most one per pattern type), rather than only
    the highest-scoring one.  This is useful when callers want to maintain
    separate leader-boards for long and short signals.

    Args:
        code: Ticker symbol (e.g. ``"7203.T"``).
        name: Human-readable company name.
        df: Raw OHLCV DataFrame as returned by :func:`src.fetcher.fetch_ohlcv`.

    Returns:
        A (possibly empty) list of result dicts compatible with the output
        JSON schema.
    """
    df = indicators.add_moving_averages(df)
    df = indicators.add_volume_ma(df)

    results: list[dict] = []

    try:
        cwh = cup_with_handle.detect(df)
        if cwh is not None:
            results.append(
                {
                    "code": code,
                    "name": name,
                    "type": "long",
                    "score": cwh["score"],
                    "signals": cwh["signals"],
                }
            )
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("CWH detection failed for %s: %s", code, exc)

    try:
        short = short_sell.detect(df)
        if short is not None:
            results.append(
                {
                    "code": code,
                    "name": name,
                    "type": "short",
                    "score": short["score"],
                    "signals": short["signals"],
                }
            )
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Short-sell detection failed for %s: %s", code, exc)

    return results
