"""Cup-with-Handle pattern detection and sub-score calculation.

Detection criteria
------------------
* Cup look-back window: 120 trading days (``CUP_WINDOW``).
* Drawdown depth: 10 %–35 % from the left peak to the cup bottom.
* Base width: at least ``MIN_BASE_DAYS`` (20) days around the bottom.
* Handle: within the last ``HANDLE_WINDOW`` (20) trading days, must not
  decline more than ``MAX_HANDLE_DROP`` (12 %) from the handle high.
* Volume: average volume during the cup is *below* the 20-day average
  computed at the left edge of the cup, and the most-recent day shows a
  relative volume above ``BREAKOUT_VOL_RATIO`` (1.0).

Scoring weights (long signal)
------------------------------
  score = cup_shape * 0.3
        + handle_quality * 0.2
        + volume_pattern * 0.2
        + breakout_strength * 0.3
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Detection constants ────────────────────────────────────────────────────────
CUP_WINDOW = 120          # trading days to scan for the cup
MIN_DRAWDOWN = 0.10       # minimum cup depth (10 %)
MAX_DRAWDOWN = 0.35       # maximum cup depth (35 %)
MIN_BASE_DAYS = 20        # minimum days forming the base of the cup
HANDLE_WINDOW = 20        # look-back window for handle detection
MAX_HANDLE_DROP = 0.12    # maximum handle decline (12 %)
BREAKOUT_VOL_RATIO = 1.0  # relative-volume threshold at breakout

# ── Score weights ──────────────────────────────────────────────────────────────
W_CUP_SHAPE = 0.3
W_HANDLE_QUALITY = 0.2
W_VOLUME_PATTERN = 0.2
W_BREAKOUT_STRENGTH = 0.3


def is_valid_drawdown(drawdown: float) -> bool:
    """Return True when *drawdown* is within the valid cup depth range.

    Args:
        drawdown: Fractional drawdown from the left peak to the cup bottom
            (e.g. 0.20 for a 20 % decline).
    """
    return MIN_DRAWDOWN <= drawdown <= MAX_DRAWDOWN


def has_sufficient_base(cup_df: pd.DataFrame, bottom_price: float) -> bool:
    """Return True when enough trading days are near the cup bottom.

    Args:
        cup_df: DataFrame slice covering the cup window.
        bottom_price: Price at the cup bottom.
    """
    base_days = int((cup_df["Close"] <= bottom_price * 1.10).sum())
    return base_days >= MIN_BASE_DAYS


def is_handle_valid(handle_drop: float) -> bool:
    """Return True when the handle decline is within the allowed maximum.

    Args:
        handle_drop: Fractional decline from the handle high to the handle low.
    """
    return handle_drop <= MAX_HANDLE_DROP


def detect(df: pd.DataFrame) -> Optional[dict]:
    """Detect a cup-with-handle pattern in *df* and return a result dict.

    Args:
        df: DataFrame with columns ``Close``, ``High``, ``Low``, ``Volume``,
            ``vol_ma``, ``ma25``, ``ma75``.  Must contain at least
            ``CUP_WINDOW`` rows.

    Returns:
        A dict with keys ``score`` (float 0–1) and ``signals`` (list of str)
        when a pattern is found, otherwise ``None``.
    """
    if len(df) < CUP_WINDOW:
        return None

    cup_df = df.iloc[-CUP_WINDOW:].copy()

    # ── Identify left peak ────────────────────────────────────────────────────
    left_peak_idx = cup_df["Close"].iloc[: CUP_WINDOW // 2].idxmax()
    left_peak_price = cup_df.loc[left_peak_idx, "Close"]
    left_loc = cup_df.index.get_loc(left_peak_idx)

    # The bottom must occur *after* the left peak
    right_portion = cup_df.iloc[left_loc:]
    if len(right_portion) < MIN_BASE_DAYS:
        return None

    bottom_idx = right_portion["Close"].idxmin()
    bottom_price = right_portion.loc[bottom_idx, "Close"]

    # ── Drawdown check ────────────────────────────────────────────────────────
    drawdown = (left_peak_price - bottom_price) / left_peak_price
    if not is_valid_drawdown(drawdown):
        return None

    # ── Base width: days within 10 % of the bottom across the full cup ───────
    if not has_sufficient_base(cup_df, bottom_price):
        return None

    # ── Handle detection ─────────────────────────────────────────────────────
    handle_df = df.iloc[-HANDLE_WINDOW:]
    handle_high = handle_df["High"].max()
    handle_low = handle_df["Low"].min()
    handle_drop = (handle_high - handle_low) / handle_high if handle_high > 0 else 1.0
    if not is_handle_valid(handle_drop):
        return None

    # ── Volume pattern ────────────────────────────────────────────────────────
    # Average relative volume during cup formation vs baseline
    cup_rel_vol = (cup_df["Volume"] / cup_df["vol_ma"].replace(0, np.nan)).mean()
    last_rel_vol = (
        df["Volume"].iloc[-1] / df["vol_ma"].iloc[-1]
        if df["vol_ma"].iloc[-1] > 0
        else 0.0
    )
    volume_contraction = cup_rel_vol < 1.0
    breakout_vol = last_rel_vol >= BREAKOUT_VOL_RATIO

    # ── Sub-scores ────────────────────────────────────────────────────────────
    # cup_shape: how symmetric / ideal the drawdown is (0.25 = best ≈ ideal mid-range)
    ideal_drawdown = 0.25
    cup_shape = max(0.0, 1.0 - abs(drawdown - ideal_drawdown) / ideal_drawdown)

    # handle_quality: smaller drop → higher score
    handle_quality = max(0.0, 1.0 - handle_drop / MAX_HANDLE_DROP)

    # volume_pattern: contraction during cup + expansion at breakout
    volume_pattern = 0.0
    if volume_contraction:
        volume_pattern += 0.5
    if breakout_vol:
        volume_pattern += 0.5

    # breakout_strength: relative volume on breakout day
    breakout_strength = min(1.0, last_rel_vol / 2.0)

    score = (
        cup_shape * W_CUP_SHAPE
        + handle_quality * W_HANDLE_QUALITY
        + volume_pattern * W_VOLUME_PATTERN
        + breakout_strength * W_BREAKOUT_STRENGTH
    )

    signals: list[str] = ["cup detected", "handle formed"]
    if volume_contraction:
        signals.append("volume contraction")
    if breakout_vol:
        signals.append("volume expansion at breakout")

    return {"score": round(float(score), 4), "signals": signals}
