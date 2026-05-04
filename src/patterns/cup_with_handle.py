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
BASE_TOLERANCE = 1.10     # price must be within 10 % of the bottom to count as base

# ── Score weights ──────────────────────────────────────────────────────────────
W_CUP_SHAPE = 0.3
W_HANDLE_QUALITY = 0.2
W_VOLUME_PATTERN = 0.2
W_BREAKOUT_STRENGTH = 0.3


def is_valid_drawdown(df: pd.DataFrame) -> bool:
    """Return True when the cup drawdown is within the valid range (10%–35%).

    Args:
        df: DataFrame with at least ``CUP_WINDOW`` rows and a ``Close`` column.
    """
    if len(df) < CUP_WINDOW:
        return False
    cup_df = df.iloc[-CUP_WINDOW:]
    left_peak_idx = cup_df["Close"].iloc[:CUP_WINDOW // 2].idxmax()
    left_peak_price = cup_df.loc[left_peak_idx, "Close"]
    left_loc = cup_df.index.get_loc(left_peak_idx)
    right_portion = cup_df.iloc[left_loc:]
    if len(right_portion) < MIN_BASE_DAYS:
        return False
    bottom_price = right_portion["Close"].min()
    drawdown = (left_peak_price - bottom_price) / left_peak_price
    return bool(MIN_DRAWDOWN <= drawdown <= MAX_DRAWDOWN)


def has_sufficient_base(df: pd.DataFrame) -> bool:
    """Return True when enough trading days are near the cup bottom.

    Args:
        df: DataFrame with at least ``CUP_WINDOW`` rows and a ``Close`` column.
    """
    if len(df) < CUP_WINDOW:
        return False
    cup_df = df.iloc[-CUP_WINDOW:]
    left_peak_idx = cup_df["Close"].iloc[:CUP_WINDOW // 2].idxmax()
    left_loc = cup_df.index.get_loc(left_peak_idx)
    right_portion = cup_df.iloc[left_loc:]
    if len(right_portion) < MIN_BASE_DAYS:
        return False
    bottom_price = right_portion["Close"].min()
    base_days = int((cup_df["Close"] <= bottom_price * BASE_TOLERANCE).sum())
    return bool(base_days >= MIN_BASE_DAYS)


def is_handle_valid(df: pd.DataFrame) -> bool:
    """Return True when the handle decline is within the allowed maximum (12%).

    Args:
        df: DataFrame with at least ``HANDLE_WINDOW`` rows and ``High``/``Low`` columns.
    """
    if len(df) < HANDLE_WINDOW:
        return False
    handle_df = df.iloc[-HANDLE_WINDOW:]
    handle_high = handle_df["High"].max()
    handle_low = handle_df["Low"].min()
    handle_drop = (handle_high - handle_low) / handle_high if handle_high > 0 else 1.0
    return bool(handle_drop <= MAX_HANDLE_DROP)


def calc_cup_shape(df: pd.DataFrame) -> float:
    """Return the cup-shape sub-score [0, 1]: ideal drawdown ≈ 25 %.

    A drawdown of exactly 25 % scores 1.0; scores decrease linearly as the
    drawdown deviates from the ideal.

    Args:
        df: DataFrame with at least ``CUP_WINDOW`` rows and a ``Close`` column.
    """
    if len(df) < CUP_WINDOW:
        return 0.0
    cup_df = df.iloc[-CUP_WINDOW:]
    left_peak_idx = cup_df["Close"].iloc[:CUP_WINDOW // 2].idxmax()
    left_peak_price = cup_df.loc[left_peak_idx, "Close"]
    left_loc = cup_df.index.get_loc(left_peak_idx)
    bottom_price = cup_df.iloc[left_loc:]["Close"].min()
    drawdown = (left_peak_price - bottom_price) / left_peak_price
    ideal_drawdown = 0.25
    return float(max(0.0, 1.0 - abs(drawdown - ideal_drawdown) / ideal_drawdown))


def calc_handle_quality(df: pd.DataFrame) -> float:
    """Return the handle-quality sub-score [0, 1]: smaller handle drop = higher score.

    A handle drop of 0 % scores 1.0; a drop equal to ``MAX_HANDLE_DROP`` scores 0.0.

    Args:
        df: DataFrame with at least ``HANDLE_WINDOW`` rows and ``High``/``Low`` columns.
    """
    if len(df) < HANDLE_WINDOW:
        return 0.0
    handle_df = df.iloc[-HANDLE_WINDOW:]
    handle_high = handle_df["High"].max()
    handle_low = handle_df["Low"].min()
    handle_drop = (handle_high - handle_low) / handle_high if handle_high > 0 else 1.0
    return float(max(0.0, 1.0 - handle_drop / MAX_HANDLE_DROP))


def calc_volume_pattern(df: pd.DataFrame) -> float:
    """Return the volume-pattern sub-score [0, 1].

    Adds 0.5 for volume contraction during the cup and 0.5 for a volume
    expansion on the most recent (breakout) day.

    Args:
        df: DataFrame with at least ``CUP_WINDOW`` rows and ``Volume``/``vol_ma``
            columns.
    """
    if len(df) < CUP_WINDOW:
        return 0.0
    cup_df = df.iloc[-CUP_WINDOW:]
    cup_rel_vol = (cup_df["Volume"] / cup_df["vol_ma"].replace(0, np.nan)).mean()
    last_vol_ma = df["vol_ma"].iloc[-1]
    last_rel_vol = df["Volume"].iloc[-1] / last_vol_ma if last_vol_ma > 0 else 0.0
    score = 0.0
    if cup_rel_vol < 1.0:
        score += 0.5
    if last_rel_vol >= BREAKOUT_VOL_RATIO:
        score += 0.5
    return float(score)


def calc_breakout_strength(df: pd.DataFrame) -> float:
    """Return the breakout-strength sub-score [0, 1]: relative volume on the last day.

    Relative volume of 2× (or more) gives a score of 1.0; lower values scale
    linearly down to 0.0.

    Args:
        df: DataFrame with ``Volume`` and ``vol_ma`` columns.
    """
    if df.empty:
        return 0.0
    last_vol_ma = df["vol_ma"].iloc[-1]
    last_rel_vol = df["Volume"].iloc[-1] / last_vol_ma if last_vol_ma > 0 else 0.0
    return float(min(1.0, last_rel_vol / 2.0))


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

    # ── Condition checks ──────────────────────────────────────────────────────
    if not is_valid_drawdown(df):
        return None

    if not has_sufficient_base(df):
        return None

    if not is_handle_valid(df):
        return None

    # ── Scoring ───────────────────────────────────────────────────────────────
    cup_shape = calc_cup_shape(df)
    handle_quality = calc_handle_quality(df)
    volume_pattern = calc_volume_pattern(df)
    breakout_strength = calc_breakout_strength(df)

    # Signal booleans derived from raw data (for the signals list)
    cup_df = df.iloc[-CUP_WINDOW:]
    cup_rel_vol = (cup_df["Volume"] / cup_df["vol_ma"].replace(0, np.nan)).mean()
    last_vol_ma = df["vol_ma"].iloc[-1]
    last_rel_vol = df["Volume"].iloc[-1] / last_vol_ma if last_vol_ma > 0 else 0.0
    volume_contraction = cup_rel_vol < 1.0
    breakout_vol = last_rel_vol >= BREAKOUT_VOL_RATIO

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
