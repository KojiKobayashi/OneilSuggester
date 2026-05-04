"""Short-sell (bearish) pattern detection and sub-score calculation.

Detection criteria
------------------
* MA25 < MA75  (primary downtrend filter).
* MA25 was crossed downward within the last ``CROSS_LOOKBACK`` (10) days.
* At least one recent rally high was *capped* at or below MA25
  (rally was rejected by MA25).
* Lower highs: the most recent swing high is below the previous one
  (confirmed by ``LOWER_HIGHS_WINDOW`` (20) days).

Scoring weights (short signal)
--------------------------------
  score = downtrend_strength * 0.4
        + rally_weakness    * 0.3
        + volume_spike      * 0.3
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Detection constants ────────────────────────────────────────────────────────
CROSS_LOOKBACK = 10       # days to look back for MA25 cross below
LOWER_HIGHS_WINDOW = 20   # days used to check lower-highs structure

# ── Score weights ──────────────────────────────────────────────────────────────
W_DOWNTREND = 0.4
W_RALLY_WEAKNESS = 0.3
W_VOLUME_SPIKE = 0.3


def is_downtrend(df: pd.DataFrame) -> bool:
    """Return True when the latest MA25 is below MA75 (primary downtrend condition).

    Args:
        df: DataFrame with columns ``ma25`` and ``ma75``.
    """
    valid = df.dropna(subset=["ma25", "ma75"])
    if valid.empty:
        return False
    latest = valid.iloc[-1]
    return float(latest["ma25"]) < float(latest["ma75"])


def has_cross_below(df: pd.DataFrame) -> bool:
    """Return True when MA25 crossed below MA75 within the last CROSS_LOOKBACK days.

    Args:
        df: DataFrame with columns ``ma25`` and ``ma75``.
    """
    valid = df.dropna(subset=["ma25", "ma75"])
    if len(valid) < CROSS_LOOKBACK + 1:
        return False
    recent = valid.iloc[-CROSS_LOOKBACK - 1:]
    ma25 = recent["ma25"].values
    ma75 = recent["ma75"].values
    return any(
        ma25[i - 1] >= ma75[i - 1] and ma25[i] < ma75[i]
        for i in range(1, len(ma25))
    )


def is_rally_capped(df: pd.DataFrame) -> bool:
    """Return True when at least one High in the recent window is at or below MA25 * 1.02.

    Args:
        df: DataFrame with columns ``High``, ``ma25``, and ``ma75``.
    """
    valid = df.dropna(subset=["ma25", "ma75"])
    recent20 = valid.iloc[-LOWER_HIGHS_WINDOW:]
    if recent20.empty:
        return False
    return bool((recent20["High"] <= recent20["ma25"] * 1.02).any())


def has_lower_highs(df: pd.DataFrame) -> bool:
    """Return True when the most recent High is below the earliest High in the recent window.

    Args:
        df: DataFrame with columns ``High``, ``ma25``, and ``ma75``.
    """
    valid = df.dropna(subset=["ma25", "ma75"])
    recent20 = valid.iloc[-LOWER_HIGHS_WINDOW:]
    highs = recent20["High"].values
    return bool(len(highs) >= 2 and highs[-1] < highs[0])


def detect(df: pd.DataFrame) -> Optional[dict]:
    """Detect a short-sell pattern in *df* and return a result dict.

    Args:
        df: DataFrame with columns ``Close``, ``High``, ``Volume``,
            ``vol_ma``, ``ma25``, ``ma75``.  Must contain at least 75 rows
            (required for MA75).

    Returns:
        A dict with keys ``score`` (float 0–1) and ``signals`` (list of str)
        when a pattern is found, otherwise ``None``.
    """
    if len(df) < 75:
        return None

    # Drop rows where MAs are NaN (first 74 rows)
    valid = df.dropna(subset=["ma25", "ma75"])
    if len(valid) < CROSS_LOOKBACK + 5:
        return None

    # ── Condition 1: MA25 < MA75 (downtrend) ─────────────────────────────────
    if not is_downtrend(df):
        return None

    # ── Condition 2: MA25 crossed below recently ─────────────────────────────
    cross_below = has_cross_below(df)

    # ── Condition 3: Rally capped at MA25 ────────────────────────────────────
    rally_cap = is_rally_capped(df)

    # ── Condition 4: Lower highs ──────────────────────────────────────────────
    lower_highs = has_lower_highs(df)

    # At least two of the three secondary conditions must hold
    secondary_count = sum([cross_below, rally_cap, lower_highs])
    if secondary_count < 2:
        return None

    # ── Sub-scores ────────────────────────────────────────────────────────────
    latest = valid.iloc[-1]
    recent20 = valid.iloc[-LOWER_HIGHS_WINDOW:]

    # downtrend_strength: gap between MA25 and MA75 relative to MA75
    ma_gap = (latest["ma75"] - latest["ma25"]) / latest["ma75"]
    downtrend_strength = min(1.0, ma_gap / 0.05)  # normalise to ~5 % gap = 1.0

    # rally_weakness: fraction of recent days where Close ≤ MA25
    rally_weakness = float((recent20["Close"] <= recent20["ma25"]).mean())

    # volume_spike: latest relative volume (bearish volume confirmation)
    last_vol_ma = df["vol_ma"].iloc[-1]
    last_vol = df["Volume"].iloc[-1]
    rel_vol = (last_vol / last_vol_ma) if last_vol_ma > 0 else 0.0
    volume_spike = min(1.0, rel_vol / 2.0)

    score = (
        downtrend_strength * W_DOWNTREND
        + rally_weakness * W_RALLY_WEAKNESS
        + volume_spike * W_VOLUME_SPIKE
    )

    signals: list[str] = ["MA25 < MA75"]
    if cross_below:
        signals.append("MA25 crossed below MA75")
    if rally_cap:
        signals.append("rally capped at MA25")
    if lower_highs:
        signals.append("lower highs")

    return {"score": round(float(score), 4), "signals": signals}
