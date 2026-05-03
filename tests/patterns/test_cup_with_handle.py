"""Unit tests for src/patterns/cup_with_handle.py."""

import numpy as np
import pandas as pd
import pytest

from src.patterns.cup_with_handle import (
    CUP_WINDOW,
    MAX_DRAWDOWN,
    MAX_HANDLE_DROP,
    MIN_BASE_DAYS,
    MIN_DRAWDOWN,
    detect,
    has_sufficient_base,
    is_handle_valid,
    is_valid_drawdown,
)


# ── Condition function tests ───────────────────────────────────────────────────

class TestIsValidDrawdown:
    def test_valid_lower_bound(self):
        assert is_valid_drawdown(MIN_DRAWDOWN) is True

    def test_valid_upper_bound(self):
        assert is_valid_drawdown(MAX_DRAWDOWN) is True

    def test_valid_midpoint(self):
        assert is_valid_drawdown(0.20) is True

    def test_too_shallow(self):
        assert is_valid_drawdown(MIN_DRAWDOWN - 0.01) is False

    def test_too_deep(self):
        assert is_valid_drawdown(MAX_DRAWDOWN + 0.01) is False

    def test_zero_drawdown_invalid(self):
        assert is_valid_drawdown(0.0) is False


class TestHasSufficientBase:
    def _make_cup_df(self, closes):
        return pd.DataFrame({"Close": closes})

    def test_sufficient_base(self):
        bottom = 80.0
        # All prices within 10 % of bottom → more than MIN_BASE_DAYS days
        closes = [bottom * 1.05] * (MIN_BASE_DAYS + 5)
        df = self._make_cup_df(closes)
        assert has_sufficient_base(df, bottom) is True

    def test_insufficient_base(self):
        bottom = 80.0
        # Only a few prices near bottom, rest are far above
        closes = [bottom * 1.05] * 5 + [bottom * 2.0] * 100
        df = self._make_cup_df(closes)
        assert has_sufficient_base(df, bottom) is False

    def test_boundary_exactly_min_base_days(self):
        bottom = 100.0
        closes = [bottom * 1.05] * MIN_BASE_DAYS
        df = self._make_cup_df(closes)
        assert has_sufficient_base(df, bottom) is True


class TestIsHandleValid:
    def test_valid_small_drop(self):
        assert is_handle_valid(0.05) is True

    def test_valid_at_maximum(self):
        assert is_handle_valid(MAX_HANDLE_DROP) is True

    def test_invalid_exceeds_maximum(self):
        assert is_handle_valid(MAX_HANDLE_DROP + 0.01) is False

    def test_zero_drop_valid(self):
        assert is_handle_valid(0.0) is True


# ── detect() integration tests ─────────────────────────────────────────────────

def _make_cup_df(n: int = 140) -> pd.DataFrame:
    """Build a synthetic cup-with-handle DataFrame.

    Structure (within the last CUP_WINDOW=120 rows):
    - Left peak: price rises to 120 in the first 30 bars.
    - Cup bottom: price falls to ~90 (25 % drawdown) over the next 60 bars.
    - Recovery: price rises back to ~115 in the remaining bars.
    - Handle: tight consolidation in the last 20 bars.
    """
    idx = pd.date_range("2022-01-01", periods=n, freq="B")

    # Build close prices with a clear cup shape in the last 120 bars
    pre = np.full(n - CUP_WINDOW, 100.0)  # pre-cup baseline

    left_peak = np.linspace(100.0, 120.0, 20)
    descent = np.linspace(120.0, 90.0, 40)
    base = np.full(20, 91.0)             # wide base near bottom
    recovery = np.linspace(91.0, 115.0, 20)
    handle = np.linspace(115.0, 112.0, 20)  # tight handle (< 12 % drop)

    closes = np.concatenate([pre, left_peak, descent, base, recovery, handle])
    assert len(closes) == n, f"Expected {n} rows, got {len(closes)}"

    highs = closes + 1.0
    lows = closes - 1.0
    volume = np.full(n, 500.0)
    vol_ma = np.full(n, 600.0)  # cup_rel_vol < 1.0 → volume contraction
    # Last bar: volume spike for breakout
    volume[-1] = 700.0
    vol_ma[-1] = 600.0  # rel_vol = 700/600 ≈ 1.17 ≥ 1.0

    df = pd.DataFrame(
        {
            "Close": closes,
            "High": highs,
            "Low": lows,
            "Open": closes,
            "Volume": volume,
            "vol_ma": vol_ma,
            "ma25": closes - 2,
            "ma75": closes - 5,
        },
        index=idx,
    )
    return df


class TestDetectCupWithHandle:
    def test_returns_none_for_insufficient_rows(self):
        df = _make_cup_df(n=140)
        # Slice to fewer than CUP_WINDOW rows
        assert detect(df.iloc[:CUP_WINDOW - 1]) is None

    def test_detects_cup_pattern(self):
        df = _make_cup_df(n=140)
        result = detect(df)
        assert result is not None
        assert "score" in result
        assert "signals" in result

    def test_score_in_valid_range(self):
        df = _make_cup_df(n=140)
        result = detect(df)
        assert result is not None
        assert 0.0 <= result["score"] <= 1.0

    def test_signals_contains_cup_and_handle(self):
        df = _make_cup_df(n=140)
        result = detect(df)
        assert result is not None
        assert "cup detected" in result["signals"]
        assert "handle formed" in result["signals"]

    def test_returns_none_when_drawdown_too_shallow(self):
        df = _make_cup_df(n=140)
        # Force all close prices flat → drawdown near 0
        df["Close"] = 100.0
        df["High"] = 101.0
        df["Low"] = 99.0
        assert detect(df) is None

    def test_returns_none_when_handle_drop_too_large(self):
        df = _make_cup_df(n=140)
        # Widen the handle range to exceed MAX_HANDLE_DROP
        df.iloc[-20:, df.columns.get_loc("High")] = 120.0
        df.iloc[-20:, df.columns.get_loc("Low")] = 100.0
        # handle_drop = (120 - 100) / 120 ≈ 0.167 > 0.12
        assert detect(df) is None
