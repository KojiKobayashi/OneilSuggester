"""Unit tests for src/patterns/cup_with_handle.py."""

import numpy as np
import pandas as pd
import pytest

from src.patterns.cup_with_handle import (
    CUP_WINDOW,
    HANDLE_WINDOW,
    MAX_DRAWDOWN,
    MAX_HANDLE_DROP,
    MIN_BASE_DAYS,
    MIN_DRAWDOWN,
    detect,
    has_sufficient_base,
    is_handle_valid,
    is_valid_drawdown,
)


# ── Shared synthetic-data builder ──────────────────────────────────────────────

def _make_cup_df(n: int = 140) -> pd.DataFrame:
    """Build a synthetic cup-with-handle DataFrame.

    Structure (within the last CUP_WINDOW=120 rows):
    - Left peak: price rises to 120 in the first 20 bars.
    - Cup descent: price falls to ~90 (~25 % drawdown) over 40 bars.
    - Base: wide base near the bottom for 20 bars.
    - Recovery: price rises back to ~115 over 20 bars.
    - Handle: tight consolidation in the last 20 bars (< 12 % drop).
    """
    idx = pd.date_range("2022-01-01", periods=n, freq="B")

    pre = np.full(n - CUP_WINDOW, 100.0)  # pre-cup baseline

    left_peak = np.linspace(100.0, 120.0, 20)
    descent = np.linspace(120.0, 90.0, 40)
    base = np.full(20, 91.0)
    recovery = np.linspace(91.0, 115.0, 20)
    handle = np.linspace(115.0, 112.0, 20)  # tight handle

    closes = np.concatenate([pre, left_peak, descent, base, recovery, handle])
    assert len(closes) == n, f"Expected {n} rows, got {len(closes)}"

    highs = closes + 1.0
    lows = closes - 1.0
    volume = np.full(n, 500.0)
    vol_ma = np.full(n, 600.0)  # cup_rel_vol < 1.0 → volume contraction
    volume[-1] = 700.0  # breakout volume spike

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


# ── Condition function tests ───────────────────────────────────────────────────

class TestIsValidDrawdown:
    def test_valid_drawdown_detected(self):
        df = _make_cup_df()
        assert is_valid_drawdown(df) is True

    def test_too_shallow_drawdown(self):
        df = _make_cup_df()
        # Flatten all close prices → ~0 % drawdown
        df["Close"] = 100.0
        assert is_valid_drawdown(df) is False

    def test_too_deep_drawdown(self):
        df = _make_cup_df()
        # Make the cup bottom extremely low → > 35 % drawdown
        cup_start = len(df) - CUP_WINDOW
        df.iloc[cup_start + 20 : cup_start + 80, df.columns.get_loc("Close")] = 50.0
        assert is_valid_drawdown(df) is False

    def test_insufficient_rows_returns_false(self):
        df = _make_cup_df()
        assert is_valid_drawdown(df.iloc[:CUP_WINDOW - 1]) is False


class TestHasSufficientBase:
    def test_sufficient_base_detected(self):
        df = _make_cup_df()
        assert has_sufficient_base(df) is True

    def test_insufficient_base(self):
        df = _make_cup_df()
        # Replace the wide base with a sharp V (only 1 day near the bottom)
        cup_start = len(df) - CUP_WINDOW
        closes = df["Close"].values.copy()
        closes[cup_start:] = np.linspace(120.0, 60.0, CUP_WINDOW)
        df["Close"] = closes
        assert has_sufficient_base(df) is False

    def test_insufficient_rows_returns_false(self):
        df = _make_cup_df()
        assert has_sufficient_base(df.iloc[:CUP_WINDOW - 1]) is False


class TestIsHandleValid:
    def test_valid_handle_detected(self):
        df = _make_cup_df()
        assert is_handle_valid(df) is True

    def test_invalid_handle_drop_too_large(self):
        df = _make_cup_df()
        # Widen the handle range to exceed MAX_HANDLE_DROP (12 %)
        df.iloc[-HANDLE_WINDOW:, df.columns.get_loc("High")] = 120.0
        df.iloc[-HANDLE_WINDOW:, df.columns.get_loc("Low")] = 100.0
        # handle_drop = (120 - 100) / 120 ≈ 0.167 > 0.12
        assert is_handle_valid(df) is False

    def test_insufficient_rows_returns_false(self):
        df = _make_cup_df()
        assert is_handle_valid(df.iloc[:HANDLE_WINDOW - 1]) is False


# ── detect() integration tests ─────────────────────────────────────────────────

class TestDetectCupWithHandle:
    def test_returns_none_for_insufficient_rows(self):
        df = _make_cup_df(n=140)
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
        df["Close"] = 100.0
        df["High"] = 101.0
        df["Low"] = 99.0
        assert detect(df) is None

    def test_returns_none_when_handle_drop_too_large(self):
        df = _make_cup_df(n=140)
        df.iloc[-HANDLE_WINDOW:, df.columns.get_loc("High")] = 120.0
        df.iloc[-HANDLE_WINDOW:, df.columns.get_loc("Low")] = 100.0
        assert detect(df) is None

