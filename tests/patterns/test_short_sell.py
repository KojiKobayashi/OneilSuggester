"""Unit tests for src/patterns/short_sell.py."""

import numpy as np
import pandas as pd
import pytest

from src.patterns.short_sell import (
    detect,
    has_cross_below,
    has_lower_highs,
    is_downtrend,
    is_rally_capped,
)


# ── Condition function tests ───────────────────────────────────────────────────

class TestIsDowntrend:
    def test_returns_true_when_ma25_below_ma75(self):
        assert is_downtrend(90.0, 100.0) is True

    def test_returns_false_when_ma25_equals_ma75(self):
        assert is_downtrend(100.0, 100.0) is False

    def test_returns_false_when_ma25_above_ma75(self):
        assert is_downtrend(110.0, 100.0) is False

    def test_small_difference(self):
        assert is_downtrend(99.99, 100.0) is True
        assert is_downtrend(100.01, 100.0) is False


class TestHasCrossBelow:
    def test_detects_cross(self):
        # ma25 was above ma75, then crossed below
        ma25 = np.array([105.0, 100.0, 95.0])
        ma75 = np.array([100.0, 100.0, 100.0])
        assert has_cross_below(ma25, ma75) is True

    def test_no_cross_always_below(self):
        ma25 = np.array([90.0, 89.0, 88.0])
        ma75 = np.array([100.0, 100.0, 100.0])
        assert has_cross_below(ma25, ma75) is False

    def test_no_cross_always_above(self):
        ma25 = np.array([110.0, 109.0, 108.0])
        ma75 = np.array([100.0, 100.0, 100.0])
        assert has_cross_below(ma25, ma75) is False

    def test_single_element_no_cross(self):
        assert has_cross_below(np.array([100.0]), np.array([100.0])) is False

    def test_cross_at_first_transition(self):
        ma25 = np.array([101.0, 99.0])
        ma75 = np.array([100.0, 100.0])
        assert has_cross_below(ma25, ma75) is True


class TestIsRallyCapped:
    def test_capped_when_high_below_ma25(self):
        highs = pd.Series([95.0, 96.0])
        ma25 = pd.Series([100.0, 100.0])
        assert is_rally_capped(highs, ma25) is True

    def test_capped_within_2_pct_tolerance(self):
        highs = pd.Series([101.5])
        ma25 = pd.Series([100.0])
        # 101.5 <= 100.0 * 1.02 = 102.0 → capped
        assert is_rally_capped(highs, ma25) is True

    def test_not_capped_when_high_exceeds_threshold(self):
        highs = pd.Series([103.0])
        ma25 = pd.Series([100.0])
        # 103.0 > 100.0 * 1.02 = 102.0 → not capped
        assert is_rally_capped(highs, ma25) is False

    def test_at_least_one_capped_returns_true(self):
        highs = pd.Series([103.0, 99.0, 110.0])
        ma25 = pd.Series([100.0, 100.0, 100.0])
        assert is_rally_capped(highs, ma25) is True


class TestHasLowerHighs:
    def test_lower_high_detected(self):
        highs = np.array([120.0, 115.0, 110.0])
        assert has_lower_highs(highs) is True

    def test_no_lower_high_rising(self):
        highs = np.array([100.0, 105.0, 110.0])
        assert has_lower_highs(highs) is False

    def test_equal_highs_no_lower_high(self):
        highs = np.array([100.0, 100.0])
        assert has_lower_highs(highs) is False

    def test_single_element_returns_false(self):
        assert has_lower_highs(np.array([100.0])) is False

    def test_empty_array_returns_false(self):
        assert has_lower_highs(np.array([])) is False


# ── detect() integration tests ─────────────────────────────────────────────────

def _make_short_df(n: int = 100, ma25_offset: float = -5.0) -> pd.DataFrame:
    """Build a DataFrame that satisfies the basic short-sell conditions.

    MA75 is set to 100, MA25 to 100 + *ma25_offset* (default -5, i.e. below MA75).
    """
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = np.full(n, 95.0)
    # Gradually declining highs so lower-highs condition is met
    highs = np.linspace(110.0, 90.0, n)
    df = pd.DataFrame(
        {
            "Close": close,
            "High": highs,
            "Low": close - 2,
            "Open": close,
            "Volume": np.full(n, 2000.0),
            "vol_ma": np.full(n, 1000.0),
            "ma25": np.full(n, 100.0 + ma25_offset),
            "ma75": np.full(n, 100.0),
        },
        index=idx,
    )
    # Create a cross-below event near the start so has_cross_below fires
    df.iloc[0, df.columns.get_loc("ma25")] = 101.0  # above ma75 at day 0
    df.iloc[1, df.columns.get_loc("ma25")] = 99.0   # below ma75 at day 1
    return df


class TestDetect:
    def test_returns_none_for_insufficient_rows(self):
        df = _make_short_df(n=50)
        assert detect(df) is None

    def test_returns_none_when_no_downtrend(self):
        df = _make_short_df(n=100, ma25_offset=5.0)  # MA25 > MA75
        assert detect(df) is None

    def test_detects_short_pattern(self):
        df = _make_short_df(n=100)
        result = detect(df)
        assert result is not None
        assert "score" in result
        assert "signals" in result
        assert 0.0 <= result["score"] <= 1.0

    def test_score_in_valid_range(self):
        df = _make_short_df(n=100)
        result = detect(df)
        assert result is not None
        assert 0.0 <= result["score"] <= 1.0

    def test_signals_contains_downtrend(self):
        df = _make_short_df(n=100)
        result = detect(df)
        assert result is not None
        assert "MA25 < MA75" in result["signals"]

    def test_returns_none_when_secondary_conditions_insufficient(self):
        """If fewer than 2 secondary conditions hold, detect returns None."""
        df = _make_short_df(n=100)
        # Force highs upward so lower_highs is False AND rally is not capped
        df["High"] = np.linspace(90.0, 110.0, len(df))  # rising highs
        df["ma25"] = 80.0  # MA25 well below highs → rally not capped
        # Remove cross-below signal
        df["ma25"] = 80.0
        result = detect(df)
        # With ma25 < ma75 already (80 < 100) but no secondary conditions:
        # lower_highs=False (rising), rally_cap=False (highs >> ma25*1.02),
        # cross_below may or may not fire — either way secondary_count < 2
        # so result should be None
        assert result is None
