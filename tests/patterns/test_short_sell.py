"""Unit tests for src/patterns/short_sell.py."""

import numpy as np
import pandas as pd
import pytest

from src.patterns.short_sell import (
    CROSS_LOOKBACK,
    LOWER_HIGHS_WINDOW,
    detect,
    has_cross_below,
    has_lower_highs,
    is_downtrend,
    is_rally_capped,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_df(
    n: int = 30,
    ma25: float = 95.0,
    ma75: float = 100.0,
    high: float = 98.0,
) -> pd.DataFrame:
    """Return a minimal DataFrame suitable for condition-function tests."""
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Close": np.full(n, 95.0),
            "High": np.full(n, high),
            "Low": np.full(n, 93.0),
            "Open": np.full(n, 95.0),
            "Volume": np.full(n, 1000.0),
            "vol_ma": np.full(n, 1000.0),
            "ma25": np.full(n, ma25),
            "ma75": np.full(n, ma75),
        },
        index=idx,
    )


# ── Condition function tests ───────────────────────────────────────────────────

class TestIsDowntrend:
    def test_returns_true_when_ma25_below_ma75(self):
        df = _make_df(ma25=90.0, ma75=100.0)
        assert is_downtrend(df) is True

    def test_returns_false_when_ma25_equals_ma75(self):
        df = _make_df(ma25=100.0, ma75=100.0)
        assert is_downtrend(df) is False

    def test_returns_false_when_ma25_above_ma75(self):
        df = _make_df(ma25=110.0, ma75=100.0)
        assert is_downtrend(df) is False

    def test_empty_valid_rows_returns_false(self):
        df = _make_df()
        df["ma25"] = np.nan
        df["ma75"] = np.nan
        assert is_downtrend(df) is False

    def test_small_difference(self):
        assert is_downtrend(_make_df(ma25=99.99, ma75=100.0)) is True
        assert is_downtrend(_make_df(ma25=100.01, ma75=100.0)) is False


class TestHasCrossBelow:
    def test_detects_cross(self):
        # Build a df where ma25 crosses below ma75 within the lookback window
        df = _make_df(n=CROSS_LOOKBACK + 5, ma25=90.0, ma75=100.0)
        # Day 0 of the lookback window: ma25 above ma75
        df.iloc[-(CROSS_LOOKBACK + 1), df.columns.get_loc("ma25")] = 101.0
        # Day 1: ma25 below ma75 → cross
        df.iloc[-CROSS_LOOKBACK, df.columns.get_loc("ma25")] = 99.0
        assert has_cross_below(df) is True

    def test_no_cross_always_below(self):
        df = _make_df(n=CROSS_LOOKBACK + 5, ma25=90.0, ma75=100.0)
        assert has_cross_below(df) is False

    def test_no_cross_always_above(self):
        df = _make_df(n=CROSS_LOOKBACK + 5, ma25=110.0, ma75=100.0)
        assert has_cross_below(df) is False

    def test_insufficient_rows_returns_false(self):
        df = _make_df(n=2, ma25=90.0, ma75=100.0)
        assert has_cross_below(df) is False


class TestIsRallyCapped:
    def test_capped_when_high_below_ma25(self):
        # high=95, ma25=100 → 95 ≤ 100*1.02=102 → capped
        df = _make_df(n=LOWER_HIGHS_WINDOW + 5, ma25=100.0, ma75=105.0, high=95.0)
        assert is_rally_capped(df) is True

    def test_capped_within_2_pct_tolerance(self):
        # high=101.5, ma25=100 → 101.5 ≤ 102.0 → capped
        df = _make_df(n=LOWER_HIGHS_WINDOW + 5, ma25=100.0, ma75=105.0, high=101.5)
        assert is_rally_capped(df) is True

    def test_not_capped_when_high_exceeds_threshold(self):
        # high=103, ma25=100 → 103 > 102 → not capped
        df = _make_df(n=LOWER_HIGHS_WINDOW + 5, ma25=100.0, ma75=105.0, high=103.0)
        assert is_rally_capped(df) is False

    def test_empty_returns_false(self):
        df = _make_df(n=5, ma25=100.0, ma75=100.0)
        df["ma25"] = np.nan
        df["ma75"] = np.nan
        assert is_rally_capped(df) is False


class TestHasLowerHighs:
    def test_lower_high_detected(self):
        df = _make_df(n=LOWER_HIGHS_WINDOW + 5, ma25=95.0, ma75=100.0)
        highs = np.linspace(120.0, 90.0, len(df))  # declining
        df["High"] = highs
        assert has_lower_highs(df) is True

    def test_no_lower_high_rising(self):
        df = _make_df(n=LOWER_HIGHS_WINDOW + 5, ma25=95.0, ma75=100.0)
        highs = np.linspace(90.0, 120.0, len(df))  # rising
        df["High"] = highs
        assert has_lower_highs(df) is False

    def test_equal_highs_no_lower_high(self):
        df = _make_df(n=LOWER_HIGHS_WINDOW + 5, ma25=95.0, ma75=100.0, high=100.0)
        assert has_lower_highs(df) is False

    def test_insufficient_valid_rows_returns_false(self):
        df = _make_df(n=1, ma25=95.0, ma75=100.0)
        assert has_lower_highs(df) is False


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
        df["ma25"] = 80.0  # MA25 well below highs so rally is not capped
        # lower_highs=False (rising), rally_cap=False (highs >> ma25*1.02),
        # cross_below may or may not fire — either way secondary_count < 2
        result = detect(df)
        assert result is None

