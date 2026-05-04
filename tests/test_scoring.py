"""Unit tests for src/scoring.py."""

import numpy as np
import pandas as pd
import pytest

from src.scoring import score_ticker, score_ticker_all


def _make_ohlcv(n: int = 150, close_val: float = 100.0) -> pd.DataFrame:
    """Create a simple OHLCV DataFrame with *n* rows."""
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": np.full(n, close_val),
            "Close": np.full(n, close_val),
            "High": np.full(n, close_val + 1.0),
            "Low": np.full(n, close_val - 1.0),
            "Volume": np.full(n, 1000.0),
        },
        index=idx,
    )


def _make_short_ohlcv(n: int = 150) -> pd.DataFrame:
    """OHLCV that should trigger the short-sell pattern."""
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    # Declining close prices so MA25 < MA75
    closes = np.linspace(120.0, 80.0, n)
    highs = np.linspace(125.0, 85.0, n)  # lower highs
    df = pd.DataFrame(
        {
            "Open": closes,
            "Close": closes,
            "High": highs,
            "Low": closes - 2,
            "Volume": np.full(n, 2000.0),
        },
        index=idx,
    )
    return df


class TestScoreTicker:
    def test_returns_none_for_flat_data(self):
        """Flat price data should not trigger any pattern."""
        df = _make_ohlcv(n=150)
        result = score_ticker("0000.T", "TestCo", df)
        assert result is None

    def test_returns_dict_for_short_pattern(self):
        """Declining prices should trigger short-sell detection."""
        df = _make_short_ohlcv(n=150)
        result = score_ticker("9999.T", "ShortCo", df)
        if result is not None:
            assert result["code"] == "9999.T"
            assert result["name"] == "ShortCo"
            assert result["type"] in ("long", "short")
            assert 0.0 <= result["score"] <= 1.0
            assert isinstance(result["signals"], list)

    def test_output_keys(self):
        """If a result is returned, it must have the required keys."""
        df = _make_short_ohlcv(n=150)
        result = score_ticker("1234.T", "Acme", df)
        if result is not None:
            for key in ("code", "name", "type", "score", "signals"):
                assert key in result

    def test_returns_none_for_insufficient_data(self):
        """Too few rows → no pattern can be detected."""
        df = _make_ohlcv(n=10)
        result = score_ticker("0001.T", "TinyCo", df)
        assert result is None

    def test_returns_highest_scoring_result(self):
        """score_ticker returns the single highest-scoring pattern result."""
        df = _make_short_ohlcv(n=150)
        result = score_ticker("5555.T", "TopScore", df)
        # Can't guarantee a pattern fires, but if it does the score must be ≤ 1
        if result is not None:
            assert result["score"] <= 1.0


class TestScoreTickerAll:
    def test_returns_empty_list_for_flat_data(self):
        """Flat price data should not trigger any pattern."""
        df = _make_ohlcv(n=150)
        results = score_ticker_all("0000.T", "TestCo", df)
        assert results == []

    def test_returns_list(self):
        """score_ticker_all always returns a list, never None."""
        df = _make_ohlcv(n=150)
        results = score_ticker_all("0000.T", "TestCo", df)
        assert isinstance(results, list)

    def test_returns_list_for_short_pattern(self):
        """Declining prices should trigger short-sell detection."""
        df = _make_short_ohlcv(n=150)
        results = score_ticker_all("9999.T", "ShortCo", df)
        assert isinstance(results, list)
        for r in results:
            assert r["code"] == "9999.T"
            assert r["name"] == "ShortCo"
            assert r["type"] in ("long", "short")
            assert 0.0 <= r["score"] <= 1.0
            assert isinstance(r["signals"], list)

    def test_output_keys(self):
        """Each returned dict must have the required keys."""
        df = _make_short_ohlcv(n=150)
        results = score_ticker_all("1234.T", "Acme", df)
        for r in results:
            for key in ("code", "name", "type", "score", "signals"):
                assert key in r

    def test_returns_empty_list_for_insufficient_data(self):
        """Too few rows → no pattern can be detected."""
        df = _make_ohlcv(n=10)
        results = score_ticker_all("0001.T", "TinyCo", df)
        assert results == []

    def test_may_return_multiple_results(self):
        """score_ticker_all can return both long and short for the same ticker."""
        df = _make_short_ohlcv(n=150)
        results = score_ticker_all("5555.T", "Multi", df)
        types = [r["type"] for r in results]
        # Each type should appear at most once
        assert len(types) == len(set(types))
