"""Unit tests for src/indicators.py."""

import numpy as np
import pandas as pd
import pytest

from src.indicators import add_moving_averages, add_volume_ma, relative_volume


def _make_close_df(closes):
    """Build a minimal OHLCV-like DataFrame from a list of Close prices."""
    return pd.DataFrame({"Close": closes, "Volume": [1_000] * len(closes)})


class TestAddMovingAverages:
    def test_columns_added(self):
        df = _make_close_df([100.0] * 100)
        result = add_moving_averages(df)
        assert "ma5" in result.columns
        assert "ma25" in result.columns
        assert "ma75" in result.columns

    def test_original_unchanged(self):
        df = _make_close_df([100.0] * 100)
        add_moving_averages(df)
        assert "ma5" not in df.columns

    def test_ma5_value(self):
        closes = list(range(1, 11))  # 1, 2, …, 10
        df = _make_close_df(closes)
        result = add_moving_averages(df)
        # ma5 for the last row = mean(6,7,8,9,10) = 8.0
        assert result["ma5"].iloc[-1] == pytest.approx(8.0)

    def test_ma25_nan_for_first_rows(self):
        df = _make_close_df([100.0] * 30)
        result = add_moving_averages(df)
        # First 24 rows (indices 0-23) should be NaN, index 24 is first valid
        assert result["ma25"].iloc[:24].isna().all()
        assert not np.isnan(result["ma25"].iloc[24])

    def test_ma75_nan_until_sufficient_rows(self):
        df = _make_close_df([100.0] * 80)
        result = add_moving_averages(df)
        # First 74 rows (indices 0-73) should be NaN, index 74 is first valid
        assert result["ma75"].iloc[:74].isna().all()
        assert not np.isnan(result["ma75"].iloc[74])

    def test_constant_series(self):
        df = _make_close_df([50.0] * 100)
        result = add_moving_averages(df)
        assert result["ma5"].dropna().eq(50.0).all()
        assert result["ma25"].dropna().eq(50.0).all()
        assert result["ma75"].dropna().eq(50.0).all()


class TestAddVolumeMa:
    def test_column_added(self):
        df = pd.DataFrame({"Volume": [1000] * 30})
        result = add_volume_ma(df)
        assert "vol_ma" in result.columns

    def test_original_unchanged(self):
        df = pd.DataFrame({"Volume": [1000] * 30})
        add_volume_ma(df)
        assert "vol_ma" not in df.columns

    def test_constant_volume(self):
        df = pd.DataFrame({"Volume": [500] * 30})
        result = add_volume_ma(df)
        assert result["vol_ma"].dropna().eq(500.0).all()

    def test_custom_window(self):
        volumes = list(range(1, 11))
        df = pd.DataFrame({"Volume": volumes})
        result = add_volume_ma(df, window=5)
        # Last row: mean(6,7,8,9,10) = 8.0
        assert result["vol_ma"].iloc[-1] == pytest.approx(8.0)


class TestRelativeVolume:
    def test_relative_volume_equals_one(self):
        df = pd.DataFrame({"Volume": [100.0] * 30, "vol_ma": [100.0] * 30})
        rv = relative_volume(df)
        assert rv.dropna().eq(1.0).all()

    def test_relative_volume_doubled(self):
        df = pd.DataFrame({"Volume": [200.0] * 30, "vol_ma": [100.0] * 30})
        rv = relative_volume(df)
        assert rv.dropna().eq(2.0).all()

    def test_zero_vol_ma_returns_nan(self):
        df = pd.DataFrame({"Volume": [100.0], "vol_ma": [0.0]})
        rv = relative_volume(df)
        assert np.isnan(rv.iloc[0])
