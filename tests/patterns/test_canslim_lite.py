"""Unit tests for CANSLIM-lite scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import indicators
from src.patterns import canslim_lite


def _make_canslim_df(n: int = 260) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    base = np.linspace(100.0, 180.0, n)
    close = base.copy()
    close[-20:] = np.linspace(170.0, 190.0, 20)
    high = close + 2.0
    low = close - 2.0
    volume = np.full(n, 1_000_000.0)
    volume[-1] = 2_000_000.0
    df = pd.DataFrame(
        {
            "Open": close,
            "Close": close,
            "High": high,
            "Low": low,
            "Volume": volume,
        },
        index=idx,
    )
    df = indicators.add_moving_averages(df)
    df = indicators.add_volume_ma(df)
    return df


class TestCanSlimLite:
    def test_detects_public_data_proxy_long(self):
        df = _make_canslim_df()
        result = canslim_lite.detect(
            df,
            market_in_uptrend=True,
            rs_score=0.9,
            fundamentals={"quarterly_revenue_growth": 0.3},
        )
        assert result is not None
        assert 0.0 <= result["score"] <= 1.0
        assert "near 52-week high" in result["signals"]

    def test_requires_market_uptrend(self):
        df = _make_canslim_df()
        result = canslim_lite.detect(df, market_in_uptrend=False, rs_score=0.9)
        assert result is None

    def test_requires_high_proximity(self):
        df = _make_canslim_df()
        df["Close"] = 100.0
        df = indicators.add_moving_averages(df)
        df = indicators.add_volume_ma(df)
        result = canslim_lite.detect(df, market_in_uptrend=True, rs_score=0.9)
        assert result is None
