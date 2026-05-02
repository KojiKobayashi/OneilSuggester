"""Technical indicator calculations."""

import numpy as np
import pandas as pd


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Add MA5, MA25 and MA75 columns to *df* (in-place copy).

    Args:
        df: DataFrame with at least a ``Close`` column.

    Returns:
        A new DataFrame with additional ``ma5``, ``ma25`` and ``ma75`` columns.
    """
    out = df.copy()
    out["ma5"] = out["Close"].rolling(5).mean()
    out["ma25"] = out["Close"].rolling(25).mean()
    out["ma75"] = out["Close"].rolling(75).mean()
    return out


def add_volume_ma(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Add a rolling-average volume column ``vol_ma`` to *df*.

    Args:
        df: DataFrame with a ``Volume`` column.
        window: Rolling window size (default: 20 days).

    Returns:
        A new DataFrame with an additional ``vol_ma`` column.
    """
    out = df.copy()
    out["vol_ma"] = out["Volume"].rolling(window).mean()
    return out


def relative_volume(df: pd.DataFrame) -> pd.Series:
    """Return the ratio of daily volume to the 20-day average volume.

    Args:
        df: DataFrame that already contains ``Volume`` and ``vol_ma`` columns.

    Returns:
        A :class:`~pandas.Series` of relative-volume values.
    """
    return df["Volume"] / df["vol_ma"].replace(0, np.nan)
