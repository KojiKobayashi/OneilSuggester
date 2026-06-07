"""CANSLIM-lite long ranking based on public price, volume, and optional fundamentals."""

from __future__ import annotations

from typing import Optional

import pandas as pd

ONE_YEAR_WINDOW = 252
RECENT_WINDOW = 20
BREAKOUT_VOLUME_RATIO = 1.5
HIGH_PROXIMITY_THRESHOLD = 0.95
MIN_RS_SCORE = 0.7

W_CURRENT_GROWTH = 0.2
W_ANNUAL_GROWTH = 0.15
W_NEW_HIGHS = 0.25
W_SUPPLY_DEMAND = 0.15
W_LEADER = 0.15
W_MARKET = 0.1


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _normalize_growth(growth: Optional[float], target: float = 0.25) -> Optional[float]:
    if growth is None:
        return None
    return _clamp(growth / target)


def is_uptrend(df: pd.DataFrame) -> bool:
    """Return True when price and moving averages show a healthy long trend."""
    valid = df.dropna(subset=["ma25", "ma75", "vol_ma"])
    if valid.empty:
        return False
    latest = valid.iloc[-1]
    return bool(
        latest["Close"] > latest["ma25"] > latest["ma75"]
        and valid["ma25"].iloc[-1] >= valid["ma25"].iloc[-5]
    )


def calc_new_high_score(df: pd.DataFrame) -> float:
    """Return a score based on 52-week-high proximity and breakout volume."""
    valid = df.dropna(subset=["vol_ma"])
    if len(valid) < RECENT_WINDOW:
        return 0.0
    window = valid.iloc[-min(len(valid), ONE_YEAR_WINDOW) :]
    latest = window.iloc[-1]
    high_52w = window["Close"].max()
    proximity = (latest["Close"] / high_52w) if high_52w > 0 else 0.0
    rel_vol = (latest["Volume"] / latest["vol_ma"]) if latest["vol_ma"] > 0 else 0.0
    proximity_score = _clamp((proximity - HIGH_PROXIMITY_THRESHOLD) / (1.0 - HIGH_PROXIMITY_THRESHOLD))
    breakout_score = _clamp(rel_vol / BREAKOUT_VOLUME_RATIO)
    return round(proximity_score * 0.7 + breakout_score * 0.3, 4)


def calc_supply_demand_score(df: pd.DataFrame) -> float:
    """Return a score based on recent relative volume and liquidity."""
    valid = df.dropna(subset=["vol_ma"])
    if len(valid) < RECENT_WINDOW:
        return 0.0
    recent = valid.iloc[-RECENT_WINDOW:]
    latest = recent.iloc[-1]
    rel_vol = (latest["Volume"] / latest["vol_ma"]) if latest["vol_ma"] > 0 else 0.0
    avg_dollar_volume = float((recent["Close"] * recent["Volume"]).mean())
    rel_vol_score = _clamp(rel_vol / 2.0)
    liquidity_score = _clamp(avg_dollar_volume / 5_000_000_000)
    return round(rel_vol_score * 0.6 + liquidity_score * 0.4, 4)


def calc_leader_score(rs_score: Optional[float]) -> Optional[float]:
    """Return the cross-sectional relative-strength score."""
    if rs_score is None:
        return None
    return _clamp(rs_score)


def calc_market_score(market_in_uptrend: bool) -> float:
    """Return the market-trend contribution."""
    return 1.0 if market_in_uptrend else 0.0


def _weighted_average(scores: list[tuple[Optional[float], float]]) -> float:
    active = [(score, weight) for score, weight in scores if score is not None]
    if not active:
        return 0.0
    total_weight = sum(weight for _, weight in active)
    return round(sum(score * weight for score, weight in active) / total_weight, 4)


def detect(
    df: pd.DataFrame,
    market_in_uptrend: bool,
    rs_score: Optional[float] = None,
    fundamentals: Optional[dict] = None,
) -> Optional[dict]:
    """Return a CANSLIM-lite long signal when public-data proxies are strong enough."""
    if len(df) < 75:
        return None
    if not market_in_uptrend or not is_uptrend(df):
        return None

    valid = df.dropna(subset=["vol_ma"])
    if valid.empty:
        return None
    latest = valid.iloc[-1]
    high_52w = valid.iloc[-min(len(valid), ONE_YEAR_WINDOW) :]["Close"].max()
    proximity = (latest["Close"] / high_52w) if high_52w > 0 else 0.0
    if proximity < HIGH_PROXIMITY_THRESHOLD:
        return None

    if rs_score is not None and rs_score < MIN_RS_SCORE:
        return None

    rel_vol = (latest["Volume"] / latest["vol_ma"]) if latest["vol_ma"] > 0 else 0.0
    if rel_vol < 1.0 and proximity < 0.99:
        return None

    fundamentals = fundamentals or {}
    current_growth = _normalize_growth(
        fundamentals.get("quarterly_eps_growth")
        or fundamentals.get("quarterly_revenue_growth")
    )
    annual_growth = _normalize_growth(
        fundamentals.get("annual_eps_growth")
        or fundamentals.get("annual_revenue_growth")
    )
    new_highs = calc_new_high_score(df)
    supply_demand = calc_supply_demand_score(df)
    leader = calc_leader_score(rs_score)
    market = calc_market_score(market_in_uptrend)

    score = _weighted_average(
        [
            (current_growth, W_CURRENT_GROWTH),
            (annual_growth, W_ANNUAL_GROWTH),
            (new_highs, W_NEW_HIGHS),
            (supply_demand, W_SUPPLY_DEMAND),
            (leader, W_LEADER),
            (market, W_MARKET),
        ]
    )

    signals = [
        "price above MA25/MA75",
        "near 52-week high",
    ]
    if rel_vol >= BREAKOUT_VOLUME_RATIO:
        signals.append("breakout volume")
    if rs_score is not None:
        signals.append(f"relative strength {rs_score:.0%}")
    if current_growth is not None and current_growth > 0:
        signals.append("current growth available")
    if annual_growth is not None and annual_growth > 0:
        signals.append("annual growth available")

    return {"score": score, "signals": signals}
