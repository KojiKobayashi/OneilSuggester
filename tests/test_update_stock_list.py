"""Unit tests for batch/update_stock_list.py."""

from __future__ import annotations

import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PROJECT_ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from batch.update_stock_list import parse_jpx_excel  # noqa: E402


def test_parse_jpx_excel_keeps_domestic_equity_segments(monkeypatch):
    sample = pd.DataFrame(
        {
            "コード": ["1301", "9999", "1489", "1234"],
            "銘柄名": ["Prime", "Standard", "ETF", "Growth"],
            "市場・商品区分": [
                "プライム（内国株式）",
                "スタンダード（内国株式）",
                "ETF・ETN",
                "グロース（内国株式）",
            ],
        }
    )
    monkeypatch.setattr(pd, "read_excel", lambda *args, **kwargs: sample)

    result = parse_jpx_excel(b"dummy")

    assert result["code"].tolist() == ["1301.T", "9999.T", "1234.T"]
    assert result["name"].tolist() == ["Prime", "Standard", "Growth"]
