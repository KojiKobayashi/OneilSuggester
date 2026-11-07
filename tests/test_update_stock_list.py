"""Unit tests for batch/update_stock_list.py."""

from __future__ import annotations

import os
import sys
import urllib.error

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PROJECT_ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from batch import update_stock_list  # noqa: E402
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


def test_parse_jpx_excel_keeps_alphanumeric_domestic_codes(monkeypatch):
    sample = pd.DataFrame(
        {
            "コード": ["130A", "275A", "1489", "ABC1"],
            "銘柄名": ["Alpha", "Beta", "ETF", "Invalid"],
            "市場・商品区分": [
                "プライム（内国株式）",
                "グロース（内国株式）",
                "ETF・ETN",
                "スタンダード（内国株式）",
            ],
        }
    )
    monkeypatch.setattr(pd, "read_excel", lambda *args, **kwargs: sample)

    result = parse_jpx_excel(b"dummy")

    assert result["code"].tolist() == ["130A.T", "275A.T"]
    assert result["name"].tolist() == ["Alpha", "Beta"]


def test_parse_jpx_excel_uses_primary_code_column(monkeypatch):
    sample = pd.DataFrame(
        {
            "コード": ["1301", "130A"],
            "銘柄名": ["Prime", "Growth"],
            "市場・商品区分": ["プライム（内国株式）", "グロース（内国株式）"],
            "33業種コード": ["50", "10"],
            "17業種コード": ["1", "2"],
        }
    )
    monkeypatch.setattr(pd, "read_excel", lambda *args, **kwargs: sample)

    result = parse_jpx_excel(b"dummy")

    assert result["code"].tolist() == ["1301.T", "130A.T"]
    assert result["name"].tolist() == ["Prime", "Growth"]


def test_resolve_jpx_data_url_picks_data_file_link(monkeypatch):
    html = b"""
    <html>
      <body>
        <a href="/markets/statistics-equities/misc/newtoken-att/data_j.xlsx">download</a>
      </body>
    </html>
    """
    monkeypatch.setattr(update_stock_list, "_download", lambda *_args, **_kwargs: html)

    result = update_stock_list.resolve_jpx_data_url()

    assert result.endswith("/markets/statistics-equities/misc/newtoken-att/data_j.xlsx")


def test_fetch_jpx_excel_retries_with_resolved_url_on_default_404(monkeypatch):
    fallback_url = "https://www.jpx.co.jp/markets/statistics-equities/misc/newtoken-att/data_j.xls"
    calls: list[str] = []

    def fake_download(url: str) -> bytes:
        calls.append(url)
        if url == update_stock_list.JPX_DATA_URL:
            raise urllib.error.HTTPError(url, 404, "Not Found", hdrs=None, fp=None)
        if url == update_stock_list.JPX_LIST_PAGE_URL:
            return (
                b'<a href="/markets/statistics-equities/misc/newtoken-att/data_j.xls">'
                b"download</a>"
            )
        if url == fallback_url:
            return b"excel-bytes"
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(update_stock_list, "_download", fake_download)

    result = update_stock_list.fetch_jpx_excel(update_stock_list.JPX_DATA_URL)

    assert result == b"excel-bytes"
    assert calls == [
        update_stock_list.JPX_DATA_URL,
        update_stock_list.JPX_LIST_PAGE_URL,
        fallback_url,
    ]
