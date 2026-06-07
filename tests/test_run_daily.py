"""Unit tests for batch/run_daily.py."""

from __future__ import annotations

import json
import os
import sys

import pytest

# Ensure the project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from batch.run_daily import (  # noqa: E402
    LEGACY_LATEST_FILENAME,
    MAX_DATES_IN_INDEX,
    remove_legacy_latest_output,
    update_index,
)


class TestUpdateIndex:
    def test_creates_index_file_when_missing(self, tmp_path):
        """update_index should create index.json if it does not exist."""
        update_index(str(tmp_path), "2026-05-01")
        index_path = tmp_path / "index.json"
        assert index_path.exists()
        data = json.loads(index_path.read_text())
        assert data["dates"] == ["2026-05-01"]

    def test_adds_new_date(self, tmp_path):
        """update_index should append a new date to an existing index."""
        index_path = tmp_path / "index.json"
        index_path.write_text(json.dumps({"dates": ["2026-05-01"]}))
        update_index(str(tmp_path), "2026-05-02")
        data = json.loads(index_path.read_text())
        assert "2026-05-02" in data["dates"]
        assert "2026-05-01" in data["dates"]

    def test_does_not_duplicate_date(self, tmp_path):
        """Adding the same date twice should not create duplicates."""
        index_path = tmp_path / "index.json"
        index_path.write_text(json.dumps({"dates": ["2026-05-01"]}))
        update_index(str(tmp_path), "2026-05-01")
        data = json.loads(index_path.read_text())
        assert data["dates"].count("2026-05-01") == 1

    def test_dates_sorted_descending(self, tmp_path):
        """Dates should be ordered newest-first."""
        index_path = tmp_path / "index.json"
        index_path.write_text(json.dumps({"dates": ["2026-05-01", "2026-04-30"]}))
        update_index(str(tmp_path), "2026-05-02")
        data = json.loads(index_path.read_text())
        assert data["dates"] == sorted(data["dates"], reverse=True)

    def test_caps_at_max_dates(self, tmp_path):
        """Index should not grow beyond MAX_DATES_IN_INDEX entries."""
        dates = [f"2026-{m:02d}-{d:02d}" for m in range(1, 13) for d in range(1, 4)]
        # Generate enough dates to exceed the cap
        index_path = tmp_path / "index.json"
        index_path.write_text(json.dumps({"dates": dates[:MAX_DATES_IN_INDEX]}))
        update_index(str(tmp_path), "2027-01-01")
        data = json.loads(index_path.read_text())
        assert len(data["dates"]) <= MAX_DATES_IN_INDEX


class TestRemoveLegacyLatestOutput:
    def test_removes_legacy_latest_json(self, tmp_path):
        legacy_path = tmp_path / LEGACY_LATEST_FILENAME
        legacy_path.write_text(json.dumps({"items": []}))

        remove_legacy_latest_output(str(tmp_path))

        assert not legacy_path.exists()

    def test_ignores_missing_legacy_latest_json(self, tmp_path):
        remove_legacy_latest_output(str(tmp_path))

        assert not (tmp_path / LEGACY_LATEST_FILENAME).exists()
