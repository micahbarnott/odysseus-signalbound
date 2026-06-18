"""Tests for agents/argus/scan_registry.py."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.argus.scan_registry import (
    ScanSession,
    clear_registry,
    create_session,
    get_session,
)


@pytest.fixture(autouse=True)
def clean():
    yield
    clear_registry()


# ---------------------------------------------------------------------------
# create_session
# ---------------------------------------------------------------------------

class TestCreateSession:
    def test_returns_scan_session(self):
        s = create_session("example.com")
        assert isinstance(s, ScanSession)

    def test_target_stored(self):
        s = create_session("example.com")
        assert s.target == "example.com"

    def test_scan_id_prefixed(self):
        s = create_session("example.com")
        assert s.scan_id.startswith("argus_")

    def test_scan_id_has_hex_suffix(self):
        s = create_session("example.com")
        suffix = s.scan_id[len("argus_"):]
        assert len(suffix) == 8
        int(suffix, 16)  # raises ValueError if not hex

    def test_initial_status_running(self):
        s = create_session("example.com")
        assert s.status == "running"

    def test_initial_phase_subfinder(self):
        s = create_session("example.com")
        assert s.phase == "subfinder"

    def test_initial_summary_mentions_target(self):
        s = create_session("example.com")
        assert "example.com" in s.summary

    def test_started_at_recent(self):
        before = time.time()
        s = create_session("example.com")
        after = time.time()
        assert before <= s.started_at <= after

    def test_completed_at_none_initially(self):
        s = create_session("example.com")
        assert s.completed_at is None

    def test_counters_zero_initially(self):
        s = create_session("example.com")
        assert s.discovered_domains == 0
        assert s.live_count == 0
        assert s.port_count == 0

    def test_errors_empty_initially(self):
        s = create_session("example.com")
        assert s.errors == []

    def test_highlights_empty_initially(self):
        s = create_session("example.com")
        assert s.highlights == []

    def test_thread_none_initially(self):
        s = create_session("example.com")
        assert s.thread is None


class TestCreateSessionUniqueness:
    def test_two_sessions_different_ids(self):
        s1 = create_session("a.com")
        s2 = create_session("b.com")
        assert s1.scan_id != s2.scan_id

    def test_ten_sessions_all_unique(self):
        ids = {create_session(f"t{i}.com").scan_id for i in range(10)}
        assert len(ids) == 10

    def test_same_target_twice_different_ids(self):
        s1 = create_session("example.com")
        s2 = create_session("example.com")
        assert s1.scan_id != s2.scan_id


# ---------------------------------------------------------------------------
# get_session
# ---------------------------------------------------------------------------

class TestGetSession:
    def test_returns_created_session(self):
        s = create_session("example.com")
        found = get_session(s.scan_id)
        assert found is s

    def test_unknown_id_returns_none(self):
        assert get_session("argus_00000000") is None

    def test_empty_id_returns_none(self):
        assert get_session("") is None

    def test_two_sessions_retrievable_by_id(self):
        s1 = create_session("a.com")
        s2 = create_session("b.com")
        assert get_session(s1.scan_id) is s1
        assert get_session(s2.scan_id) is s2

    def test_mutation_visible_via_get(self):
        s = create_session("example.com")
        s.status = "complete"
        s.phase = "done"
        found = get_session(s.scan_id)
        assert found.status == "complete"
        assert found.phase == "done"


# ---------------------------------------------------------------------------
# clear_registry
# ---------------------------------------------------------------------------

class TestClearRegistry:
    def test_clear_makes_session_unreachable(self):
        s = create_session("example.com")
        clear_registry()
        assert get_session(s.scan_id) is None

    def test_clear_then_create_works(self):
        create_session("example.com")
        clear_registry()
        s2 = create_session("other.com")
        assert get_session(s2.scan_id) is s2


# ---------------------------------------------------------------------------
# ScanSession.elapsed()
# ---------------------------------------------------------------------------

class TestScanSessionElapsed:
    def test_elapsed_increases_while_running(self):
        s = create_session("example.com")
        e1 = s.elapsed()
        time.sleep(0.05)
        e2 = s.elapsed()
        assert e2 >= e1

    def test_elapsed_frozen_after_completion(self):
        s = create_session("example.com")
        s.completed_at = time.time()
        e1 = s.elapsed()
        time.sleep(0.05)
        e2 = s.elapsed()
        assert e1 == e2


# ---------------------------------------------------------------------------
# ScanSession.to_response()
# ---------------------------------------------------------------------------

class TestScanSessionToResponse:
    def test_always_includes_required_keys(self):
        s = create_session("example.com")
        r = s.to_response()
        for key in ("scan_id", "target", "status", "phase", "summary", "elapsed_seconds"):
            assert key in r, f"missing key: {key}"

    def test_scan_id_matches(self):
        s = create_session("example.com")
        assert s.to_response()["scan_id"] == s.scan_id

    def test_target_matches(self):
        s = create_session("example.com")
        assert s.to_response()["target"] == "example.com"

    def test_zero_counters_omitted(self):
        s = create_session("example.com")
        r = s.to_response()
        assert "discovered_domains" not in r
        assert "live_hosts" not in r
        assert "open_ports" not in r

    def test_nonzero_counters_included(self):
        s = create_session("example.com")
        s.discovered_domains = 47
        s.live_count = 12
        s.port_count = 84
        r = s.to_response()
        assert r["discovered_domains"] == 47
        assert r["live_hosts"] == 12
        assert r["open_ports"] == 84

    def test_errors_omitted_when_empty(self):
        s = create_session("example.com")
        assert "errors" not in s.to_response()

    def test_errors_included_when_present(self):
        s = create_session("example.com")
        s.errors = ["subfinder: binary not found"]
        assert s.to_response()["errors"] == ["subfinder: binary not found"]

    def test_highlights_omitted_when_empty(self):
        s = create_session("example.com")
        assert "highlights" not in s.to_response()

    def test_highlights_included_when_present(self):
        s = create_session("example.com")
        s.highlights = ["dev.example.com — dev surface exposed"]
        assert "highlights" in s.to_response()

    def test_report_field_not_in_response(self):
        s = create_session("example.com")
        s.report = {"some": "data"}
        assert "report" not in s.to_response()

    def test_thread_field_not_in_response(self):
        s = create_session("example.com")
        s.thread = threading.Thread(target=lambda: None)
        assert "thread" not in s.to_response()


# ---------------------------------------------------------------------------
# Thread safety (smoke test)
# ---------------------------------------------------------------------------

class TestScanSessionThreadSafety:
    def test_concurrent_creates_all_unique(self):
        results: list[str] = []
        lock = threading.Lock()

        def create_one():
            s = create_session("example.com")
            with lock:
                results.append(s.scan_id)

        threads = [threading.Thread(target=create_one) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 20
        assert len(set(results)) == 20  # all unique
