"""Tests for mcp_servers/argus_server.py.

Design
------
* pytest.importorskip("mcp") — skips when mcp isn't installed, matching other
  MCP server tests (test_rag_server_*).
* All tests monkeypatch the module-level _agent / _initialized flags so no
  real binary is invoked.
* Background threads are joined via session.thread before asserting on
  completion state — deterministic, no sleeps.
* asyncio.run() — no pytest-asyncio dependency.
* clear_registry() fixture ensures no state leaks between tests.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("mcp")

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import mcp_servers.argus_server as _srv

from agents.argus.models import ReconHost, ReconPort, ReconReport
from agents.argus.scan_registry import clear_registry, get_session
from tools.nmap.nmap_tool import NmapHost, NmapPort, NmapRunStats, NmapScanResult, NmapService
from tools.subfinder.subfinder_tool import SubfinderDomain, SubfinderScanResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean():
    yield
    clear_registry()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    return asyncio.run(coro)


def _decode(contents) -> dict:
    assert len(contents) == 1
    return json.loads(contents[0].text)


def _patch_agent(monkeypatch, mock):
    monkeypatch.setattr(_srv, "_agent", mock)
    monkeypatch.setattr(_srv, "_initialized", True)


def _join_scan(scan_id: str, timeout: float = 5.0) -> None:
    """Wait for the background thread stored on a session to finish."""
    s = get_session(scan_id)
    if s and s.thread:
        s.thread.join(timeout=timeout)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def _make_report(
    target="example.com",
    domains=("a.example.com", "b.example.com"),
    hosts=(),
    errors=(),
) -> ReconReport:
    return ReconReport(
        target=target,
        discovered_domains=list(domains),
        live_hosts=list(hosts),
        errors=list(errors),
    )


def _make_nmap_ok(ports=((80, "tcp", "http"),)) -> NmapScanResult:
    nmap_ports = [
        NmapPort(port=p[0], protocol=p[1], state="open",
                 service=NmapService(name=p[2]) if len(p) > 2 else None)
        for p in ports
    ]
    host = NmapHost(address="1.2.3.4", address_type="ipv4", state="up", ports=nmap_ports)
    return NmapScanResult(
        command="nmap ...",
        hosts=[host],
        stats=NmapRunStats(elapsed=1.0, hosts_up=1, hosts_down=0, hosts_total=1),
    )


def _make_sf_ok(hosts=("a.example.com",)) -> SubfinderScanResult:
    return SubfinderScanResult(
        command="subfinder ...",
        domains=[SubfinderDomain(host=h) for h in hosts],
    )


# ---------------------------------------------------------------------------
# list_tools
# ---------------------------------------------------------------------------

class TestListTools:
    def test_returns_four_tools(self):
        tools = _run(_srv.list_tools())
        assert len(tools) == 4

    def test_tool_names(self):
        tools = _run(_srv.list_tools())
        names = {t.name for t in tools}
        assert names == {
            "argus_recon",
            "argus_scan_host",
            "argus_enumerate_subdomains",
            "argus_get_scan",
        }

    def test_each_tool_has_required_field(self):
        tools = _run(_srv.list_tools())
        for t in tools:
            assert t.inputSchema.get("required"), f"{t.name} missing required fields"

    def test_argus_recon_requires_target(self):
        tools = _run(_srv.list_tools())
        t = next(x for x in tools if x.name == "argus_recon")
        assert "target" in t.inputSchema["required"]

    def test_argus_scan_host_requires_target(self):
        tools = _run(_srv.list_tools())
        t = next(x for x in tools if x.name == "argus_scan_host")
        assert "target" in t.inputSchema["required"]

    def test_argus_enumerate_requires_domain(self):
        tools = _run(_srv.list_tools())
        t = next(x for x in tools if x.name == "argus_enumerate_subdomains")
        assert "domain" in t.inputSchema["required"]

    def test_argus_get_scan_requires_scan_id(self):
        tools = _run(_srv.list_tools())
        t = next(x for x in tools if x.name == "argus_get_scan")
        assert "scan_id" in t.inputSchema["required"]


# ---------------------------------------------------------------------------
# argus_recon — start behavior
# ---------------------------------------------------------------------------

class TestArgusReconStart:
    def test_returns_scan_id(self, monkeypatch):
        agent = MagicMock()
        agent.run.return_value = _make_report()
        _patch_agent(monkeypatch, agent)
        result = _decode(_run(_srv.call_tool("argus_recon", {"target": "example.com"})))
        assert result["scan_id"].startswith("argus_")

    def test_initial_status_running(self, monkeypatch):
        agent = MagicMock()
        agent.run.return_value = _make_report()
        _patch_agent(monkeypatch, agent)
        result = _decode(_run(_srv.call_tool("argus_recon", {"target": "example.com"})))
        assert result["status"] == "running"

    def test_target_in_response(self, monkeypatch):
        agent = MagicMock()
        agent.run.return_value = _make_report()
        _patch_agent(monkeypatch, agent)
        result = _decode(_run(_srv.call_tool("argus_recon", {"target": "example.com"})))
        assert result["target"] == "example.com"

    def test_missing_target_returns_error(self, monkeypatch):
        agent = MagicMock()
        _patch_agent(monkeypatch, agent)
        result = _decode(_run(_srv.call_tool("argus_recon", {})))
        assert "error" in result
        agent.run.assert_not_called()

    def test_empty_target_returns_error(self, monkeypatch):
        agent = MagicMock()
        _patch_agent(monkeypatch, agent)
        result = _decode(_run(_srv.call_tool("argus_recon", {"target": "   "})))
        assert "error" in result
        agent.run.assert_not_called()

    def test_two_scans_have_different_ids(self, monkeypatch):
        agent = MagicMock()
        agent.run.return_value = _make_report()
        _patch_agent(monkeypatch, agent)
        r1 = _decode(_run(_srv.call_tool("argus_recon", {"target": "a.com"})))
        r2 = _decode(_run(_srv.call_tool("argus_recon", {"target": "b.com"})))
        assert r1["scan_id"] != r2["scan_id"]


# ---------------------------------------------------------------------------
# argus_recon — completion via argus_get_scan
# ---------------------------------------------------------------------------

class TestArgusReconCompletion:
    def test_scan_completes(self, monkeypatch):
        agent = MagicMock()
        agent.run.return_value = _make_report()
        _patch_agent(monkeypatch, agent)
        r = _decode(_run(_srv.call_tool("argus_recon", {"target": "example.com"})))
        _join_scan(r["scan_id"])
        r2 = _decode(_run(_srv.call_tool("argus_get_scan", {"scan_id": r["scan_id"]})))
        assert r2["status"] == "complete"

    def test_agent_called_with_target(self, monkeypatch):
        agent = MagicMock()
        agent.run.return_value = _make_report()
        _patch_agent(monkeypatch, agent)
        r = _decode(_run(_srv.call_tool("argus_recon", {"target": "target.io"})))
        _join_scan(r["scan_id"])
        agent.run.assert_called_once()
        call_target = agent.run.call_args[0][0]
        assert call_target == "target.io"

    def test_agent_exception_results_in_failed_status(self, monkeypatch):
        agent = MagicMock()
        agent.run.side_effect = RuntimeError("unexpected crash")
        _patch_agent(monkeypatch, agent)
        r = _decode(_run(_srv.call_tool("argus_recon", {"target": "example.com"})))
        _join_scan(r["scan_id"])
        r2 = _decode(_run(_srv.call_tool("argus_get_scan", {"scan_id": r["scan_id"]})))
        assert r2["status"] == "failed"
        assert any("unexpected crash" in e for e in r2.get("errors", []))


# ---------------------------------------------------------------------------
# argus_scan_host — start and completion
# ---------------------------------------------------------------------------

class TestArgusScanHost:
    def _run_with_nmap(self, monkeypatch, nm_result, target="1.2.3.4"):
        nmap_tool = MagicMock()
        nmap_tool.scan.return_value = nm_result
        import tools.nmap as _nmap_pkg
        monkeypatch.setattr(_nmap_pkg, "NmapTool", lambda: nmap_tool)
        return _decode(_run(_srv.call_tool("argus_scan_host", {"target": target})))

    def test_returns_scan_id(self, monkeypatch):
        r = self._run_with_nmap(monkeypatch, _make_nmap_ok())
        assert r["scan_id"].startswith("argus_")

    def test_initial_status_running(self, monkeypatch):
        r = self._run_with_nmap(monkeypatch, _make_nmap_ok())
        assert r["status"] == "running"

    def test_missing_target_returns_error(self, monkeypatch):
        result = _decode(_run(_srv.call_tool("argus_scan_host", {})))
        assert "error" in result

    def test_empty_target_returns_error(self, monkeypatch):
        result = _decode(_run(_srv.call_tool("argus_scan_host", {"target": ""})))
        assert "error" in result

    def test_scan_completes_with_port_count(self, monkeypatch):
        nmap_tool = MagicMock()
        nmap_tool.scan.return_value = _make_nmap_ok(ports=((80, "tcp", "http"), (443, "tcp", "https")))
        import tools.nmap as _nmap_pkg
        monkeypatch.setattr(_nmap_pkg, "NmapTool", lambda: nmap_tool)
        r = _decode(_run(_srv.call_tool("argus_scan_host", {"target": "1.2.3.4"})))
        _join_scan(r["scan_id"])
        r2 = _decode(_run(_srv.call_tool("argus_get_scan", {"scan_id": r["scan_id"]})))
        assert r2["status"] == "complete"
        assert r2["open_ports"] == 2

    def test_nmap_error_in_completed_scan(self, monkeypatch):
        nmap_tool = MagicMock()
        nmap_tool.scan.return_value = NmapScanResult(command="nmap", error="binary not found")
        import tools.nmap as _nmap_pkg
        monkeypatch.setattr(_nmap_pkg, "NmapTool", lambda: nmap_tool)
        r = _decode(_run(_srv.call_tool("argus_scan_host", {"target": "1.2.3.4"})))
        _join_scan(r["scan_id"])
        r2 = _decode(_run(_srv.call_tool("argus_get_scan", {"scan_id": r["scan_id"]})))
        assert r2["status"] == "complete"
        assert "errors" in r2

    def test_nmap_exception_results_in_failed_status(self, monkeypatch):
        nmap_tool = MagicMock()
        nmap_tool.scan.side_effect = RuntimeError("boom")
        import tools.nmap as _nmap_pkg
        monkeypatch.setattr(_nmap_pkg, "NmapTool", lambda: nmap_tool)
        r = _decode(_run(_srv.call_tool("argus_scan_host", {"target": "1.2.3.4"})))
        _join_scan(r["scan_id"])
        r2 = _decode(_run(_srv.call_tool("argus_get_scan", {"scan_id": r["scan_id"]})))
        assert r2["status"] == "failed"


# ---------------------------------------------------------------------------
# argus_enumerate_subdomains — start and completion
# ---------------------------------------------------------------------------

class TestArgusEnumerateSubdomains:
    def _run_with_sf(self, monkeypatch, sf_result, domain="example.com"):
        sf_tool = MagicMock()
        sf_tool.scan.return_value = sf_result
        import tools.subfinder as _sf_pkg
        monkeypatch.setattr(_sf_pkg, "SubfinderTool", lambda: sf_tool)
        return _decode(_run(_srv.call_tool("argus_enumerate_subdomains", {"domain": domain})))

    def test_returns_scan_id(self, monkeypatch):
        r = self._run_with_sf(monkeypatch, _make_sf_ok())
        assert r["scan_id"].startswith("argus_")

    def test_initial_status_running(self, monkeypatch):
        r = self._run_with_sf(monkeypatch, _make_sf_ok())
        assert r["status"] == "running"

    def test_missing_domain_returns_error(self, monkeypatch):
        result = _decode(_run(_srv.call_tool("argus_enumerate_subdomains", {})))
        assert "error" in result

    def test_empty_domain_returns_error(self, monkeypatch):
        result = _decode(_run(_srv.call_tool("argus_enumerate_subdomains", {"domain": ""})))
        assert "error" in result

    def test_scan_completes_with_domain_count(self, monkeypatch):
        sf = _make_sf_ok(hosts=["a.example.com", "b.example.com", "c.example.com"])
        r = self._run_with_sf(monkeypatch, sf)
        _join_scan(r["scan_id"])
        r2 = _decode(_run(_srv.call_tool("argus_get_scan", {"scan_id": r["scan_id"]})))
        assert r2["status"] == "complete"
        assert r2["discovered_domains"] == 3

    def test_subfinder_error_recorded(self, monkeypatch):
        sf = SubfinderScanResult(command="subfinder", error="binary not found")
        r = self._run_with_sf(monkeypatch, sf)
        _join_scan(r["scan_id"])
        r2 = _decode(_run(_srv.call_tool("argus_get_scan", {"scan_id": r["scan_id"]})))
        assert r2["status"] == "complete"
        assert "errors" in r2

    def test_domain_passed_to_subfinder(self, monkeypatch):
        sf_tool = MagicMock()
        sf_tool.scan.return_value = _make_sf_ok()
        import tools.subfinder as _sf_pkg
        monkeypatch.setattr(_sf_pkg, "SubfinderTool", lambda: sf_tool)
        r = _decode(_run(_srv.call_tool("argus_enumerate_subdomains", {"domain": "target.io"})))
        _join_scan(r["scan_id"])
        sf_tool.scan.assert_called_once_with("target.io")


# ---------------------------------------------------------------------------
# argus_get_scan
# ---------------------------------------------------------------------------

class TestArgusGetScan:
    def test_running_scan_returns_running_status(self, monkeypatch):
        agent = MagicMock()
        # Slow agent — still running when we poll
        import threading, time as _t
        barrier = threading.Barrier(2)
        def _slow_run(target, session=None):
            barrier.wait()   # sync with test thread
            return _make_report()
        agent.run.side_effect = _slow_run
        _patch_agent(monkeypatch, agent)
        r = _decode(_run(_srv.call_tool("argus_recon", {"target": "example.com"})))
        # Poll before barrier releases — scan is still running
        r2 = _decode(_run(_srv.call_tool("argus_get_scan", {"scan_id": r["scan_id"]})))
        assert r2["status"] == "running"
        barrier.wait()   # unblock the background thread
        _join_scan(r["scan_id"])

    def test_unknown_scan_id_returns_error(self, monkeypatch):
        result = _decode(_run(_srv.call_tool("argus_get_scan", {"scan_id": "argus_00000000"})))
        assert "error" in result
        assert "not found" in result["error"]

    def test_missing_scan_id_returns_error(self, monkeypatch):
        result = _decode(_run(_srv.call_tool("argus_get_scan", {})))
        assert "error" in result

    def test_response_includes_target(self, monkeypatch):
        agent = MagicMock()
        agent.run.return_value = _make_report()
        _patch_agent(monkeypatch, agent)
        r = _decode(_run(_srv.call_tool("argus_recon", {"target": "example.com"})))
        _join_scan(r["scan_id"])
        r2 = _decode(_run(_srv.call_tool("argus_get_scan", {"scan_id": r["scan_id"]})))
        assert r2["target"] == "example.com"

    def test_response_includes_elapsed_seconds(self, monkeypatch):
        agent = MagicMock()
        agent.run.return_value = _make_report()
        _patch_agent(monkeypatch, agent)
        r = _decode(_run(_srv.call_tool("argus_recon", {"target": "example.com"})))
        _join_scan(r["scan_id"])
        r2 = _decode(_run(_srv.call_tool("argus_get_scan", {"scan_id": r["scan_id"]})))
        assert "elapsed_seconds" in r2
        assert isinstance(r2["elapsed_seconds"], (int, float))

    def test_completed_scan_has_phase_done(self, monkeypatch):
        agent = MagicMock()
        agent.run.return_value = _make_report()
        _patch_agent(monkeypatch, agent)
        r = _decode(_run(_srv.call_tool("argus_recon", {"target": "example.com"})))
        _join_scan(r["scan_id"])
        r2 = _decode(_run(_srv.call_tool("argus_get_scan", {"scan_id": r["scan_id"]})))
        assert r2["phase"] == "done"

    def test_scan_id_stable_across_calls(self, monkeypatch):
        agent = MagicMock()
        agent.run.return_value = _make_report()
        _patch_agent(monkeypatch, agent)
        r = _decode(_run(_srv.call_tool("argus_recon", {"target": "example.com"})))
        scan_id = r["scan_id"]
        _join_scan(scan_id)
        r2 = _decode(_run(_srv.call_tool("argus_get_scan", {"scan_id": scan_id})))
        r3 = _decode(_run(_srv.call_tool("argus_get_scan", {"scan_id": scan_id})))
        assert r2["scan_id"] == scan_id
        assert r3["scan_id"] == scan_id


# ---------------------------------------------------------------------------
# Unknown tool
# ---------------------------------------------------------------------------

class TestUnknownTool:
    def test_unknown_tool_returns_error(self, monkeypatch):
        monkeypatch.setattr(_srv, "_initialized", True)
        monkeypatch.setattr(_srv, "_agent", MagicMock())
        result = _decode(_run(_srv.call_tool("does_not_exist", {})))
        assert "error" in result
        assert "does_not_exist" in result["error"]


# ---------------------------------------------------------------------------
# Response format invariants
# ---------------------------------------------------------------------------

class TestResponseFormat:
    def test_returns_single_text_content(self, monkeypatch):
        agent = MagicMock()
        agent.run.return_value = _make_report()
        _patch_agent(monkeypatch, agent)
        contents = _run(_srv.call_tool("argus_recon", {"target": "example.com"}))
        assert len(contents) == 1
        assert contents[0].type == "text"

    def test_output_is_valid_json(self, monkeypatch):
        agent = MagicMock()
        agent.run.return_value = _make_report()
        _patch_agent(monkeypatch, agent)
        contents = _run(_srv.call_tool("argus_recon", {"target": "example.com"}))
        assert isinstance(json.loads(contents[0].text), dict)

    def test_compact_response_well_within_limits(self, monkeypatch):
        """Initial scan_id response must be tiny — no data blobs."""
        agent = MagicMock()
        agent.run.return_value = _make_report()
        _patch_agent(monkeypatch, agent)
        contents = _run(_srv.call_tool("argus_recon", {"target": "example.com"}))
        assert len(contents[0].text) < 512

    def test_get_scan_response_within_limits(self, monkeypatch):
        """Completed scan response must stay well under MAX_OUTPUT_CHARS."""
        agent = MagicMock()
        agent.run.return_value = _make_report()
        _patch_agent(monkeypatch, agent)
        r = _decode(_run(_srv.call_tool("argus_recon", {"target": "example.com"})))
        _join_scan(r["scan_id"])
        contents = _run(_srv.call_tool("argus_get_scan", {"scan_id": r["scan_id"]}))
        assert len(contents[0].text) < 2048
