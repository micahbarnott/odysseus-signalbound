"""Tests for mcp_servers/argus_server.py.

Strategy
--------
* pytest.importorskip("mcp") — skips whole module when mcp isn't installed,
  matching the pattern used by other MCP server tests (test_rag_server_*).
* Monkeypatch `_agent` / `_initialized` so no real binary is invoked.
* For argus_scan_host / argus_enumerate_subdomains, monkeypatch the tool
  class on its source module so the local `from tools.nmap import NmapTool`
  inside the handler picks up the mock.
* asyncio.run() — no pytest-asyncio dependency.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("mcp")

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import mcp_servers.argus_server as _srv

from agents.argus.models import ReconHost, ReconPort, ReconReport
from tools.nmap.nmap_tool import NmapHost, NmapPort, NmapRunStats, NmapScanResult, NmapService
from tools.subfinder.subfinder_tool import SubfinderDomain, SubfinderScanResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    return asyncio.run(coro)


def _decode(text_contents) -> dict:
    assert len(text_contents) == 1
    return json.loads(text_contents[0].text)


def _patch_agent(monkeypatch, agent_mock):
    monkeypatch.setattr(_srv, "_agent", agent_mock)
    monkeypatch.setattr(_srv, "_initialized", True)


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


def _make_host(hostname="a.example.com", ports=()) -> ReconHost:
    return ReconHost(hostname=hostname, live=True, ports=list(ports), technologies=[])


def _make_port(port=80, protocol="tcp", state="open", service="http") -> ReconPort:
    return ReconPort(port=port, protocol=protocol, state=state, service=service)


def _make_nmap_result(ports=()) -> NmapScanResult:
    nmap_ports = [
        NmapPort(
            port=p[0],
            protocol=p[1],
            state="open",
            service=NmapService(name=p[2]) if len(p) > 2 else None,
        )
        for p in ports
    ]
    host = NmapHost(address="1.2.3.4", address_type="ipv4", state="up", ports=nmap_ports)
    return NmapScanResult(
        command="nmap -sV -oX - 1.2.3.4",
        hosts=[host],
        stats=NmapRunStats(elapsed=1.0, hosts_up=1, hosts_down=0, hosts_total=1),
    )


def _make_sf_result(hosts=("a.example.com",)) -> SubfinderScanResult:
    return SubfinderScanResult(
        command="subfinder -silent -json -d example.com",
        domains=[SubfinderDomain(host=h) for h in hosts],
    )


# ---------------------------------------------------------------------------
# list_tools
# ---------------------------------------------------------------------------

class TestListTools:
    def test_returns_three_tools(self):
        tools = _run(_srv.list_tools())
        assert len(tools) == 3

    def test_tool_names(self):
        tools = _run(_srv.list_tools())
        names = {t.name for t in tools}
        assert names == {"argus_recon", "argus_scan_host", "argus_enumerate_subdomains"}

    def test_each_tool_has_required_field(self):
        tools = _run(_srv.list_tools())
        for t in tools:
            assert t.inputSchema.get("required"), f"{t.name} has no required fields"

    def test_argus_recon_requires_target(self):
        tools = _run(_srv.list_tools())
        recon = next(t for t in tools if t.name == "argus_recon")
        assert "target" in recon.inputSchema["required"]

    def test_argus_scan_host_requires_target(self):
        tools = _run(_srv.list_tools())
        scan = next(t for t in tools if t.name == "argus_scan_host")
        assert "target" in scan.inputSchema["required"]

    def test_argus_enumerate_requires_domain(self):
        tools = _run(_srv.list_tools())
        enum = next(t for t in tools if t.name == "argus_enumerate_subdomains")
        assert "domain" in enum.inputSchema["required"]


# ---------------------------------------------------------------------------
# argus_recon
# ---------------------------------------------------------------------------

class TestArgusRecon:
    def test_happy_path_returns_report(self, monkeypatch):
        report = _make_report(
            domains=["a.example.com", "b.example.com"],
            hosts=[_make_host("a.example.com", [_make_port()])],
        )
        agent = MagicMock()
        agent.run.return_value = report
        _patch_agent(monkeypatch, agent)

        result = _decode(_run(_srv.call_tool("argus_recon", {"target": "example.com"})))

        assert result["target"] == "example.com"
        assert "a.example.com" in result["discovered_domains"]
        assert len(result["live_hosts"]) == 1
        assert result["errors"] == []

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

    def test_report_with_partial_errors(self, monkeypatch):
        report = _make_report(errors=["subfinder: binary not found"])
        agent = MagicMock()
        agent.run.return_value = report
        _patch_agent(monkeypatch, agent)

        result = _decode(_run(_srv.call_tool("argus_recon", {"target": "example.com"})))

        assert result["errors"] == ["subfinder: binary not found"]

    def test_domains_truncated_when_over_limit(self, monkeypatch):
        many_domains = [f"sub{i}.example.com" for i in range(250)]
        report = _make_report(domains=many_domains)
        agent = MagicMock()
        agent.run.return_value = report
        _patch_agent(monkeypatch, agent)

        result = _decode(_run(_srv.call_tool("argus_recon", {"target": "example.com"})))

        assert len(result["discovered_domains"]) == _srv._DOMAIN_LIMIT
        assert "_domains_truncated" in result
        assert "250" in result["_domains_truncated"]

    def test_hosts_truncated_when_over_limit(self, monkeypatch):
        many_hosts = [_make_host(f"h{i}.example.com") for i in range(25)]
        report = _make_report(hosts=many_hosts)
        agent = MagicMock()
        agent.run.return_value = report
        _patch_agent(monkeypatch, agent)

        result = _decode(_run(_srv.call_tool("argus_recon", {"target": "example.com"})))

        assert len(result["live_hosts"]) == _srv._HOST_LIMIT
        assert "_hosts_truncated" in result

    def test_domains_not_truncated_at_exact_limit(self, monkeypatch):
        domains = [f"sub{i}.example.com" for i in range(_srv._DOMAIN_LIMIT)]
        report = _make_report(domains=domains)
        agent = MagicMock()
        agent.run.return_value = report
        _patch_agent(monkeypatch, agent)

        result = _decode(_run(_srv.call_tool("argus_recon", {"target": "example.com"})))

        assert len(result["discovered_domains"]) == _srv._DOMAIN_LIMIT
        assert "_domains_truncated" not in result

    def test_hosts_not_truncated_at_exact_limit(self, monkeypatch):
        hosts = [_make_host(f"h{i}.example.com") for i in range(_srv._HOST_LIMIT)]
        report = _make_report(hosts=hosts)
        agent = MagicMock()
        agent.run.return_value = report
        _patch_agent(monkeypatch, agent)

        result = _decode(_run(_srv.call_tool("argus_recon", {"target": "example.com"})))

        assert len(result["live_hosts"]) == _srv._HOST_LIMIT
        assert "_hosts_truncated" not in result

    def test_agent_exception_caught_by_safety_net(self, monkeypatch):
        agent = MagicMock()
        agent.run.side_effect = RuntimeError("unexpected crash")
        _patch_agent(monkeypatch, agent)

        result = _decode(_run(_srv.call_tool("argus_recon", {"target": "example.com"})))

        assert "error" in result
        assert "unexpected crash" in result["error"]

    def test_target_passed_to_agent(self, monkeypatch):
        report = _make_report(target="target.io")
        agent = MagicMock()
        agent.run.return_value = report
        _patch_agent(monkeypatch, agent)

        _run(_srv.call_tool("argus_recon", {"target": "target.io"}))

        agent.run.assert_called_once_with("target.io")


# ---------------------------------------------------------------------------
# argus_scan_host
# ---------------------------------------------------------------------------

class TestArgusScanHost:
    def _run_with_nmap(self, monkeypatch, nm_result, target="1.2.3.4"):
        nmap_tool = MagicMock()
        nmap_tool.scan.return_value = nm_result
        monkeypatch.setattr(_srv, "_initialized", True)
        monkeypatch.setattr(_srv, "_agent", MagicMock())
        # Patch NmapTool on its source module so the `from tools.nmap import
        # NmapTool` inside _handle_argus_scan_host picks up the mock.
        import tools.nmap as _nmap_pkg
        monkeypatch.setattr(_nmap_pkg, "NmapTool", lambda: nmap_tool)
        return _decode(_run(_srv.call_tool("argus_scan_host", {"target": target})))

    def test_happy_path_returns_nmap_result(self, monkeypatch):
        nm = _make_nmap_result(ports=[(80, "tcp", "http"), (443, "tcp", "https")])
        result = self._run_with_nmap(monkeypatch, nm)
        assert "hosts" in result
        assert result["error"] is None

    def test_missing_target_returns_error(self, monkeypatch):
        monkeypatch.setattr(_srv, "_initialized", True)
        monkeypatch.setattr(_srv, "_agent", MagicMock())
        result = _decode(_run(_srv.call_tool("argus_scan_host", {})))
        assert "error" in result

    def test_empty_target_returns_error(self, monkeypatch):
        monkeypatch.setattr(_srv, "_initialized", True)
        monkeypatch.setattr(_srv, "_agent", MagicMock())
        result = _decode(_run(_srv.call_tool("argus_scan_host", {"target": ""})))
        assert "error" in result

    def test_nmap_error_surfaced(self, monkeypatch):
        nm = NmapScanResult(command="nmap", error="binary not found")
        result = self._run_with_nmap(monkeypatch, nm)
        assert result["error"] == "binary not found"

    def test_exception_caught_by_safety_net(self, monkeypatch):
        nmap_tool = MagicMock()
        nmap_tool.scan.side_effect = RuntimeError("boom")
        monkeypatch.setattr(_srv, "_initialized", True)
        monkeypatch.setattr(_srv, "_agent", MagicMock())
        import tools.nmap as _nmap_pkg
        monkeypatch.setattr(_nmap_pkg, "NmapTool", lambda: nmap_tool)
        result = _decode(_run(_srv.call_tool("argus_scan_host", {"target": "1.2.3.4"})))
        assert "error" in result
        assert "boom" in result["error"]

    def test_target_passed_to_nmap(self, monkeypatch):
        nm = _make_nmap_result()
        nmap_tool = MagicMock()
        nmap_tool.scan.return_value = nm
        monkeypatch.setattr(_srv, "_initialized", True)
        monkeypatch.setattr(_srv, "_agent", MagicMock())
        import tools.nmap as _nmap_pkg
        monkeypatch.setattr(_nmap_pkg, "NmapTool", lambda: nmap_tool)
        _run(_srv.call_tool("argus_scan_host", {"target": "192.168.1.1"}))
        nmap_tool.scan.assert_called_once_with("192.168.1.1")


# ---------------------------------------------------------------------------
# argus_enumerate_subdomains
# ---------------------------------------------------------------------------

class TestArgusEnumerateSubdomains:
    def _run_with_sf(self, monkeypatch, sf_result, domain="example.com"):
        sf_tool = MagicMock()
        sf_tool.scan.return_value = sf_result
        monkeypatch.setattr(_srv, "_initialized", True)
        monkeypatch.setattr(_srv, "_agent", MagicMock())
        import tools.subfinder as _sf_pkg
        monkeypatch.setattr(_sf_pkg, "SubfinderTool", lambda: sf_tool)
        return _decode(_run(_srv.call_tool("argus_enumerate_subdomains", {"domain": domain})))

    def test_happy_path_returns_domains(self, monkeypatch):
        sf = _make_sf_result(hosts=["a.example.com", "b.example.com"])
        result = self._run_with_sf(monkeypatch, sf)
        assert len(result["domains"]) == 2
        assert result["error"] is None

    def test_missing_domain_returns_error(self, monkeypatch):
        monkeypatch.setattr(_srv, "_initialized", True)
        monkeypatch.setattr(_srv, "_agent", MagicMock())
        result = _decode(_run(_srv.call_tool("argus_enumerate_subdomains", {})))
        assert "error" in result

    def test_empty_domain_returns_error(self, monkeypatch):
        monkeypatch.setattr(_srv, "_initialized", True)
        monkeypatch.setattr(_srv, "_agent", MagicMock())
        result = _decode(_run(_srv.call_tool("argus_enumerate_subdomains", {"domain": ""})))
        assert "error" in result

    def test_domains_truncated_when_over_limit(self, monkeypatch):
        sf = _make_sf_result(hosts=[f"sub{i}.example.com" for i in range(250)])
        result = self._run_with_sf(monkeypatch, sf)
        assert len(result["domains"]) == _srv._DOMAIN_LIMIT
        assert "_domains_truncated" in result
        assert "250" in result["_domains_truncated"]

    def test_domains_not_truncated_at_exact_limit(self, monkeypatch):
        sf = _make_sf_result(hosts=[f"sub{i}.example.com" for i in range(_srv._DOMAIN_LIMIT)])
        result = self._run_with_sf(monkeypatch, sf)
        assert len(result["domains"]) == _srv._DOMAIN_LIMIT
        assert "_domains_truncated" not in result

    def test_subfinder_error_surfaced(self, monkeypatch):
        sf = SubfinderScanResult(command="subfinder", error="binary not found")
        result = self._run_with_sf(monkeypatch, sf)
        assert result["error"] == "binary not found"

    def test_exception_caught_by_safety_net(self, monkeypatch):
        sf_tool = MagicMock()
        sf_tool.scan.side_effect = RuntimeError("crash")
        monkeypatch.setattr(_srv, "_initialized", True)
        monkeypatch.setattr(_srv, "_agent", MagicMock())
        import tools.subfinder as _sf_pkg
        monkeypatch.setattr(_sf_pkg, "SubfinderTool", lambda: sf_tool)
        result = _decode(_run(_srv.call_tool("argus_enumerate_subdomains", {"domain": "example.com"})))
        assert "error" in result
        assert "crash" in result["error"]

    def test_domain_passed_to_subfinder(self, monkeypatch):
        sf = _make_sf_result()
        sf_tool = MagicMock()
        sf_tool.scan.return_value = sf
        monkeypatch.setattr(_srv, "_initialized", True)
        monkeypatch.setattr(_srv, "_agent", MagicMock())
        import tools.subfinder as _sf_pkg
        monkeypatch.setattr(_sf_pkg, "SubfinderTool", lambda: sf_tool)
        _run(_srv.call_tool("argus_enumerate_subdomains", {"domain": "target.io"}))
        sf_tool.scan.assert_called_once_with("target.io")


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
# Response format
# ---------------------------------------------------------------------------

class TestResponseFormat:
    def test_returns_single_text_content(self, monkeypatch):
        report = _make_report()
        agent = MagicMock()
        agent.run.return_value = report
        _patch_agent(monkeypatch, agent)

        contents = _run(_srv.call_tool("argus_recon", {"target": "example.com"}))

        assert len(contents) == 1
        assert contents[0].type == "text"

    def test_output_is_valid_json(self, monkeypatch):
        report = _make_report()
        agent = MagicMock()
        agent.run.return_value = report
        _patch_agent(monkeypatch, agent)

        contents = _run(_srv.call_tool("argus_recon", {"target": "example.com"}))

        parsed = json.loads(contents[0].text)
        assert isinstance(parsed, dict)

    def test_output_within_max_output_chars(self, monkeypatch):
        """Worst-case output (at truncation limits) must stay under 10,000 chars."""
        many_hosts = [_make_host(f"h{i}.example.com", [_make_port(80 + i)]) for i in range(_srv._HOST_LIMIT)]
        report = _make_report(
            domains=[f"sub{i}.example.com" for i in range(_srv._DOMAIN_LIMIT)],
            hosts=many_hosts,
        )
        agent = MagicMock()
        agent.run.return_value = report
        _patch_agent(monkeypatch, agent)

        contents = _run(_srv.call_tool("argus_recon", {"target": "example.com"}))

        assert len(contents[0].text) < 10_000
