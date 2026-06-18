"""Unit tests for agents/argus/recon_agent.py and agents/argus/models.py.

All tool calls are mocked — no binaries, no network I/O.
Tool return values are real Pydantic model instances so that the agent's
attribute access paths (`.success`, `.live`, `.open_ports`, etc.) are
exercised against real model code.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agents.argus.models import ReconHost, ReconPort, ReconReport
from agents.argus.recon_agent import ArgusAgent
from tools.httpx.httpx_tool import HttpxProbeResult, HttpxScanResult
from tools.nmap.nmap_tool import NmapHost, NmapPort, NmapScanResult, NmapService
from tools.subfinder.subfinder_tool import SubfinderDomain, SubfinderScanResult

# ---------------------------------------------------------------------------
# Result factories — construct real model instances for mock return values
# ---------------------------------------------------------------------------


def _sf_ok(*hosts: str, target: str = "example.com") -> SubfinderScanResult:
    return SubfinderScanResult(
        command=f"subfinder -silent -json -d {target}",
        domains=[SubfinderDomain(host=h, input=target, source="test") for h in hosts],
    )


def _sf_err(msg: str = "subfinder failed") -> SubfinderScanResult:
    return SubfinderScanResult(command="subfinder ...", error=msg)


def _hx_probe(
    input_host: str,
    host_ip: str = "1.2.3.4",
    status_code: int = 200,
    tech: list[str] | None = None,
) -> HttpxProbeResult:
    return HttpxProbeResult(
        url=f"https://{input_host}",
        input=input_host,
        host=host_ip,
        status_code=status_code,
        failed=False,
        tech=tech or [],
    )


def _hx_ok(*probes: HttpxProbeResult) -> HttpxScanResult:
    return HttpxScanResult(command="httpx ...", probes=list(probes))


def _hx_err(msg: str = "httpx failed") -> HttpxScanResult:
    return HttpxScanResult(command="httpx ...", error=msg)


def _nm_ok(*port_tuples: tuple) -> NmapScanResult:
    """Build a NmapScanResult with one host containing the given open ports.

    Each tuple is (port_number, protocol, service_name).
    """
    ports = [
        NmapPort(
            port=t[0],
            protocol=t[1],
            state="open",
            service=NmapService(name=t[2]) if len(t) > 2 else None,
        )
        for t in port_tuples
    ]
    host = NmapHost(address="1.2.3.4", address_type="ipv4", state="up", ports=ports)
    return NmapScanResult(command="nmap ...", hosts=[host])


def _nm_err(msg: str = "nmap failed") -> NmapScanResult:
    return NmapScanResult(command="nmap ...", error=msg)


def _agent(
    sf_result: SubfinderScanResult | None = None,
    hx_result: HttpxScanResult | None = None,
    nm_result: NmapScanResult | None = None,
) -> ArgusAgent:
    """Build an ArgusAgent with each tool replaced by a MagicMock."""
    sf_mock = MagicMock()
    sf_mock.scan.return_value = sf_result or _sf_ok("www.example.com")

    hx_mock = MagicMock()
    hx_mock.probe.return_value = hx_result or _hx_ok()

    nm_mock = MagicMock()
    nm_mock.scan.return_value = nm_result or _nm_ok()

    return ArgusAgent(subfinder=sf_mock, httpx=hx_mock, nmap=nm_mock)


# ---------------------------------------------------------------------------
# ReconPort model
# ---------------------------------------------------------------------------


class TestReconPort:
    def test_fields_stored(self):
        p = ReconPort(port=443, protocol="tcp", state="open", service="https")
        assert p.port == 443
        assert p.protocol == "tcp"
        assert p.state == "open"
        assert p.service == "https"

    def test_service_defaults_empty(self):
        p = ReconPort(port=80, protocol="tcp", state="open")
        assert p.service == ""

    def test_closed_state_stored(self):
        p = ReconPort(port=22, protocol="tcp", state="closed")
        assert p.state == "closed"


# ---------------------------------------------------------------------------
# ReconHost model
# ---------------------------------------------------------------------------


class TestReconHostDefaults:
    def test_live_defaults_false(self):
        h = ReconHost(hostname="example.com")
        assert h.live is False

    def test_ports_default_empty(self):
        h = ReconHost(hostname="example.com")
        assert h.ports == []

    def test_technologies_default_empty(self):
        h = ReconHost(hostname="example.com")
        assert h.technologies == []


class TestReconHostIsWeb:
    @pytest.mark.parametrize("web_port", [80, 443, 8080, 8443])
    def test_open_web_port_is_web(self, web_port: int):
        h = ReconHost(
            hostname="h",
            ports=[ReconPort(port=web_port, protocol="tcp", state="open")],
        )
        assert h.is_web is True

    def test_non_web_port_not_is_web(self):
        h = ReconHost(
            hostname="h",
            ports=[ReconPort(port=22, protocol="tcp", state="open")],
        )
        assert h.is_web is False

    def test_filtered_web_port_not_is_web(self):
        h = ReconHost(
            hostname="h",
            ports=[ReconPort(port=80, protocol="tcp", state="filtered")],
        )
        assert h.is_web is False

    def test_tech_list_triggers_is_web(self):
        h = ReconHost(hostname="h", technologies=["nginx"])
        assert h.is_web is True

    def test_no_ports_no_tech_not_is_web(self):
        h = ReconHost(hostname="h")
        assert h.is_web is False

    def test_is_web_false_when_empty_tech_and_no_open_web_ports(self):
        h = ReconHost(hostname="h", technologies=[], ports=[])
        assert h.is_web is False


# ---------------------------------------------------------------------------
# ReconReport model
# ---------------------------------------------------------------------------


class TestReconReportSuccess:
    def test_success_when_no_errors(self):
        r = ReconReport(target="t")
        assert r.success is True

    def test_not_success_when_errors(self):
        r = ReconReport(target="t", errors=["oops"])
        assert r.success is False

    def test_multiple_errors_still_not_success(self):
        r = ReconReport(target="t", errors=["e1", "e2", "e3"])
        assert r.success is False


class TestReconReportHostCount:
    def test_host_count_matches_discovered_domains(self):
        r = ReconReport(target="t", discovered_domains=["a.t", "b.t", "c.t"])
        assert r.host_count == 3

    def test_host_count_zero_when_empty(self):
        r = ReconReport(target="t")
        assert r.host_count == 0


class TestReconReportLiveCount:
    def test_live_count_matches_live_hosts(self):
        r = ReconReport(
            target="t",
            live_hosts=[
                ReconHost(hostname="a", live=True),
                ReconHost(hostname="b", live=True),
            ],
        )
        assert r.live_count == 2

    def test_live_count_zero_when_empty(self):
        r = ReconReport(target="t")
        assert r.live_count == 0


class TestReconReportPortCount:
    def test_port_count_sums_across_hosts(self):
        r = ReconReport(
            target="t",
            live_hosts=[
                ReconHost(
                    hostname="a",
                    ports=[
                        ReconPort(port=80, protocol="tcp", state="open"),
                        ReconPort(port=443, protocol="tcp", state="open"),
                    ],
                ),
                ReconHost(
                    hostname="b",
                    ports=[ReconPort(port=22, protocol="tcp", state="open")],
                ),
            ],
        )
        assert r.port_count == 3

    def test_port_count_zero_when_no_live_hosts(self):
        r = ReconReport(target="t")
        assert r.port_count == 0

    def test_port_count_zero_when_hosts_have_no_ports(self):
        r = ReconReport(
            target="t",
            live_hosts=[ReconHost(hostname="a"), ReconHost(hostname="b")],
        )
        assert r.port_count == 0


# ---------------------------------------------------------------------------
# ArgusAgent — happy path
# ---------------------------------------------------------------------------


class TestArgusAgentHappyPath:
    def setup_method(self):
        probe = _hx_probe("www.example.com", host_ip="93.184.216.34", tech=["nginx"])
        self.sf_mock = MagicMock()
        self.hx_mock = MagicMock()
        self.nm_mock = MagicMock()

        self.sf_mock.scan.return_value = _sf_ok("www.example.com")
        self.hx_mock.probe.return_value = _hx_ok(probe)
        self.nm_mock.scan.return_value = _nm_ok((80, "tcp", "http"), (443, "tcp", "https"))

        self.agent = ArgusAgent(
            subfinder=self.sf_mock, httpx=self.hx_mock, nmap=self.nm_mock
        )
        self.report = self.agent.run("example.com")

    def test_no_errors(self):
        assert self.report.errors == []

    def test_success(self):
        assert self.report.success is True

    def test_target_recorded(self):
        assert self.report.target == "example.com"

    def test_subfinder_called_with_target(self):
        self.sf_mock.scan.assert_called_once_with("example.com")

    def test_httpx_called_with_discovered_domains(self):
        self.hx_mock.probe.assert_called_once_with(["www.example.com"])

    def test_nmap_called_with_host_ip(self):
        self.nm_mock.scan.assert_called_once_with("93.184.216.34")

    def test_discovered_domains_populated(self):
        assert self.report.discovered_domains == ["www.example.com"]

    def test_live_hosts_populated(self):
        assert self.report.live_count == 1

    def test_live_host_hostname(self):
        assert self.report.live_hosts[0].hostname == "www.example.com"

    def test_live_host_technologies(self):
        assert "nginx" in self.report.live_hosts[0].technologies

    def test_ports_populated(self):
        assert self.report.port_count == 2

    def test_port_numbers(self):
        port_numbers = {p.port for p in self.report.live_hosts[0].ports}
        assert port_numbers == {80, 443}

    def test_port_services(self):
        services = {p.service for p in self.report.live_hosts[0].ports}
        assert "http" in services
        assert "https" in services


# ---------------------------------------------------------------------------
# ArgusAgent — subfinder failure
# ---------------------------------------------------------------------------


class TestArgusAgentSubfinderFailure:
    def setup_method(self):
        probe = _hx_probe("example.com")
        sf_mock = MagicMock()
        hx_mock = MagicMock()
        nm_mock = MagicMock()

        sf_mock.scan.return_value = _sf_err("network timeout")
        hx_mock.probe.return_value = _hx_ok(probe)
        nm_mock.scan.return_value = _nm_ok((80, "tcp", "http"))

        self.hx_mock = hx_mock
        self.report = ArgusAgent(
            subfinder=sf_mock, httpx=hx_mock, nmap=nm_mock
        ).run("example.com")

    def test_error_recorded(self):
        assert any("subfinder" in e for e in self.report.errors)

    def test_error_contains_original_message(self):
        assert any("network timeout" in e for e in self.report.errors)

    def test_not_success(self):
        assert self.report.success is False

    def test_fallback_domain_is_target(self):
        assert self.report.discovered_domains == ["example.com"]

    def test_httpx_still_called_with_fallback(self):
        self.hx_mock.probe.assert_called_once_with(["example.com"])

    def test_live_hosts_still_populated(self):
        assert self.report.live_count == 1


# ---------------------------------------------------------------------------
# ArgusAgent — subfinder returns zero results (not an error)
# ---------------------------------------------------------------------------


class TestArgusAgentSubfinderEmpty:
    def setup_method(self):
        probe = _hx_probe("example.com")
        sf_mock = MagicMock()
        hx_mock = MagicMock()
        nm_mock = MagicMock()

        sf_mock.scan.return_value = _sf_ok()  # zero domains, no error
        hx_mock.probe.return_value = _hx_ok(probe)
        nm_mock.scan.return_value = _nm_ok()

        self.hx_mock = hx_mock
        self.report = ArgusAgent(
            subfinder=sf_mock, httpx=hx_mock, nmap=nm_mock
        ).run("example.com")

    def test_no_subfinder_error_added(self):
        assert not any("subfinder" in e for e in self.report.errors)

    def test_fallback_to_target(self):
        assert self.report.discovered_domains == ["example.com"]

    def test_httpx_called_with_target_fallback(self):
        self.hx_mock.probe.assert_called_once_with(["example.com"])


# ---------------------------------------------------------------------------
# ArgusAgent — httpx failure
# ---------------------------------------------------------------------------


class TestArgusAgentHttpxFailure:
    def setup_method(self):
        sf_mock = MagicMock()
        hx_mock = MagicMock()
        nm_mock = MagicMock()

        sf_mock.scan.return_value = _sf_ok("www.example.com", "api.example.com")
        hx_mock.probe.return_value = _hx_err("connection refused")
        # nmap should NOT be called

        self.nm_mock = nm_mock
        self.report = ArgusAgent(
            subfinder=sf_mock, httpx=hx_mock, nmap=nm_mock
        ).run("example.com")

    def test_error_recorded(self):
        assert any("httpx" in e for e in self.report.errors)

    def test_error_contains_original_message(self):
        assert any("connection refused" in e for e in self.report.errors)

    def test_not_success(self):
        assert self.report.success is False

    def test_no_live_hosts(self):
        assert self.report.live_count == 0

    def test_nmap_never_called(self):
        self.nm_mock.scan.assert_not_called()

    def test_domains_still_populated(self):
        assert set(self.report.discovered_domains) == {"www.example.com", "api.example.com"}


# ---------------------------------------------------------------------------
# ArgusAgent — nmap failure (one host)
# ---------------------------------------------------------------------------


class TestArgusAgentNmapFailure:
    def setup_method(self):
        probe = _hx_probe("www.example.com")
        sf_mock = MagicMock()
        hx_mock = MagicMock()
        nm_mock = MagicMock()

        sf_mock.scan.return_value = _sf_ok("www.example.com")
        hx_mock.probe.return_value = _hx_ok(probe)
        nm_mock.scan.return_value = _nm_err("scan permission denied")

        self.report = ArgusAgent(
            subfinder=sf_mock, httpx=hx_mock, nmap=nm_mock
        ).run("example.com")

    def test_error_recorded(self):
        assert any("nmap" in e for e in self.report.errors)

    def test_error_contains_host(self):
        assert any("1.2.3.4" in e for e in self.report.errors)

    def test_error_contains_original_message(self):
        assert any("permission denied" in e for e in self.report.errors)

    def test_not_success(self):
        assert self.report.success is False

    def test_live_host_still_added(self):
        assert self.report.live_count == 1

    def test_live_host_has_empty_ports(self):
        assert self.report.live_hosts[0].ports == []


# ---------------------------------------------------------------------------
# ArgusAgent — nmap failure on one of multiple hosts
# ---------------------------------------------------------------------------


class TestArgusAgentNmapPartialFailure:
    def setup_method(self):
        probe_a = _hx_probe("www.example.com", host_ip="1.1.1.1")
        probe_b = _hx_probe("api.example.com", host_ip="2.2.2.2")

        sf_mock = MagicMock()
        hx_mock = MagicMock()
        nm_mock = MagicMock()

        sf_mock.scan.return_value = _sf_ok("www.example.com", "api.example.com")
        hx_mock.probe.return_value = _hx_ok(probe_a, probe_b)

        # Use target-based dispatch so results are deterministic under parallel
        # execution (as_completed order is non-deterministic with real threads).
        def _nmap_side_effect(target):
            if target == "1.1.1.1":
                return _nm_ok((80, "tcp", "http"))
            return _nm_err("timeout")

        nm_mock.scan.side_effect = _nmap_side_effect

        self.report = ArgusAgent(
            subfinder=sf_mock, httpx=hx_mock, nmap=nm_mock
        ).run("example.com")

    def test_one_error_recorded(self):
        nmap_errors = [e for e in self.report.errors if "nmap" in e]
        assert len(nmap_errors) == 1

    def test_two_live_hosts_added(self):
        assert self.report.live_count == 2

    def test_successful_host_has_ports(self):
        www = next(h for h in self.report.live_hosts if h.hostname == "www.example.com")
        assert len(www.ports) == 1

    def test_failed_host_has_empty_ports(self):
        api = next(h for h in self.report.live_hosts if h.hostname == "api.example.com")
        assert api.ports == []

    def test_not_success(self):
        assert self.report.success is False


# ---------------------------------------------------------------------------
# ArgusAgent — all three tools fail
# ---------------------------------------------------------------------------


class TestArgusAgentAllToolsFail:
    def setup_method(self):
        sf_mock = MagicMock()
        hx_mock = MagicMock()
        nm_mock = MagicMock()

        sf_mock.scan.return_value = _sf_err("no resolvers")
        hx_mock.probe.return_value = _hx_err("dial error")
        # nmap won't be called because httpx failed

        self.report = ArgusAgent(
            subfinder=sf_mock, httpx=hx_mock, nmap=nm_mock
        ).run("example.com")

    def test_two_errors_recorded(self):
        assert len(self.report.errors) == 2

    def test_not_success(self):
        assert self.report.success is False

    def test_discovered_domains_has_fallback(self):
        assert self.report.discovered_domains == ["example.com"]

    def test_no_live_hosts(self):
        assert self.report.live_count == 0


# ---------------------------------------------------------------------------
# ArgusAgent — no live hosts from httpx
# ---------------------------------------------------------------------------


class TestArgusAgentNoLiveHosts:
    def setup_method(self):
        # httpx succeeds but nothing is live (all failed probes)
        dead_probe = HttpxProbeResult(
            url="https://dead.example.com",
            input="dead.example.com",
            host="",
            status_code=None,
            failed=True,
        )
        sf_mock = MagicMock()
        hx_mock = MagicMock()
        nm_mock = MagicMock()

        sf_mock.scan.return_value = _sf_ok("dead.example.com")
        hx_mock.probe.return_value = _hx_ok(dead_probe)

        self.nm_mock = nm_mock
        self.report = ArgusAgent(
            subfinder=sf_mock, httpx=hx_mock, nmap=nm_mock
        ).run("example.com")

    def test_no_errors(self):
        assert self.report.errors == []

    def test_success(self):
        assert self.report.success is True

    def test_live_count_zero(self):
        assert self.report.live_count == 0

    def test_nmap_never_called(self):
        self.nm_mock.scan.assert_not_called()


# ---------------------------------------------------------------------------
# ArgusAgent — unexpected exception from a tool (safety net)
# ---------------------------------------------------------------------------


class TestArgusAgentNeverRaises:
    def test_subfinder_raises_unexpectedly(self):
        sf_mock = MagicMock()
        hx_mock = MagicMock()
        nm_mock = MagicMock()

        sf_mock.scan.side_effect = RuntimeError("unexpected crash")
        hx_mock.probe.return_value = _hx_ok()
        nm_mock.scan.return_value = _nm_ok()

        report = ArgusAgent(subfinder=sf_mock, httpx=hx_mock, nmap=nm_mock).run("example.com")
        assert isinstance(report, ReconReport)
        assert not report.success
        assert any("unexpected crash" in e for e in report.errors)

    def test_httpx_raises_unexpectedly(self):
        sf_mock = MagicMock()
        hx_mock = MagicMock()
        nm_mock = MagicMock()

        sf_mock.scan.return_value = _sf_ok("www.example.com")
        hx_mock.probe.side_effect = RuntimeError("httpx crash")
        nm_mock.scan.return_value = _nm_ok()

        report = ArgusAgent(subfinder=sf_mock, httpx=hx_mock, nmap=nm_mock).run("example.com")
        assert isinstance(report, ReconReport)
        assert not report.success
        assert any("httpx crash" in e for e in report.errors)

    def test_nmap_raises_unexpectedly(self):
        probe = _hx_probe("www.example.com")
        sf_mock = MagicMock()
        hx_mock = MagicMock()
        nm_mock = MagicMock()

        sf_mock.scan.return_value = _sf_ok("www.example.com")
        hx_mock.probe.return_value = _hx_ok(probe)
        nm_mock.scan.side_effect = RuntimeError("nmap crash")

        report = ArgusAgent(subfinder=sf_mock, httpx=hx_mock, nmap=nm_mock).run("example.com")
        assert isinstance(report, ReconReport)
        assert not report.success
        assert any("nmap crash" in e for e in report.errors)


# ---------------------------------------------------------------------------
# ArgusAgent — command recording
# ---------------------------------------------------------------------------


class TestArgusAgentCommandRecording:
    def test_report_target_matches_input(self):
        report = _agent(sf_result=_sf_ok("www.target.io")).run("target.io")
        assert report.target == "target.io"

    def test_different_targets_produce_separate_reports(self):
        agent = ArgusAgent(
            subfinder=MagicMock(**{"scan.return_value": _sf_ok()}),
            httpx=MagicMock(**{"probe.return_value": _hx_ok()}),
            nmap=MagicMock(**{"scan.return_value": _nm_ok()}),
        )
        r1 = agent.run("alpha.com")
        r2 = agent.run("beta.com")
        assert r1.target == "alpha.com"
        assert r2.target == "beta.com"


# ---------------------------------------------------------------------------
# ArgusAgent — ScanSession phase updates
# ---------------------------------------------------------------------------

from agents.argus.scan_registry import ScanSession, clear_registry, create_session


@pytest.fixture(autouse=False)
def clean_registry():
    yield
    clear_registry()


class TestArgusAgentSessionUpdates:
    """Verify that run() updates a ScanSession at each phase boundary."""

    def _make_agent(self, sf_result=None, hx_result=None, nm_result=None):
        sf = MagicMock()
        sf.scan.return_value = sf_result or _sf_ok("www.example.com")
        hx = MagicMock()
        hx.probe.return_value = hx_result or _hx_ok()
        nm = MagicMock()
        nm.scan.return_value = nm_result or _nm_ok()
        return ArgusAgent(subfinder=sf, httpx=hx, nmap=nm)

    def test_session_status_complete_after_run(self, clean_registry):
        session = create_session("example.com")
        self._make_agent().run("example.com", session=session)
        assert session.status == "complete"

    def test_session_phase_done_after_run(self, clean_registry):
        session = create_session("example.com")
        self._make_agent().run("example.com", session=session)
        assert session.phase == "done"

    def test_session_completed_at_set(self, clean_registry):
        session = create_session("example.com")
        self._make_agent().run("example.com", session=session)
        assert session.completed_at is not None

    def test_session_discovered_domains_updated(self, clean_registry):
        session = create_session("example.com")
        self._make_agent(
            sf_result=_sf_ok("a.example.com", "b.example.com", "c.example.com")
        ).run("example.com", session=session)
        assert session.discovered_domains == 3

    def test_session_live_count_updated(self, clean_registry):
        probe = _hx_probe("www.example.com")
        session = create_session("example.com")
        self._make_agent(hx_result=_hx_ok(probe)).run("example.com", session=session)
        assert session.live_count == 1

    def test_session_port_count_updated(self, clean_registry):
        probe = _hx_probe("www.example.com")
        session = create_session("example.com")
        self._make_agent(
            hx_result=_hx_ok(probe),
            nm_result=_nm_ok((80, "tcp", "http"), (443, "tcp", "https")),
        ).run("example.com", session=session)
        assert session.port_count == 2

    def test_session_errors_populated_on_failure(self, clean_registry):
        session = create_session("example.com")
        self._make_agent(sf_result=_sf_err("bad network")).run(
            "example.com", session=session
        )
        assert any("bad network" in e for e in session.errors)

    def test_session_summary_updated_after_run(self, clean_registry):
        probe = _hx_probe("www.example.com")
        session = create_session("example.com")
        self._make_agent(hx_result=_hx_ok(probe)).run("example.com", session=session)
        assert session.summary  # non-empty

    def test_run_without_session_still_works(self):
        """Passing no session must not raise."""
        report = self._make_agent().run("example.com")
        assert isinstance(report, ReconReport)

    def test_session_highlights_populated_for_ssh_port(self, clean_registry):
        probe = _hx_probe("dev.example.com", host_ip="1.2.3.4")
        session = create_session("example.com")
        self._make_agent(
            sf_result=_sf_ok("dev.example.com"),
            hx_result=_hx_ok(probe),
            nm_result=_nm_ok((22, "tcp", "ssh")),
        ).run("example.com", session=session)
        assert any("SSH" in h or "dev" in h for h in session.highlights)

    def test_httpx_failure_marks_session_complete(self, clean_registry):
        session = create_session("example.com")
        self._make_agent(hx_result=_hx_err("connection refused")).run(
            "example.com", session=session
        )
        assert session.status == "complete"
        assert session.phase == "done"


# ---------------------------------------------------------------------------
# ArgusAgent — parallel nmap
# ---------------------------------------------------------------------------

import threading as _threading


class TestArgusAgentParallelNmap:
    """Verify _scan_hosts_parallel correctness and concurrency properties."""

    def _make_agent_with_nm(self, nm_mock):
        sf = MagicMock()
        sf.scan.return_value = _sf_ok()
        hx = MagicMock()
        hx.probe.return_value = _hx_ok()
        return ArgusAgent(subfinder=sf, httpx=hx, nmap=nm_mock)

    def test_all_probes_scanned(self):
        probes = [_hx_probe(f"h{i}.example.com", host_ip=f"1.2.3.{i}") for i in range(5)]
        nm = MagicMock()
        nm.scan.return_value = _nm_ok((80, "tcp", "http"))
        agent = self._make_agent_with_nm(nm)
        errors: list[str] = []
        hosts = agent._scan_hosts_parallel(probes, errors)
        assert nm.scan.call_count == 5

    def test_all_hosts_returned(self):
        probes = [_hx_probe(f"h{i}.example.com", host_ip=f"1.2.3.{i}") for i in range(4)]
        nm = MagicMock()
        nm.scan.return_value = _nm_ok((443, "tcp", "https"))
        agent = self._make_agent_with_nm(nm)
        errors: list[str] = []
        hosts = agent._scan_hosts_parallel(probes, errors)
        assert len(hosts) == 4
        assert errors == []

    def test_empty_probes_returns_empty(self):
        nm = MagicMock()
        agent = self._make_agent_with_nm(nm)
        errors: list[str] = []
        hosts = agent._scan_hosts_parallel([], errors)
        assert hosts == []
        nm.scan.assert_not_called()

    def test_failed_probe_included_with_empty_ports(self):
        probe = _hx_probe("www.example.com", host_ip="1.2.3.4")
        nm = MagicMock()
        nm.scan.return_value = _nm_err("timeout")
        agent = self._make_agent_with_nm(nm)
        errors: list[str] = []
        hosts = agent._scan_hosts_parallel([probe], errors)
        assert len(hosts) == 1
        assert hosts[0].ports == []
        assert len(errors) == 1

    def test_partial_failure_error_recorded(self):
        probe_ok = _hx_probe("ok.example.com", host_ip="1.1.1.1")
        probe_bad = _hx_probe("bad.example.com", host_ip="2.2.2.2")
        nm = MagicMock()

        def _side(target):
            return _nm_ok((80, "tcp", "http")) if target == "1.1.1.1" else _nm_err("refused")

        nm.scan.side_effect = _side
        agent = self._make_agent_with_nm(nm)
        errors: list[str] = []
        hosts = agent._scan_hosts_parallel([probe_ok, probe_bad], errors)
        assert len(hosts) == 2
        assert len(errors) == 1
        assert any("refused" in e for e in errors)

    def test_successful_host_has_ports(self):
        probe = _hx_probe("www.example.com", host_ip="1.2.3.4")
        nm = MagicMock()
        nm.scan.return_value = _nm_ok((80, "tcp", "http"), (443, "tcp", "https"))
        agent = self._make_agent_with_nm(nm)
        errors: list[str] = []
        hosts = agent._scan_hosts_parallel([probe], errors)
        assert len(hosts[0].ports) == 2

    def test_concurrency_bounded_by_max_workers(self):
        """Verify simultaneous nmap calls never exceed max_workers."""
        max_concurrent = 0
        current = 0
        lock = _threading.Lock()

        def _slow_scan(target):
            nonlocal max_concurrent, current
            with lock:
                current += 1
                max_concurrent = max(max_concurrent, current)
            import time as _t
            _t.sleep(0.02)
            with lock:
                current -= 1
            return _nm_ok((80, "tcp", "http"))

        probes = [_hx_probe(f"h{i}.example.com", host_ip=f"1.2.3.{i}") for i in range(8)]
        nm = MagicMock()
        nm.scan.side_effect = _slow_scan
        agent = self._make_agent_with_nm(nm)
        errors: list[str] = []
        agent._scan_hosts_parallel(probes, errors, max_workers=3)
        assert max_concurrent <= 3

    def test_nmap_exception_becomes_error_not_crash(self):
        probe = _hx_probe("www.example.com", host_ip="1.2.3.4")
        nm = MagicMock()
        nm.scan.side_effect = RuntimeError("unexpected crash")
        agent = self._make_agent_with_nm(nm)
        errors: list[str] = []
        hosts = agent._scan_hosts_parallel([probe], errors)
        assert len(hosts) == 1
        assert hosts[0].ports == []
        assert any("unexpected crash" in e for e in errors)

    def test_technologies_propagated_from_probe(self):
        probe = _hx_probe("www.example.com", host_ip="1.2.3.4", tech=["nginx", "react"])
        nm = MagicMock()
        nm.scan.return_value = _nm_ok()
        agent = self._make_agent_with_nm(nm)
        errors: list[str] = []
        hosts = agent._scan_hosts_parallel([probe], errors)
        assert "nginx" in hosts[0].technologies
        assert "react" in hosts[0].technologies
