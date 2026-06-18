"""Unit tests for tools/nmap/nmap_tool.py.

All tests are pure-Python and require neither nmap nor a live network.
subprocess.run is monkey-patched wherever an actual process would be invoked.
"""
from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tools.nmap.nmap_tool import (
    NmapHost,
    NmapPort,
    NmapRunStats,
    NmapScanResult,
    NmapService,
    NmapTool,
    _parse_xml,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIMPLE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<nmaprun scanner="nmap" args="nmap -sV -oX - 127.0.0.1" version="7.94" xmloutputversion="1.05">
  <host starttime="1700000000" endtime="1700000010">
    <status state="up" reason="localhost-response" />
    <address addr="127.0.0.1" addrtype="ipv4" />
    <hostnames>
      <hostname name="localhost" type="PTR" />
    </hostnames>
    <ports>
      <port protocol="tcp" portid="22">
        <state state="open" reason="syn-ack" />
        <service name="ssh" product="OpenSSH" version="8.9" extrainfo="Ubuntu Linux" conf="10" />
      </port>
      <port protocol="tcp" portid="80">
        <state state="open" reason="syn-ack" />
        <service name="http" product="nginx" version="1.24.0" extrainfo="" conf="10" />
      </port>
      <port protocol="tcp" portid="443">
        <state state="filtered" reason="no-response" />
      </port>
    </ports>
  </host>
  <runstats>
    <finished time="1700000010" elapsed="10.12" />
    <hosts up="1" down="0" total="1" />
  </runstats>
</nmaprun>
"""

MULTI_HOST_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<nmaprun scanner="nmap" version="7.94" xmloutputversion="1.05">
  <host>
    <status state="up" reason="arp-response" />
    <address addr="192.168.1.1" addrtype="ipv4" />
    <hostnames/>
    <ports>
      <port protocol="tcp" portid="80">
        <state state="open" reason="syn-ack" />
        <service name="http" product="Apache" version="2.4.57" conf="9" />
      </port>
    </ports>
  </host>
  <host>
    <status state="down" reason="no-response" />
    <address addr="192.168.1.2" addrtype="ipv4" />
    <hostnames/>
    <ports/>
  </host>
  <runstats>
    <finished time="1700001000" elapsed="30.00" />
    <hosts up="1" down="1" total="2" />
  </runstats>
</nmaprun>
"""

NO_PORTS_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<nmaprun scanner="nmap" version="7.94" xmloutputversion="1.05">
  <host>
    <status state="up" reason="echo-reply" />
    <address addr="10.0.0.1" addrtype="ipv4" />
    <hostnames/>
  </host>
  <runstats>
    <finished time="1700002000" elapsed="1.00" />
    <hosts up="1" down="0" total="1" />
  </runstats>
</nmaprun>
"""

IPV6_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<nmaprun scanner="nmap" version="7.94" xmloutputversion="1.05">
  <host>
    <status state="up" reason="nd-response" />
    <address addr="::1" addrtype="ipv6" />
    <hostnames/>
    <ports>
      <port protocol="tcp" portid="8080">
        <state state="open" reason="syn-ack" />
        <service name="http-alt" conf="5" />
      </port>
    </ports>
  </host>
  <runstats>
    <finished time="1700003000" elapsed="2.50" />
    <hosts up="1" down="0" total="1" />
  </runstats>
</nmaprun>
"""


# ---------------------------------------------------------------------------
# _parse_xml — unit tests
# ---------------------------------------------------------------------------


class TestParseXmlSimple:
    def setup_method(self):
        self.result = _parse_xml(SIMPLE_XML, "nmap -sV -oX - 127.0.0.1")

    def test_success(self):
        assert self.result.success

    def test_error_is_none(self):
        assert self.result.error is None

    def test_version(self):
        assert self.result.version == "7.94"

    def test_command_preserved(self):
        assert self.result.command == "nmap -sV -oX - 127.0.0.1"

    def test_one_host(self):
        assert len(self.result.hosts) == 1

    def test_host_address(self):
        assert self.result.hosts[0].address == "127.0.0.1"

    def test_host_address_type(self):
        assert self.result.hosts[0].address_type == "ipv4"

    def test_host_state(self):
        assert self.result.hosts[0].state == "up"

    def test_host_hostname(self):
        assert "localhost" in self.result.hosts[0].hostnames

    def test_three_ports(self):
        assert len(self.result.hosts[0].ports) == 3

    def test_open_ports_count(self):
        assert len(self.result.hosts[0].open_ports) == 2

    def test_ssh_port(self):
        ssh = self.result.hosts[0].ports[0]
        assert ssh.port == 22
        assert ssh.protocol == "tcp"
        assert ssh.state == "open"

    def test_ssh_service(self):
        svc = self.result.hosts[0].ports[0].service
        assert svc is not None
        assert svc.name == "ssh"
        assert svc.product == "OpenSSH"
        assert svc.version == "8.9"
        assert svc.extra_info == "Ubuntu Linux"
        assert svc.confidence == 10

    def test_filtered_port(self):
        filtered = self.result.hosts[0].ports[2]
        assert filtered.port == 443
        assert filtered.state == "filtered"
        assert filtered.service is None

    def test_run_stats(self):
        assert self.result.stats is not None
        assert self.result.stats.elapsed == pytest.approx(10.12)
        assert self.result.stats.hosts_up == 1
        assert self.result.stats.hosts_down == 0
        assert self.result.stats.hosts_total == 1


class TestParseXmlMultiHost:
    def setup_method(self):
        self.result = _parse_xml(MULTI_HOST_XML, "nmap -sV -oX - 192.168.1.0/24")

    def test_two_hosts(self):
        assert len(self.result.hosts) == 2

    def test_first_host_up(self):
        assert self.result.hosts[0].state == "up"

    def test_second_host_down(self):
        assert self.result.hosts[1].state == "down"

    def test_down_host_no_ports(self):
        assert self.result.hosts[1].open_ports == []

    def test_stats_two_total(self):
        assert self.result.stats.hosts_total == 2


class TestParseXmlNoPorts:
    def test_host_with_no_ports_container(self):
        result = _parse_xml(NO_PORTS_XML, "cmd")
        assert result.hosts[0].ports == []


class TestParseXmlIpv6:
    def test_ipv6_address_type(self):
        result = _parse_xml(IPV6_XML, "cmd")
        assert result.hosts[0].address_type == "ipv6"
        assert result.hosts[0].address == "::1"


class TestParseXmlMalformed:
    def test_returns_error_not_exception(self):
        result = _parse_xml("<not valid xml <<", "cmd")
        assert not result.success
        assert "XML parse error" in (result.error or "")

    def test_empty_string(self):
        result = _parse_xml("", "cmd")
        assert not result.success


# ---------------------------------------------------------------------------
# NmapTool — subprocess integration tests (mocked)
# ---------------------------------------------------------------------------


def _make_proc(stdout: str = "", stderr: str = "", returncode: int = 0) -> MagicMock:
    proc = MagicMock()
    proc.stdout = stdout
    proc.stderr = stderr
    proc.returncode = returncode
    return proc


class TestNmapToolScanSuccess:
    def test_happy_path(self):
        tool = NmapTool()
        with patch("tools.nmap.nmap_tool.subprocess.run", return_value=_make_proc(stdout=SIMPLE_XML)) as mock_run:
            result = tool.scan("127.0.0.1")

        assert result.success
        assert len(result.hosts) == 1
        mock_run.assert_called_once()

    def test_command_line_includes_sV_and_oX(self):
        tool = NmapTool()
        with patch("tools.nmap.nmap_tool.subprocess.run", return_value=_make_proc(stdout=SIMPLE_XML)) as mock_run:
            tool.scan("127.0.0.1")

        args = mock_run.call_args[0][0]
        assert "-sV" in args
        assert "-oX" in args
        assert "-" in args
        assert "127.0.0.1" in args

    def test_extra_args_forwarded(self):
        tool = NmapTool()
        with patch("tools.nmap.nmap_tool.subprocess.run", return_value=_make_proc(stdout=SIMPLE_XML)) as mock_run:
            tool.scan("10.0.0.1", extra_args=["--top-ports", "100"])

        args = mock_run.call_args[0][0]
        assert "--top-ports" in args
        assert "100" in args

    def test_target_is_last_arg(self):
        tool = NmapTool()
        with patch("tools.nmap.nmap_tool.subprocess.run", return_value=_make_proc(stdout=SIMPLE_XML)):
            result = tool.scan("10.0.0.1")

        assert result.command.endswith("10.0.0.1")


class TestNmapToolErrors:
    def test_nmap_not_found(self):
        tool = NmapTool(nmap_path="/nonexistent/nmap")
        with patch("tools.nmap.nmap_tool.subprocess.run", side_effect=FileNotFoundError):
            result = tool.scan("127.0.0.1")

        assert not result.success
        assert "not found" in (result.error or "").lower()

    def test_nonzero_exit_code(self):
        tool = NmapTool()
        with patch("tools.nmap.nmap_tool.subprocess.run",
                   return_value=_make_proc(returncode=1, stderr="Permission denied")):
            result = tool.scan("127.0.0.1")

        assert not result.success
        assert "Permission denied" in (result.error or "")

    def test_timeout(self):
        tool = NmapTool(timeout=1)
        with patch("tools.nmap.nmap_tool.subprocess.run",
                   side_effect=subprocess.TimeoutExpired(cmd="nmap", timeout=1)):
            result = tool.scan("127.0.0.1")

        assert not result.success
        assert "timed out" in (result.error or "").lower()

    def test_empty_stdout(self):
        tool = NmapTool()
        with patch("tools.nmap.nmap_tool.subprocess.run", return_value=_make_proc(stdout="")):
            result = tool.scan("127.0.0.1")

        assert not result.success
        assert "no output" in (result.error or "").lower()

    def test_os_error(self):
        tool = NmapTool()
        with patch("tools.nmap.nmap_tool.subprocess.run", side_effect=OSError("Permission denied")):
            result = tool.scan("127.0.0.1")

        assert not result.success
        assert "OS error" in (result.error or "")

    def test_malformed_xml_output(self):
        tool = NmapTool()
        with patch("tools.nmap.nmap_tool.subprocess.run",
                   return_value=_make_proc(stdout="<broken <xml")):
            result = tool.scan("127.0.0.1")

        assert not result.success
        assert "XML parse error" in (result.error or "")


# ---------------------------------------------------------------------------
# Model property tests
# ---------------------------------------------------------------------------


class TestNmapHostOpenPorts:
    def test_open_ports_filter(self):
        host = NmapHost(
            address="1.2.3.4",
            address_type="ipv4",
            state="up",
            ports=[
                NmapPort(port=22, protocol="tcp", state="open"),
                NmapPort(port=443, protocol="tcp", state="filtered"),
                NmapPort(port=80, protocol="tcp", state="open"),
            ],
        )
        open_ports = host.open_ports
        assert len(open_ports) == 2
        assert all(p.state == "open" for p in open_ports)

    def test_open_ports_empty_when_no_ports(self):
        host = NmapHost(address="1.2.3.4", address_type="ipv4", state="up")
        assert host.open_ports == []


class TestNmapScanResultSuccess:
    def test_success_true_when_no_error(self):
        r = NmapScanResult(command="cmd")
        assert r.success is True

    def test_success_false_when_error_set(self):
        r = NmapScanResult(command="cmd", error="something went wrong")
        assert r.success is False


class TestNmapServiceDefaults:
    def test_empty_optional_fields(self):
        svc = NmapService(name="http")
        assert svc.product == ""
        assert svc.version == ""
        assert svc.extra_info == ""
        assert svc.confidence == 0
