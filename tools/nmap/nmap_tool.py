"""Argus Phase 1 — Nmap wrapper.

Runs ``nmap -sV -oX -`` and returns strongly-typed Pydantic models.
No AI logic is included; this is a pure scan-and-parse utility.
"""
from __future__ import annotations

import shutil
import subprocess
import xml.etree.ElementTree as ET
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class NmapService(BaseModel):
    name: str
    product: str = ""
    version: str = ""
    extra_info: str = ""
    confidence: int = Field(default=0, ge=0, le=10)


class NmapPort(BaseModel):
    port: int = Field(ge=0, le=65535)
    protocol: str
    state: str
    reason: str = ""
    service: Optional[NmapService] = None


class NmapHost(BaseModel):
    address: str
    address_type: str
    hostnames: list[str] = Field(default_factory=list)
    state: str
    ports: list[NmapPort] = Field(default_factory=list)

    @property
    def open_ports(self) -> list[NmapPort]:
        return [p for p in self.ports if p.state == "open"]


class NmapRunStats(BaseModel):
    elapsed: float
    hosts_up: int
    hosts_down: int
    hosts_total: int


class NmapScanResult(BaseModel):
    command: str
    version: str = ""
    hosts: list[NmapHost] = Field(default_factory=list)
    stats: Optional[NmapRunStats] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------


def _parse_service(port_elem: ET.Element) -> Optional[NmapService]:
    svc = port_elem.find("service")
    if svc is None:
        return None
    conf_raw = svc.get("conf", "0")
    try:
        conf = int(conf_raw)
    except ValueError:
        conf = 0
    return NmapService(
        name=svc.get("name", ""),
        product=svc.get("product", ""),
        version=svc.get("version", ""),
        extra_info=svc.get("extrainfo", ""),
        confidence=conf,
    )


def _parse_port(port_elem: ET.Element) -> NmapPort:
    state_elem = port_elem.find("state")
    state = state_elem.get("state", "") if state_elem is not None else ""
    reason = state_elem.get("reason", "") if state_elem is not None else ""
    try:
        portid = int(port_elem.get("portid", "0"))
    except ValueError:
        portid = 0
    return NmapPort(
        port=portid,
        protocol=port_elem.get("protocol", ""),
        state=state,
        reason=reason,
        service=_parse_service(port_elem),
    )


def _parse_host(host_elem: ET.Element) -> NmapHost:
    status = host_elem.find("status")
    state = status.get("state", "") if status is not None else ""

    address = ""
    address_type = ""
    for addr_elem in host_elem.findall("address"):
        if addr_elem.get("addrtype") in ("ipv4", "ipv6"):
            address = addr_elem.get("addr", "")
            address_type = addr_elem.get("addrtype", "")
            break

    hostnames: list[str] = []
    hn_container = host_elem.find("hostnames")
    if hn_container is not None:
        for hn in hn_container.findall("hostname"):
            name = hn.get("name", "")
            if name:
                hostnames.append(name)

    ports: list[NmapPort] = []
    ports_elem = host_elem.find("ports")
    if ports_elem is not None:
        for port_elem in ports_elem.findall("port"):
            ports.append(_parse_port(port_elem))

    return NmapHost(
        address=address,
        address_type=address_type,
        hostnames=hostnames,
        state=state,
        ports=ports,
    )


def _parse_stats(root: ET.Element) -> Optional[NmapRunStats]:
    runstats = root.find("runstats")
    if runstats is None:
        return None
    finished = runstats.find("finished")
    hosts_elem = runstats.find("hosts")
    if finished is None or hosts_elem is None:
        return None
    try:
        elapsed = float(finished.get("elapsed", "0"))
        up = int(hosts_elem.get("up", "0"))
        down = int(hosts_elem.get("down", "0"))
        total = int(hosts_elem.get("total", "0"))
    except ValueError:
        return None
    return NmapRunStats(elapsed=elapsed, hosts_up=up, hosts_down=down, hosts_total=total)


def _parse_xml(xml_text: str, command: str) -> NmapScanResult:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        return NmapScanResult(command=command, error=f"XML parse error: {exc}")

    version = root.get("version", "")
    hosts = [_parse_host(h) for h in root.findall("host")]
    stats = _parse_stats(root)

    return NmapScanResult(command=command, version=version, hosts=hosts, stats=stats)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT = 300  # seconds


class NmapTool:
    """Thin wrapper around the nmap CLI.

    Raises ``FileNotFoundError`` at construction time when nmap is absent so
    callers fail fast rather than at scan time.
    """

    def __init__(self, nmap_path: Optional[str] = None, timeout: int = _DEFAULT_TIMEOUT) -> None:
        self._nmap = nmap_path or shutil.which("nmap") or "nmap"
        self._timeout = timeout

    def scan(self, target: str, extra_args: Optional[list[str]] = None) -> NmapScanResult:
        """Run ``nmap -sV -oX - <target>`` and return parsed results.

        Args:
            target: Host, IP, or CIDR range accepted by nmap.
            extra_args: Additional nmap flags inserted before the target.

        Returns:
            ``NmapScanResult`` — ``result.success`` is ``False`` and
            ``result.error`` is set when the scan cannot complete.
        """
        args = [self._nmap, "-sV", "-oX", "-"] + (extra_args or []) + [target]
        command = " ".join(args)

        try:
            proc = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
        except FileNotFoundError:
            return NmapScanResult(
                command=command,
                error="nmap executable not found — install nmap and ensure it is on PATH",
            )
        except subprocess.TimeoutExpired:
            return NmapScanResult(
                command=command,
                error=f"nmap scan timed out after {self._timeout}s",
            )
        except OSError as exc:
            return NmapScanResult(command=command, error=f"OS error launching nmap: {exc}")

        if proc.returncode != 0:
            stderr = proc.stderr.strip() or "(no stderr)"
            return NmapScanResult(
                command=command,
                error=f"nmap exited with code {proc.returncode}: {stderr}",
            )

        if not proc.stdout.strip():
            return NmapScanResult(command=command, error="nmap produced no output")

        return _parse_xml(proc.stdout, command)
