"""Argus recon agent.

Orchestrates subfinder → httpx → nmap against a single target domain and
returns a ReconReport.  No AI logic, no prompts, no UI interaction.

Error contract
--------------
ArgusAgent.run() never raises.  Every tool failure is caught, recorded in
ReconReport.errors, and execution continues with the next step where possible:

* Subfinder failure  → error recorded; target domain used as fallback.
* HTTPX failure      → error recorded; live_hosts stays empty.
* Nmap failure       → error recorded per host; host added with empty ports.

Observable state
----------------
Pass a ScanSession to run() and it will be updated at each phase boundary so
callers can read progress without waiting for completion.

Parallel nmap
-------------
Port scans run concurrently across live hosts (bounded by max_workers=5).
Each scan runs in its own thread; results are collected after all futures
complete.  The rest of the pipeline stays single-threaded.
"""
from __future__ import annotations

import concurrent.futures
import time
from typing import TYPE_CHECKING, Optional

from tools.httpx.httpx_tool import HttpxProbeResult, HttpxScanResult, HttpxTool
from tools.nmap.nmap_tool import NmapScanResult, NmapTool
from tools.subfinder.subfinder_tool import SubfinderScanResult, SubfinderTool

from .models import ReconHost, ReconPort, ReconReport

if TYPE_CHECKING:
    from .scan_registry import ScanSession


# ---------------------------------------------------------------------------
# Highlight extraction
# ---------------------------------------------------------------------------

_NOTABLE_PORTS: dict[int, str] = {
    21: "FTP (cleartext)",
    22: "SSH",
    23: "Telnet (cleartext)",
    3389: "RDP",
}

_NOTABLE_HIGH_PORTS: frozenset[int] = frozenset({3000, 4000, 8080, 8443, 8888, 9000, 9090})

_NOTABLE_LABELS: frozenset[str] = frozenset({
    "dev", "development", "staging", "stage", "test", "qa", "uat",
    "admin", "manage", "management", "dashboard", "portal", "beta",
})


def _compute_highlights(live_hosts: list[ReconHost]) -> list[str]:
    """Return up to 10 human-readable strings flagging interesting findings."""
    highlights: list[str] = []

    for host in live_hosts:
        first_label = host.hostname.split(".")[0].lower() if host.hostname else ""
        if first_label in _NOTABLE_LABELS:
            highlights.append(f"{host.hostname} — {first_label} surface exposed")

        for p in host.ports:
            if p.state != "open":
                continue
            desc = _NOTABLE_PORTS.get(p.port)
            if desc:
                highlights.append(f"{host.hostname}:{p.port} — {desc}")
            elif p.port in _NOTABLE_HIGH_PORTS:
                svc = f" ({p.service})" if p.service else ""
                highlights.append(f"{host.hostname}:{p.port} — non-standard port{svc}")

    return highlights[:10]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ArgusAgent:
    """Passive recon orchestrator.

    Inject tool instances to override binaries or supply mocks in tests.
    Default construction resolves each binary from PATH at first use.
    """

    def __init__(
        self,
        subfinder: Optional[SubfinderTool] = None,
        httpx: Optional[HttpxTool] = None,
        nmap: Optional[NmapTool] = None,
    ) -> None:
        self._subfinder: SubfinderTool = subfinder if subfinder is not None else SubfinderTool()
        self._httpx: HttpxTool = httpx if httpx is not None else HttpxTool()
        self._nmap: NmapTool = nmap if nmap is not None else NmapTool()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        target: str,
        session: Optional["ScanSession"] = None,
    ) -> ReconReport:
        """Run the full recon workflow against *target* and return a report.

        If *session* is supplied it is updated in-place as phases complete,
        allowing callers to observe progress without blocking.

        Args:
            target:  Apex domain to recon (e.g. ``example.com``).
            session: Optional ScanSession to update at each phase boundary.

        Returns:
            ``ReconReport`` — always returned, never raises.
        """
        errors: list[str] = []
        discovered_domains: list[str] = []
        live_hosts: list[ReconHost] = []

        # ------------------------------------------------------------------
        # Phase 1 — subdomain enumeration
        # ------------------------------------------------------------------
        if session:
            session.phase = "subfinder"
            session.summary = f"enumerating subdomains of {target}..."

        try:
            sf_result = self._subfinder.scan(target)
        except Exception as exc:
            sf_result = SubfinderScanResult(command="subfinder", error=str(exc))

        if not sf_result.success:
            errors.append(f"subfinder: {sf_result.error}")
            discovered_domains = [target]
        else:
            discovered_domains = [d.host for d in sf_result.domains] or [target]

        if session:
            session.discovered_domains = len(discovered_domains)
            session.phase = "httpx"
            session.summary = (
                f"{len(discovered_domains)} subdomains found, probing live hosts..."
            )

        # ------------------------------------------------------------------
        # Phase 2 — HTTP/S probe
        # ------------------------------------------------------------------
        try:
            hx_result = self._httpx.probe(discovered_domains)
        except Exception as exc:
            hx_result = HttpxScanResult(command="httpx", error=str(exc))

        if not hx_result.success:
            errors.append(f"httpx: {hx_result.error}")
            if session:
                session.status = "complete"
                session.phase = "done"
                session.errors = list(errors)
                session.summary = f"httpx failed: {hx_result.error}"
                session.completed_at = time.time()
            return ReconReport(
                target=target,
                discovered_domains=discovered_domains,
                live_hosts=[],
                errors=errors,
            )

        live_probes = hx_result.live

        if session:
            session.live_count = len(live_probes)
            session.phase = "nmap"
            session.summary = (
                f"{len(discovered_domains)} subdomains, "
                f"{len(live_probes)} live — scanning ports..."
            )

        # ------------------------------------------------------------------
        # Phase 3 — parallel port scan
        # ------------------------------------------------------------------
        live_hosts = self._scan_hosts_parallel(live_probes, errors)

        port_count = sum(len(h.ports) for h in live_hosts)

        if session:
            session.port_count = port_count
            session.live_count = len(live_hosts)
            session.status = "complete"
            session.phase = "done"
            session.highlights = _compute_highlights(live_hosts)
            session.errors = list(errors)
            session.completed_at = time.time()
            session.summary = (
                f"complete — {len(live_hosts)} live hosts, {port_count} open ports"
            )

        return ReconReport(
            target=target,
            discovered_domains=discovered_domains,
            live_hosts=live_hosts,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _scan_hosts_parallel(
        self,
        probes: list[HttpxProbeResult],
        errors: list[str],
        max_workers: int = 5,
    ) -> list[ReconHost]:
        """Scan each probe host with nmap, running up to *max_workers* concurrently.

        Always returns one ReconHost per probe — hosts whose nmap scan fails
        are included with an empty port list (matching the sequential contract).
        The corresponding error string is appended to *errors*.
        """
        if not probes:
            return []

        def _scan_one(probe: HttpxProbeResult) -> tuple[ReconHost, Optional[str]]:
            nmap_target = probe.host or probe.input
            error: Optional[str] = None

            try:
                nm_result = self._nmap.scan(nmap_target)
            except Exception as exc:
                nm_result = NmapScanResult(command="nmap", error=str(exc))

            ports: list[ReconPort] = []
            if not nm_result.success:
                error = f"nmap({nmap_target}): {nm_result.error}"
            else:
                for nmap_host in nm_result.hosts:
                    for p in nmap_host.open_ports:
                        ports.append(
                            ReconPort(
                                port=p.port,
                                protocol=p.protocol,
                                state=p.state,
                                service=p.service.name if p.service else "",
                            )
                        )

            host = ReconHost(
                hostname=probe.input or probe.url,
                live=True,
                ports=ports,
                technologies=list(probe.tech),
            )
            return host, error

        live_hosts: list[ReconHost] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_scan_one, probe) for probe in probes]
            for fut in concurrent.futures.as_completed(futures):
                host, error = fut.result()
                live_hosts.append(host)
                if error:
                    errors.append(error)

        return live_hosts
