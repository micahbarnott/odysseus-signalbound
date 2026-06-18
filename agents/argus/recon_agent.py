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
"""
from __future__ import annotations

from typing import Optional

from tools.httpx.httpx_tool import HttpxScanResult, HttpxTool
from tools.nmap.nmap_tool import NmapScanResult, NmapTool
from tools.subfinder.subfinder_tool import SubfinderScanResult, SubfinderTool

from .models import ReconHost, ReconPort, ReconReport


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

    def run(self, target: str) -> ReconReport:
        """Run the full recon workflow against *target* and return a report.

        Args:
            target: Apex domain to recon (e.g. ``example.com``).

        Returns:
            ``ReconReport`` — always returned, never raises.
        """
        errors: list[str] = []
        discovered_domains: list[str] = []
        live_hosts: list[ReconHost] = []

        # ------------------------------------------------------------------
        # Step 1 — subdomain enumeration
        # ------------------------------------------------------------------
        try:
            sf_result = self._subfinder.scan(target)
        except Exception as exc:  # safety net for unexpected tool failure
            sf_result = SubfinderScanResult(command="subfinder", error=str(exc))

        if not sf_result.success:
            errors.append(f"subfinder: {sf_result.error}")
            discovered_domains = [target]
        else:
            # Subfinder may legitimately return zero results for private targets.
            # Fall back to the target itself so httpx/nmap still have something to work with.
            discovered_domains = [d.host for d in sf_result.domains] or [target]

        # ------------------------------------------------------------------
        # Step 2 — HTTP/S probe
        # ------------------------------------------------------------------
        try:
            hx_result = self._httpx.probe(discovered_domains)
        except Exception as exc:
            hx_result = HttpxScanResult(command="httpx", error=str(exc))

        if not hx_result.success:
            errors.append(f"httpx: {hx_result.error}")
        else:
            # ------------------------------------------------------------------
            # Step 3 — port scan each live HTTP/S host
            # ------------------------------------------------------------------
            for probe in hx_result.live:
                # Prefer the resolved IP for nmap; fall back to the input domain.
                nmap_target = probe.host or probe.input

                try:
                    nm_result = self._nmap.scan(nmap_target)
                except Exception as exc:
                    nm_result = NmapScanResult(command="nmap", error=str(exc))

                ports: list[ReconPort] = []
                if not nm_result.success:
                    errors.append(f"nmap({nmap_target}): {nm_result.error}")
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

                live_hosts.append(
                    ReconHost(
                        hostname=probe.input or probe.url,
                        live=True,
                        ports=ports,
                        technologies=list(probe.tech),
                    )
                )

        return ReconReport(
            target=target,
            discovered_domains=discovered_domains,
            live_hosts=live_hosts,
            errors=errors,
        )
