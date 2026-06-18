"""Argus agent data models.

These are agent-layer types — they sit above the tool layer and summarise
results from subfinder, httpx, and nmap into a single coherent report.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

_WEB_PORTS: frozenset[int] = frozenset({80, 443, 8080, 8443})


class ReconPort(BaseModel):
    """A single port observed on a live host."""

    port: int
    protocol: str
    state: str
    service: str = ""


class ReconHost(BaseModel):
    """A host that responded to an HTTP/S probe and was subsequently port-scanned."""

    hostname: str
    live: bool = False
    ports: list[ReconPort] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)

    @property
    def is_web(self) -> bool:
        """True when the host serves HTTP/S traffic.

        Triggered by an open port in the standard web-service set
        (80, 443, 8080, 8443) or by any technology fingerprint returned
        by httpx (which only fingerprints over HTTP).
        """
        has_open_web_port = any(
            p.port in _WEB_PORTS and p.state == "open" for p in self.ports
        )
        return has_open_web_port or bool(self.technologies)


class ReconReport(BaseModel):
    """Top-level output produced by ArgusAgent.run()."""

    target: str
    discovered_domains: list[str] = Field(default_factory=list)
    live_hosts: list[ReconHost] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @property
    def success(self) -> bool:
        """True when every tool step completed without error."""
        return len(self.errors) == 0

    @property
    def host_count(self) -> int:
        """Number of domains discovered by subfinder (or fallback)."""
        return len(self.discovered_domains)

    @property
    def live_count(self) -> int:
        """Number of hosts that responded to the HTTP/S probe."""
        return len(self.live_hosts)

    @property
    def port_count(self) -> int:
        """Total number of ports recorded across all live hosts."""
        return sum(len(h.ports) for h in self.live_hosts)
