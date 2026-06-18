"""
scan_registry.py

Lightweight in-memory registry of Argus scan sessions.

A ScanSession is created when a scan starts and updated in-place as each
phase completes.  No disk I/O, no database — sessions live for the lifetime
of the argus_server.py subprocess.

Thread safety
-------------
The registry dict is protected by a lock.  Individual ScanSession fields are
written by one background thread (the scanning thread) and read by the MCP
handler thread.  For simple scalar assignments the CPython GIL provides
sufficient protection; for list fields (errors, highlights) we write complete
replacement lists rather than appending from mixed threads.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ScanSession:
    """Live state for a single Argus scan."""

    scan_id: str
    target: str
    status: str        # "running" | "complete" | "failed"
    phase: str         # "subfinder" | "httpx" | "nmap" | "done"
    summary: str = ""
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    # Lightweight counters — filled in as phases complete.
    discovered_domains: int = 0
    live_count: int = 0
    port_count: int = 0

    errors: list[str] = field(default_factory=list)
    highlights: list[str] = field(default_factory=list)

    # Full tool result stored on completion; not included in to_response().
    report: Optional[Any] = field(default=None, repr=False, compare=False)

    # Background thread reference — used by tests to join before asserting.
    thread: Optional[threading.Thread] = field(
        default=None, repr=False, compare=False
    )

    def elapsed(self) -> float:
        end = self.completed_at or time.time()
        return round(end - self.started_at, 1)

    def to_response(self) -> dict:
        """Return a compact dict safe to JSON-serialise and return to the LLM."""
        resp: dict = {
            "scan_id": self.scan_id,
            "target": self.target,
            "status": self.status,
            "phase": self.phase,
            "summary": self.summary,
            "elapsed_seconds": self.elapsed(),
        }
        if self.discovered_domains:
            resp["discovered_domains"] = self.discovered_domains
        if self.live_count:
            resp["live_hosts"] = self.live_count
        if self.port_count:
            resp["open_ports"] = self.port_count
        if self.errors:
            resp["errors"] = self.errors
        if self.highlights:
            resp["highlights"] = self.highlights
        return resp


_registry: dict[str, ScanSession] = {}
_lock = threading.Lock()


def create_session(target: str) -> ScanSession:
    """Allocate a new ScanSession, register it, and return it."""
    scan_id = "argus_" + uuid.uuid4().hex[:8]
    session = ScanSession(
        scan_id=scan_id,
        target=target,
        status="running",
        phase="subfinder",
        summary=f"starting enumeration of {target}...",
    )
    with _lock:
        _registry[scan_id] = session
    return session


def get_session(scan_id: str) -> Optional[ScanSession]:
    """Return the session for scan_id, or None if not found."""
    with _lock:
        return _registry.get(scan_id)


def clear_registry() -> None:
    """Remove all sessions.  Used in tests to prevent state leakage."""
    with _lock:
        _registry.clear()
