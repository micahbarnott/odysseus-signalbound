"""
argus_server.py

MCP server exposing Argus recon tools to the Odysseus agent loop.

Four tools
----------
argus_recon(target)
    Start a full pipeline scan (subfinder → httpx → nmap).  Returns a
    scan_id immediately; the scan runs in a background thread.

argus_scan_host(target)
    Start an nmap-only scan of a single host or IP.  Returns scan_id
    immediately; runs in a background thread.

argus_enumerate_subdomains(domain)
    Start a subfinder-only subdomain enumeration.  Returns scan_id
    immediately; runs in a background thread.

argus_get_scan(scan_id)
    Return current state of any scan: phase, summary, counts, highlights.
    Instant lookup — never blocks.

Observable lifecycle
--------------------
Every start tool returns:
    {"scan_id": "argus_xxxxxxxx", "status": "running", "phase": "...", ...}

The scan updates an in-memory ScanSession while it runs.  argus_get_scan
reads that session at any point to report progress or final results.

Response format
---------------
All responses are compact JSON (no raw tool dumps).  The LLM sees counts,
highlights, and a plain-English summary — never multi-kilobyte data blobs.
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

server = Server("argus")

_agent = None
_initialized = False


def _ensure_init() -> None:
    global _agent, _initialized
    if _initialized:
        return
    _initialized = True
    from agents.argus import ArgusAgent
    _agent = ArgusAgent()


def _text(text: str) -> list[TextContent]:
    return [TextContent(type="text", text=text)]


def _json(obj) -> list[TextContent]:
    return _text(json.dumps(obj, default=str))


# ---------------------------------------------------------------------------
# Highlight helpers for single-tool scans
# ---------------------------------------------------------------------------

_NOTABLE_PORTS: dict[int, str] = {
    21: "FTP (cleartext)",
    22: "SSH",
    23: "Telnet (cleartext)",
    3389: "RDP",
}
_NOTABLE_HIGH_PORTS: frozenset[int] = frozenset({3000, 4000, 8080, 8443, 8888, 9000, 9090})


def _nmap_highlights(result) -> list[str]:
    """Extract notable findings from a NmapScanResult."""
    highlights: list[str] = []
    for nmap_host in result.hosts:
        for p in nmap_host.open_ports:
            desc = _NOTABLE_PORTS.get(p.port)
            if desc:
                highlights.append(f":{p.port} — {desc}")
            elif p.port in _NOTABLE_HIGH_PORTS:
                svc = f" ({p.service.name})" if p.service else ""
                highlights.append(f":{p.port} — non-standard port{svc}")
    return highlights[:10]


# ---------------------------------------------------------------------------
# Background scan workers — run in daemon threads, update session in-place
# ---------------------------------------------------------------------------

def _run_scan_sync(session) -> None:
    """Full recon pipeline worker.  Called from a background daemon thread."""
    try:
        _agent.run(session.target, session)
        # Ensure completion is marked even when the agent didn't update session
        # (e.g. in tests where _agent is a mock that ignores the session arg).
        if session.status == "running":
            session.status = "complete"
            session.phase = "done"
            session.completed_at = time.time()
            if not session.summary or "starting" in session.summary:
                session.summary = "complete"
    except Exception as exc:
        session.status = "failed"
        session.phase = "done"
        session.summary = f"failed: {exc}"
        errs = list(session.errors)
        if str(exc) not in errs:
            errs.append(str(exc))
        session.errors = errs
        session.completed_at = time.time()


def _run_host_scan_sync(session, target: str) -> None:
    """Single-host nmap worker.  Called from a background daemon thread."""
    try:
        from tools.nmap import NmapTool
        nmap = NmapTool()
        result = nmap.scan(target)
        if result.success:
            port_count = sum(len(h.open_ports) for h in result.hosts)
            session.port_count = port_count
            session.highlights = _nmap_highlights(result)
            session.summary = f"complete — {port_count} open ports on {target}"
        else:
            session.errors = [result.error or "scan failed"]
            session.summary = f"nmap: {result.error}"
        session.status = "complete"
        session.phase = "done"
        session.completed_at = time.time()
        session.report = result
    except Exception as exc:
        session.status = "failed"
        session.phase = "done"
        session.summary = f"failed: {exc}"
        session.errors = [str(exc)]
        session.completed_at = time.time()


def _run_enum_sync(session, domain: str) -> None:
    """Subfinder enumeration worker.  Called from a background daemon thread."""
    try:
        from tools.subfinder import SubfinderTool
        sf = SubfinderTool()
        result = sf.scan(domain)
        if result.success:
            session.discovered_domains = len(result.domains)
            session.summary = f"complete — {len(result.domains)} subdomains found"
        else:
            session.errors = [result.error or "enumeration failed"]
            session.summary = f"subfinder: {result.error}"
        session.status = "complete"
        session.phase = "done"
        session.completed_at = time.time()
        session.report = result
    except Exception as exc:
        session.status = "failed"
        session.phase = "done"
        session.summary = f"failed: {exc}"
        session.errors = [str(exc)]
        session.completed_at = time.time()


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

async def _handle_argus_recon(arguments: dict) -> list[TextContent]:
    target = (arguments.get("target") or "").strip()
    if not target:
        return _json({"error": "target is required"})

    _ensure_init()

    from agents.argus.scan_registry import create_session
    session = create_session(target)

    t = threading.Thread(target=_run_scan_sync, args=(session,), daemon=True)
    session.thread = t
    initial = session.to_response()  # snapshot before thread can mutate session
    t.start()

    return _json(initial)


async def _handle_argus_scan_host(arguments: dict) -> list[TextContent]:
    target = (arguments.get("target") or "").strip()
    if not target:
        return _json({"error": "target is required"})

    from agents.argus.scan_registry import create_session
    session = create_session(target)
    session.phase = "nmap"
    session.summary = f"scanning ports on {target}..."

    t = threading.Thread(target=_run_host_scan_sync, args=(session, target), daemon=True)
    session.thread = t
    initial = session.to_response()
    t.start()

    return _json(initial)


async def _handle_argus_enumerate_subdomains(arguments: dict) -> list[TextContent]:
    domain = (arguments.get("domain") or "").strip()
    if not domain:
        return _json({"error": "domain is required"})

    from agents.argus.scan_registry import create_session
    session = create_session(domain)
    session.phase = "subfinder"
    session.summary = f"enumerating subdomains of {domain}..."

    t = threading.Thread(target=_run_enum_sync, args=(session, domain), daemon=True)
    session.thread = t
    initial = session.to_response()
    t.start()

    return _json(initial)


async def _handle_argus_get_scan(arguments: dict) -> list[TextContent]:
    scan_id = (arguments.get("scan_id") or "").strip()
    if not scan_id:
        return _json({"error": "scan_id is required"})

    from agents.argus.scan_registry import get_session
    session = get_session(scan_id)
    if not session:
        return _json({"error": f"scan '{scan_id}' not found"})

    return _json(session.to_response())


# ---------------------------------------------------------------------------
# MCP server wiring
# ---------------------------------------------------------------------------

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="argus_recon",
            description=(
                "Start a full passive recon scan on a target domain: "
                "enumerate subdomains (subfinder), probe live HTTP/S hosts (httpx), "
                "then port-scan each live host (nmap). "
                "Returns a scan_id immediately; use argus_get_scan to check progress."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Apex domain to recon, e.g. example.com",
                    },
                },
                "required": ["target"],
            },
        ),
        Tool(
            name="argus_scan_host",
            description=(
                "Start an nmap port scan of a single host or IP address (-sV). "
                "Returns a scan_id immediately; use argus_get_scan to check progress."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Hostname or IP to scan, e.g. 93.184.216.34",
                    },
                },
                "required": ["target"],
            },
        ),
        Tool(
            name="argus_enumerate_subdomains",
            description=(
                "Start a subfinder subdomain enumeration for a domain. "
                "Returns a scan_id immediately; use argus_get_scan to check progress."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "Apex domain to enumerate, e.g. example.com",
                    },
                },
                "required": ["domain"],
            },
        ),
        Tool(
            name="argus_get_scan",
            description=(
                "Return the current state of any Argus scan: phase, summary, "
                "counts, and highlights. Instant lookup — never blocks. "
                "Call repeatedly to poll progress, or once after completion."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "scan_id": {
                        "type": "string",
                        "description": "scan_id returned by a previous argus_* start call",
                    },
                },
                "required": ["scan_id"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "argus_recon":
            return await _handle_argus_recon(arguments)
        if name == "argus_scan_host":
            return await _handle_argus_scan_host(arguments)
        if name == "argus_enumerate_subdomains":
            return await _handle_argus_enumerate_subdomains(arguments)
        if name == "argus_get_scan":
            return await _handle_argus_get_scan(arguments)
        return _json({"error": f"Unknown tool: {name}"})
    except Exception as exc:
        return _json({"error": str(exc)})


if __name__ == "__main__":
    asyncio.run(stdio_server(server).run())
