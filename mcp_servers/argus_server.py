"""
argus_server.py

MCP server exposing Argus recon tools to the Odysseus agent loop.

Three tools
-----------
argus_recon(target)
    Full pipeline: subfinder → httpx → nmap.  Returns a ReconReport.

argus_scan_host(target)
    Single-host port scan via nmap only.  Returns a NmapScanResult.

argus_enumerate_subdomains(domain)
    Subdomain enumeration via subfinder only.  Returns a SubfinderScanResult.

Response format
---------------
All tools return compact JSON (no pretty-print) as a single TextContent item.
Lists are capped before serialisation so output stays under MAX_OUTPUT_CHARS:
  - discovered_domains  → first 200 (argus_recon)
  - live_hosts          → first 20  (argus_recon)
  - domains             → first 200 (argus_enumerate_subdomains)

Error contract
--------------
Never raises out of call_tool().  Tool-level errors are surfaced as JSON with
an "error" key mirroring the tool-layer never-raise contract.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

server = Server("argus")

_agent = None          # ArgusAgent  — set by _ensure_init()
_initialized = False   # guard so _ensure_init() runs once


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
# Tool handlers
# ---------------------------------------------------------------------------

_DOMAIN_LIMIT = 200
_HOST_LIMIT = 20


async def _handle_argus_recon(arguments: dict) -> list[TextContent]:
    target = (arguments.get("target") or "").strip()
    if not target:
        return _json({"error": "target is required"})

    _ensure_init()
    report = await asyncio.to_thread(_agent.run, target)

    data = report.model_dump()
    total_domains = len(data["discovered_domains"])
    total_hosts = len(data["live_hosts"])

    if total_domains > _DOMAIN_LIMIT:
        data["discovered_domains"] = data["discovered_domains"][:_DOMAIN_LIMIT]
        data["_domains_truncated"] = f"{total_domains} total, showing first {_DOMAIN_LIMIT}"

    if total_hosts > _HOST_LIMIT:
        data["live_hosts"] = data["live_hosts"][:_HOST_LIMIT]
        data["_hosts_truncated"] = f"{total_hosts} total, showing first {_HOST_LIMIT}"

    return _json(data)


async def _handle_argus_scan_host(arguments: dict) -> list[TextContent]:
    target = (arguments.get("target") or "").strip()
    if not target:
        return _json({"error": "target is required"})

    _ensure_init()
    from tools.nmap import NmapTool
    nmap = NmapTool()
    result = await asyncio.to_thread(nmap.scan, target)
    return _json(result.model_dump())


async def _handle_argus_enumerate_subdomains(arguments: dict) -> list[TextContent]:
    domain = (arguments.get("domain") or "").strip()
    if not domain:
        return _json({"error": "domain is required"})

    _ensure_init()
    from tools.subfinder import SubfinderTool
    sf = SubfinderTool()
    result = await asyncio.to_thread(sf.scan, domain)

    data = result.model_dump()
    total = len(data["domains"])
    if total > _DOMAIN_LIMIT:
        data["domains"] = data["domains"][:_DOMAIN_LIMIT]
        data["_domains_truncated"] = f"{total} total, showing first {_DOMAIN_LIMIT}"

    return _json(data)


# ---------------------------------------------------------------------------
# MCP server wiring
# ---------------------------------------------------------------------------

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="argus_recon",
            description=(
                "Full passive recon pipeline against a target domain: "
                "enumerate subdomains (subfinder), probe live HTTP/S hosts (httpx), "
                "then port-scan each live host (nmap). Returns a JSON ReconReport."
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
                "Port-scan a single host or IP address with nmap (-sV). "
                "Returns a JSON NmapScanResult with open ports and service versions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Hostname or IP address to scan, e.g. 93.184.216.34",
                    },
                },
                "required": ["target"],
            },
        ),
        Tool(
            name="argus_enumerate_subdomains",
            description=(
                "Enumerate subdomains for a domain using subfinder. "
                "Returns a JSON SubfinderScanResult with discovered hostnames."
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
        return _json({"error": f"Unknown tool: {name}"})
    except Exception as exc:  # safety net — MCP call_tool must not raise
        return _json({"error": str(exc)})


if __name__ == "__main__":
    asyncio.run(stdio_server(server).run())
