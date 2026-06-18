"""Argus Phase 1 — subfinder wrapper (projectdiscovery/subfinder).

Runs ``subfinder -silent -json -d <domain>`` and returns strongly-typed
Pydantic models parsed from the NDJSON output.
No AI logic is included; this is a pure enumeration-and-parse utility.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class SubfinderDomain(BaseModel):
    """A single subdomain discovered by subfinder."""

    host: str
    input: str = ""
    source: str = ""

    @property
    def root_domain(self) -> str:
        """Return the two-label root domain (naive; single-part TLDs only).

        Examples:
            ``www.example.com`` → ``example.com``
            ``api.v2.example.com`` → ``example.com``
            ``example.com`` → ``example.com``
        """
        if not self.host:
            return ""
        parts = self.host.split(".")
        return ".".join(parts[-2:]) if len(parts) >= 2 else self.host


class SubfinderScanResult(BaseModel):
    """Aggregated result of a subfinder enumeration run."""

    command: str
    domains: list[SubfinderDomain] = Field(default_factory=list)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def count(self) -> int:
        return len(self.domains)


# ---------------------------------------------------------------------------
# NDJSON parsing
# ---------------------------------------------------------------------------


def _coerce_str(value: object) -> str:
    return str(value) if value is not None else ""


def _parse_domain(data: dict) -> SubfinderDomain:  # type: ignore[type-arg]
    return SubfinderDomain(
        host=_coerce_str(data.get("host")),
        input=_coerce_str(data.get("input")),
        source=_coerce_str(data.get("source")),
    )


def _parse_ndjson(output: str, command: str) -> SubfinderScanResult:
    domains: list[SubfinderDomain] = []
    parse_errors: list[str] = []

    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    if not lines:
        return SubfinderScanResult(command=command, error="subfinder produced no parseable output")

    for line in lines:
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            parse_errors.append(f"JSON decode error: {exc}")
            continue
        if not isinstance(data, dict):
            parse_errors.append(f"Unexpected JSON type on line: {type(data).__name__}")
            continue
        domains.append(_parse_domain(data))

    if parse_errors and not domains:
        return SubfinderScanResult(command=command, error="; ".join(parse_errors))

    return SubfinderScanResult(command=command, domains=domains)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT = 120  # seconds — passive DNS enumeration can take time
_BASE_FLAGS = ["-silent", "-json"]


class SubfinderTool:
    """Thin wrapper around the projectdiscovery subfinder CLI.

    Pass ``subfinder_path`` to point at a non-PATH binary; otherwise the tool
    resolves the binary lazily at scan time so construction never raises.
    """

    def __init__(
        self,
        subfinder_path: Optional[str] = None,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        self._subfinder = subfinder_path or shutil.which("subfinder") or "subfinder"
        self._timeout = timeout

    def scan(
        self,
        domain: str,
        extra_args: Optional[list[str]] = None,
    ) -> SubfinderScanResult:
        """Enumerate subdomains with ``subfinder -silent -json -d <domain>``.

        Args:
            domain: The apex domain to enumerate (e.g. ``example.com``).
            extra_args: Additional subfinder flags appended after the base flags.

        Returns:
            ``SubfinderScanResult`` — ``result.success`` is ``False`` and
            ``result.error`` is set when the run cannot complete.
            Partial-parse failures (bad JSON lines) are silently dropped
            as long as at least one domain parses successfully.
        """
        args = [self._subfinder] + _BASE_FLAGS + ["-d", domain] + (extra_args or [])
        command = " ".join(args)

        try:
            proc = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
        except FileNotFoundError:
            return SubfinderScanResult(
                command=command,
                error="subfinder executable not found — install projectdiscovery/subfinder and ensure it is on PATH",
            )
        except subprocess.TimeoutExpired:
            return SubfinderScanResult(
                command=command,
                error=f"subfinder scan timed out after {self._timeout}s",
            )
        except OSError as exc:
            return SubfinderScanResult(command=command, error=f"OS error launching subfinder: {exc}")

        if proc.returncode != 0:
            stderr = proc.stderr.strip() or "(no stderr)"
            return SubfinderScanResult(
                command=command,
                error=f"subfinder exited with code {proc.returncode}: {stderr}",
            )

        if not proc.stdout.strip():
            return SubfinderScanResult(command=command, error="subfinder produced no output")

        return _parse_ndjson(proc.stdout, command)
