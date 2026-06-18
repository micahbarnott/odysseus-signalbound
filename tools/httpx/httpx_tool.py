"""Argus Phase 1 — httpx wrapper (projectdiscovery/httpx).

Runs ``httpx -json -silent`` against one or more targets and returns
strongly-typed Pydantic models parsed from the NDJSON output.
No AI logic is included; this is a pure probe-and-parse utility.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from typing import Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class HttpxProbeResult(BaseModel):
    """Result for a single probed URL/host."""

    url: str
    input: str = ""
    final_url: str = ""
    scheme: str = ""
    host: str = ""
    port: str = ""
    path: str = ""
    method: str = ""
    status_code: Optional[int] = None
    content_type: str = ""
    content_length: int = 0
    title: str = ""
    webserver: str = ""
    tech: list[str] = Field(default_factory=list)
    words: int = 0
    lines: int = 0
    response_time: str = ""
    failed: bool = False
    chain_status_codes: list[int] = Field(default_factory=list)
    response_headers: dict[str, str] = Field(default_factory=dict)
    a_records: list[str] = Field(default_factory=list)

    @property
    def is_live(self) -> bool:
        return not self.failed and self.status_code is not None

    @property
    def is_redirect(self) -> bool:
        return self.status_code is not None and 300 <= self.status_code < 400


class HttpxScanResult(BaseModel):
    """Aggregated result of an httpx probe run."""

    command: str
    probes: list[HttpxProbeResult] = Field(default_factory=list)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def live(self) -> list[HttpxProbeResult]:
        return [p for p in self.probes if p.is_live]


# ---------------------------------------------------------------------------
# NDJSON parsing
# ---------------------------------------------------------------------------


def _coerce_int(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _coerce_str(value: object) -> str:
    return str(value) if value is not None else ""


def _parse_probe(data: dict) -> HttpxProbeResult:  # type: ignore[type-arg]
    # httpx has emitted both hyphenated and underscored key names across
    # versions; accept either form for every field that varies.
    def pick(*keys: str) -> object:
        for k in keys:
            v = data.get(k)
            if v is not None:
                return v
        return None

    status_raw = pick("status_code", "status-code")
    status_code: Optional[int] = None
    if status_raw is not None:
        try:
            status_code = int(status_raw)
        except (TypeError, ValueError):
            pass

    headers_raw = pick("response_headers", "response-headers")
    headers: dict[str, str] = {}
    if isinstance(headers_raw, dict):
        headers = {str(k): str(v) for k, v in headers_raw.items()}

    tech_raw = pick("tech", "technologies") or []
    tech: list[str] = [str(t) for t in tech_raw] if isinstance(tech_raw, list) else []

    chain_raw = pick("chain_status_codes", "chain-status-codes") or []
    chain: list[int] = []
    if isinstance(chain_raw, list):
        for c in chain_raw:
            try:
                chain.append(int(c))
            except (TypeError, ValueError):
                pass

    a_raw = pick("a") or []
    a_records: list[str] = [str(a) for a in a_raw] if isinstance(a_raw, list) else []

    return HttpxProbeResult(
        url=_coerce_str(pick("url")),
        input=_coerce_str(pick("input")),
        final_url=_coerce_str(pick("final_url", "final-url")),
        scheme=_coerce_str(pick("scheme")),
        host=_coerce_str(pick("host")),
        port=_coerce_str(pick("port")),
        path=_coerce_str(pick("path")),
        method=_coerce_str(pick("method")),
        status_code=status_code,
        content_type=_coerce_str(pick("content_type", "content-type")),
        content_length=_coerce_int(pick("content_length", "content-length")),
        title=_coerce_str(pick("title")),
        webserver=_coerce_str(pick("webserver")),
        tech=tech,
        words=_coerce_int(pick("words")),
        lines=_coerce_int(pick("lines")),
        response_time=_coerce_str(pick("time")),
        failed=bool(data.get("failed", False)),
        chain_status_codes=chain,
        response_headers=headers,
        a_records=a_records,
    )


def _parse_ndjson(output: str, command: str) -> HttpxScanResult:
    probes: list[HttpxProbeResult] = []
    parse_errors: list[str] = []

    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    if not lines:
        return HttpxScanResult(command=command, error="httpx produced no parseable output")

    for line in lines:
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            parse_errors.append(f"JSON decode error: {exc}")
            continue
        if not isinstance(data, dict):
            parse_errors.append(f"Unexpected JSON type on line: {type(data).__name__}")
            continue
        probes.append(_parse_probe(data))

    if parse_errors and not probes:
        return HttpxScanResult(command=command, error="; ".join(parse_errors))

    return HttpxScanResult(command=command, probes=probes)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT = 60  # seconds — httpx probes are fast; keep it tight
_BASE_FLAGS = ["-json", "-silent", "-no-color"]


class HttpxTool:
    """Thin wrapper around the projectdiscovery httpx CLI.

    Pass ``httpx_path`` to point at a non-PATH binary; otherwise the tool
    resolves the binary lazily at probe time so construction never raises.
    """

    def __init__(
        self,
        httpx_path: Optional[str] = None,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        self._httpx = httpx_path or shutil.which("httpx") or "httpx"
        self._timeout = timeout

    def probe(
        self,
        targets: Union[str, list[str]],
        extra_args: Optional[list[str]] = None,
    ) -> HttpxScanResult:
        """Probe one or more targets with ``httpx -json -silent``.

        Args:
            targets: A single URL/host string or a list of them.
            extra_args: Additional httpx flags appended after the base flags.

        Returns:
            ``HttpxScanResult`` — ``result.success`` is ``False`` and
            ``result.error`` is set when the run cannot complete.
            Partial-parse failures (bad JSON lines) are silently dropped
            as long as at least one probe parses successfully.
        """
        extra = extra_args or []

        if isinstance(targets, str):
            args = [self._httpx] + _BASE_FLAGS + ["-u", targets] + extra
            stdin_input: Optional[str] = None
        else:
            args = [self._httpx] + _BASE_FLAGS + ["-l", "-"] + extra
            stdin_input = "\n".join(targets)

        command = " ".join(args)

        try:
            proc = subprocess.run(
                args,
                input=stdin_input,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
        except FileNotFoundError:
            return HttpxScanResult(
                command=command,
                error="httpx executable not found — install projectdiscovery/httpx and ensure it is on PATH",
            )
        except subprocess.TimeoutExpired:
            return HttpxScanResult(
                command=command,
                error=f"httpx probe timed out after {self._timeout}s",
            )
        except OSError as exc:
            return HttpxScanResult(command=command, error=f"OS error launching httpx: {exc}")

        if proc.returncode != 0:
            stderr = proc.stderr.strip() or "(no stderr)"
            return HttpxScanResult(
                command=command,
                error=f"httpx exited with code {proc.returncode}: {stderr}",
            )

        if not proc.stdout.strip():
            return HttpxScanResult(command=command, error="httpx produced no output")

        return _parse_ndjson(proc.stdout, command)
