"""Unit tests for tools/httpx/httpx_tool.py.

All tests are pure-Python: no httpx binary, no network I/O.
subprocess.run is monkey-patched wherever a real process would be invoked.
"""
from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from tools.httpx.httpx_tool import (
    HttpxProbeResult,
    HttpxScanResult,
    HttpxTool,
    _parse_ndjson,
    _parse_probe,
)

# ---------------------------------------------------------------------------
# NDJSON fixtures
# ---------------------------------------------------------------------------

_PROBE_HTTPS = {
    "url": "https://example.com",
    "input": "example.com",
    "final_url": "https://www.example.com/",
    "scheme": "https",
    "host": "93.184.216.34",
    "port": "443",
    "path": "/",
    "method": "GET",
    "status_code": 200,
    "content_type": "text/html; charset=UTF-8",
    "content_length": 1256,
    "title": "Example Domain",
    "webserver": "ECS (dcb/7EEF)",
    "tech": ["Varnish", "Amazon CloudFront"],
    "words": 298,
    "lines": 47,
    "time": "272.895ms",
    "failed": False,
    "chain_status_codes": [200],
    "response_headers": {
        "content-type": "text/html; charset=UTF-8",
        "server": "ECS (dcb/7EEF)",
    },
    "a": ["93.184.216.34"],
}

_PROBE_REDIRECT = {
    "url": "http://example.com",
    "input": "example.com",
    "final_url": "https://example.com/",
    "scheme": "http",
    "host": "93.184.216.34",
    "port": "80",
    "path": "/",
    "method": "GET",
    "status_code": 301,
    "content_type": "",
    "content_length": 0,
    "title": "",
    "webserver": "",
    "tech": [],
    "words": 0,
    "lines": 0,
    "time": "41.2ms",
    "failed": False,
    "chain_status_codes": [301, 200],
    "response_headers": {"location": "https://example.com/"},
    "a": [],
}

_PROBE_FAILED = {
    "url": "https://dead.invalid",
    "input": "dead.invalid",
    "final_url": "",
    "scheme": "https",
    "host": "",
    "port": "443",
    "path": "/",
    "method": "GET",
    "status_code": None,
    "content_type": "",
    "content_length": 0,
    "title": "",
    "webserver": "",
    "tech": [],
    "words": 0,
    "lines": 0,
    "time": "",
    "failed": True,
    "chain_status_codes": [],
    "response_headers": {},
    "a": [],
}

_PROBE_HYPHEN_KEYS = {
    "url": "https://legacy.example.com",
    "input": "legacy.example.com",
    "status-code": 200,
    "content-type": "text/html",
    "content-length": 512,
    "chain-status-codes": [200],
    "response-headers": {"server": "nginx"},
    "final-url": "https://legacy.example.com/",
    "failed": False,
    "time": "50ms",
}

SINGLE_NDJSON = json.dumps(_PROBE_HTTPS) + "\n"

MULTI_NDJSON = "\n".join([
    json.dumps(_PROBE_HTTPS),
    json.dumps(_PROBE_REDIRECT),
    json.dumps(_PROBE_FAILED),
]) + "\n"

REDIRECT_NDJSON = json.dumps(_PROBE_REDIRECT) + "\n"

FAILED_NDJSON = json.dumps(_PROBE_FAILED) + "\n"

HYPHEN_KEYS_NDJSON = json.dumps(_PROBE_HYPHEN_KEYS) + "\n"


# ---------------------------------------------------------------------------
# _parse_probe — unit tests
# ---------------------------------------------------------------------------


class TestParseProbeBasic:
    def setup_method(self):
        self.probe = _parse_probe(_PROBE_HTTPS)

    def test_url(self):
        assert self.probe.url == "https://example.com"

    def test_input(self):
        assert self.probe.input == "example.com"

    def test_final_url(self):
        assert self.probe.final_url == "https://www.example.com/"

    def test_scheme(self):
        assert self.probe.scheme == "https"

    def test_host(self):
        assert self.probe.host == "93.184.216.34"

    def test_port(self):
        assert self.probe.port == "443"

    def test_status_code(self):
        assert self.probe.status_code == 200

    def test_content_type(self):
        assert self.probe.content_type == "text/html; charset=UTF-8"

    def test_content_length(self):
        assert self.probe.content_length == 1256

    def test_title(self):
        assert self.probe.title == "Example Domain"

    def test_webserver(self):
        assert self.probe.webserver == "ECS (dcb/7EEF)"

    def test_tech_list(self):
        assert "Varnish" in self.probe.tech
        assert "Amazon CloudFront" in self.probe.tech

    def test_words(self):
        assert self.probe.words == 298

    def test_lines(self):
        assert self.probe.lines == 47

    def test_response_time(self):
        assert self.probe.response_time == "272.895ms"

    def test_not_failed(self):
        assert self.probe.failed is False

    def test_chain_status_codes(self):
        assert self.probe.chain_status_codes == [200]

    def test_response_headers(self):
        assert self.probe.response_headers["server"] == "ECS (dcb/7EEF)"

    def test_a_records(self):
        assert "93.184.216.34" in self.probe.a_records


class TestParseProbeHyphenKeys:
    """httpx older releases emit hyphenated JSON keys; both forms must work."""

    def setup_method(self):
        self.probe = _parse_probe(_PROBE_HYPHEN_KEYS)

    def test_status_code(self):
        assert self.probe.status_code == 200

    def test_content_type(self):
        assert self.probe.content_type == "text/html"

    def test_content_length(self):
        assert self.probe.content_length == 512

    def test_chain_status_codes(self):
        assert self.probe.chain_status_codes == [200]

    def test_response_headers(self):
        assert self.probe.response_headers["server"] == "nginx"

    def test_final_url(self):
        assert self.probe.final_url == "https://legacy.example.com/"


class TestParseProbeFailedHost:
    def setup_method(self):
        self.probe = _parse_probe(_PROBE_FAILED)

    def test_failed_flag(self):
        assert self.probe.failed is True

    def test_status_code_is_none(self):
        assert self.probe.status_code is None

    def test_is_live_false(self):
        assert self.probe.is_live is False


class TestParseProbeRedirect:
    def setup_method(self):
        self.probe = _parse_probe(_PROBE_REDIRECT)

    def test_status_301(self):
        assert self.probe.status_code == 301

    def test_is_redirect(self):
        assert self.probe.is_redirect is True

    def test_chain_has_two_codes(self):
        assert len(self.probe.chain_status_codes) == 2
        assert self.probe.chain_status_codes[0] == 301

    def test_is_live_true(self):
        assert self.probe.is_live is True


class TestParseProbeDefaults:
    def test_empty_dict_does_not_raise(self):
        probe = _parse_probe({})
        assert probe.url == ""
        assert probe.status_code is None
        assert probe.failed is False
        assert probe.tech == []
        assert probe.chain_status_codes == []
        assert probe.response_headers == {}
        assert probe.a_records == []

    def test_non_list_tech_field_coerced_to_empty(self):
        probe = _parse_probe({"tech": "not-a-list"})
        assert probe.tech == []

    def test_non_dict_response_headers_coerced_to_empty(self):
        probe = _parse_probe({"response_headers": "not-a-dict"})
        assert probe.response_headers == {}

    def test_bad_status_code_type_yields_none(self):
        probe = _parse_probe({"status_code": "not-a-number"})
        assert probe.status_code is None


# ---------------------------------------------------------------------------
# _parse_ndjson — unit tests
# ---------------------------------------------------------------------------


class TestParseNdjsonSingle:
    def setup_method(self):
        self.result = _parse_ndjson(SINGLE_NDJSON, "httpx -json -silent -u example.com")

    def test_success(self):
        assert self.result.success

    def test_one_probe(self):
        assert len(self.result.probes) == 1

    def test_command_preserved(self):
        assert self.result.command == "httpx -json -silent -u example.com"

    def test_probe_url(self):
        assert self.result.probes[0].url == "https://example.com"


class TestParseNdjsonMulti:
    def setup_method(self):
        self.result = _parse_ndjson(MULTI_NDJSON, "cmd")

    def test_three_probes(self):
        assert len(self.result.probes) == 3

    def test_live_filters_failed(self):
        assert len(self.result.live) == 2

    def test_failed_probe_present(self):
        failed = [p for p in self.result.probes if p.failed]
        assert len(failed) == 1


class TestParseNdjsonBlankLines:
    def test_blank_lines_skipped(self):
        output = "\n\n" + json.dumps(_PROBE_HTTPS) + "\n\n"
        result = _parse_ndjson(output, "cmd")
        assert len(result.probes) == 1

    def test_whitespace_only_output_returns_error(self):
        result = _parse_ndjson("   \n\n  ", "cmd")
        assert not result.success
        assert result.error is not None


class TestParseNdjsonMalformed:
    def test_entirely_bad_json_returns_error(self):
        result = _parse_ndjson("{not valid json}\n{also bad}", "cmd")
        assert not result.success
        assert "JSON decode error" in (result.error or "")

    def test_one_bad_line_among_good_lines_is_dropped(self):
        output = json.dumps(_PROBE_HTTPS) + "\n{bad json}\n" + json.dumps(_PROBE_REDIRECT) + "\n"
        result = _parse_ndjson(output, "cmd")
        assert result.success
        assert len(result.probes) == 2

    def test_non_object_json_line_dropped(self):
        output = json.dumps(_PROBE_HTTPS) + "\n[1,2,3]\n"
        result = _parse_ndjson(output, "cmd")
        assert result.success
        assert len(result.probes) == 1

    def test_empty_string_returns_error(self):
        result = _parse_ndjson("", "cmd")
        assert not result.success


# ---------------------------------------------------------------------------
# HttpxTool — subprocess integration tests (mocked)
# ---------------------------------------------------------------------------


def _make_proc(stdout: str = "", stderr: str = "", returncode: int = 0) -> MagicMock:
    proc = MagicMock()
    proc.stdout = stdout
    proc.stderr = stderr
    proc.returncode = returncode
    return proc


class TestHttpxToolSingleTarget:
    def test_happy_path(self):
        tool = HttpxTool()
        with patch("tools.httpx.httpx_tool.subprocess.run",
                   return_value=_make_proc(stdout=SINGLE_NDJSON)) as mock_run:
            result = tool.probe("example.com")

        assert result.success
        assert len(result.probes) == 1
        mock_run.assert_called_once()

    def test_single_target_uses_dash_u_flag(self):
        tool = HttpxTool()
        with patch("tools.httpx.httpx_tool.subprocess.run",
                   return_value=_make_proc(stdout=SINGLE_NDJSON)) as mock_run:
            tool.probe("example.com")

        args = mock_run.call_args[0][0]
        assert "-u" in args
        assert "example.com" in args

    def test_base_flags_present(self):
        tool = HttpxTool()
        with patch("tools.httpx.httpx_tool.subprocess.run",
                   return_value=_make_proc(stdout=SINGLE_NDJSON)) as mock_run:
            tool.probe("example.com")

        args = mock_run.call_args[0][0]
        assert "-json" in args
        assert "-silent" in args
        assert "-no-color" in args

    def test_stdin_is_none_for_single_target(self):
        tool = HttpxTool()
        with patch("tools.httpx.httpx_tool.subprocess.run",
                   return_value=_make_proc(stdout=SINGLE_NDJSON)) as mock_run:
            tool.probe("example.com")

        call_kwargs = mock_run.call_args[1]
        assert call_kwargs.get("input") is None

    def test_extra_args_forwarded(self):
        tool = HttpxTool()
        with patch("tools.httpx.httpx_tool.subprocess.run",
                   return_value=_make_proc(stdout=SINGLE_NDJSON)) as mock_run:
            tool.probe("example.com", extra_args=["-follow-redirects", "-timeout", "5"])

        args = mock_run.call_args[0][0]
        assert "-follow-redirects" in args
        assert "-timeout" in args


class TestHttpxToolMultipleTargets:
    def test_multiple_targets_use_stdin(self):
        tool = HttpxTool()
        with patch("tools.httpx.httpx_tool.subprocess.run",
                   return_value=_make_proc(stdout=MULTI_NDJSON)) as mock_run:
            tool.probe(["example.com", "example.org"])

        call_kwargs = mock_run.call_args[1]
        assert call_kwargs.get("input") == "example.com\nexample.org"

    def test_multiple_targets_use_dash_l_dash(self):
        tool = HttpxTool()
        with patch("tools.httpx.httpx_tool.subprocess.run",
                   return_value=_make_proc(stdout=MULTI_NDJSON)) as mock_run:
            tool.probe(["example.com", "example.org"])

        args = mock_run.call_args[0][0]
        assert "-l" in args
        assert "-" in args

    def test_returns_all_probes(self):
        tool = HttpxTool()
        with patch("tools.httpx.httpx_tool.subprocess.run",
                   return_value=_make_proc(stdout=MULTI_NDJSON)):
            result = tool.probe(["example.com", "example.org", "dead.invalid"])

        assert len(result.probes) == 3

    def test_single_element_list_accepted(self):
        tool = HttpxTool()
        with patch("tools.httpx.httpx_tool.subprocess.run",
                   return_value=_make_proc(stdout=SINGLE_NDJSON)):
            result = tool.probe(["example.com"])

        assert result.success


class TestHttpxToolErrors:
    def test_binary_not_found(self):
        tool = HttpxTool(httpx_path="/nonexistent/httpx")
        with patch("tools.httpx.httpx_tool.subprocess.run", side_effect=FileNotFoundError):
            result = tool.probe("example.com")

        assert not result.success
        assert "not found" in (result.error or "").lower()

    def test_timeout(self):
        tool = HttpxTool(timeout=5)
        with patch("tools.httpx.httpx_tool.subprocess.run",
                   side_effect=subprocess.TimeoutExpired(cmd="httpx", timeout=5)):
            result = tool.probe("example.com")

        assert not result.success
        assert "timed out" in (result.error or "").lower()

    def test_nonzero_exit_code(self):
        tool = HttpxTool()
        with patch("tools.httpx.httpx_tool.subprocess.run",
                   return_value=_make_proc(returncode=1, stderr="flag not found: -bad-flag")):
            result = tool.probe("example.com")

        assert not result.success
        assert "flag not found" in (result.error or "")

    def test_nonzero_no_stderr_shows_placeholder(self):
        tool = HttpxTool()
        with patch("tools.httpx.httpx_tool.subprocess.run",
                   return_value=_make_proc(returncode=2, stderr="")):
            result = tool.probe("example.com")

        assert not result.success
        assert "(no stderr)" in (result.error or "")

    def test_empty_stdout_returns_error(self):
        tool = HttpxTool()
        with patch("tools.httpx.httpx_tool.subprocess.run",
                   return_value=_make_proc(stdout="")):
            result = tool.probe("example.com")

        assert not result.success
        assert "no output" in (result.error or "").lower()

    def test_whitespace_only_stdout_returns_error(self):
        tool = HttpxTool()
        with patch("tools.httpx.httpx_tool.subprocess.run",
                   return_value=_make_proc(stdout="   \n  ")):
            result = tool.probe("example.com")

        assert not result.success

    def test_malformed_json_output_returns_error(self):
        tool = HttpxTool()
        with patch("tools.httpx.httpx_tool.subprocess.run",
                   return_value=_make_proc(stdout="{not valid json}")):
            result = tool.probe("example.com")

        assert not result.success
        assert "JSON decode error" in (result.error or "")

    def test_os_error_returns_error(self):
        tool = HttpxTool()
        with patch("tools.httpx.httpx_tool.subprocess.run",
                   side_effect=OSError("Permission denied")):
            result = tool.probe("example.com")

        assert not result.success
        assert "OS error" in (result.error or "")

    def test_command_string_recorded_on_error(self):
        tool = HttpxTool(httpx_path="httpx")
        with patch("tools.httpx.httpx_tool.subprocess.run", side_effect=FileNotFoundError):
            result = tool.probe("example.com")

        assert "example.com" in result.command


# ---------------------------------------------------------------------------
# Model property tests
# ---------------------------------------------------------------------------


class TestHttpxProbeResultIsLive:
    def test_live_when_status_set_and_not_failed(self):
        p = HttpxProbeResult(url="https://example.com", status_code=200, failed=False)
        assert p.is_live is True

    def test_not_live_when_failed(self):
        p = HttpxProbeResult(url="https://example.com", status_code=200, failed=True)
        assert p.is_live is False

    def test_not_live_when_no_status(self):
        p = HttpxProbeResult(url="https://example.com", status_code=None, failed=False)
        assert p.is_live is False


class TestHttpxProbeResultIsRedirect:
    @pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
    def test_redirect_codes(self, code: int):
        p = HttpxProbeResult(url="u", status_code=code)
        assert p.is_redirect is True

    @pytest.mark.parametrize("code", [200, 404, 500])
    def test_non_redirect_codes(self, code: int):
        p = HttpxProbeResult(url="u", status_code=code)
        assert p.is_redirect is False

    def test_none_status_not_redirect(self):
        p = HttpxProbeResult(url="u", status_code=None)
        assert p.is_redirect is False


class TestHttpxScanResultSuccess:
    def test_success_when_no_error(self):
        r = HttpxScanResult(command="cmd")
        assert r.success is True

    def test_not_success_when_error_set(self):
        r = HttpxScanResult(command="cmd", error="something went wrong")
        assert r.success is False


class TestHttpxScanResultLive:
    def test_live_filters_failed_and_null_status(self):
        r = HttpxScanResult(
            command="cmd",
            probes=[
                HttpxProbeResult(url="a", status_code=200, failed=False),
                HttpxProbeResult(url="b", status_code=None, failed=False),
                HttpxProbeResult(url="c", status_code=200, failed=True),
                HttpxProbeResult(url="d", status_code=404, failed=False),
            ],
        )
        live = r.live
        assert len(live) == 2
        assert all(p.is_live for p in live)

    def test_live_empty_when_no_probes(self):
        r = HttpxScanResult(command="cmd")
        assert r.live == []
