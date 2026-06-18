"""Unit tests for tools/subfinder/subfinder_tool.py.

All tests are pure-Python: no subfinder binary, no network I/O.
subprocess.run is monkey-patched wherever a real process would be invoked.
"""
from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from tools.subfinder.subfinder_tool import (
    SubfinderDomain,
    SubfinderScanResult,
    SubfinderTool,
    _parse_domain,
    _parse_ndjson,
)

# ---------------------------------------------------------------------------
# NDJSON fixtures
# ---------------------------------------------------------------------------

_DOMAIN_WWW = {
    "host": "www.example.com",
    "input": "example.com",
    "source": "certspotter",
}

_DOMAIN_MAIL = {
    "host": "mail.example.com",
    "input": "example.com",
    "source": "dnsdumpster",
}

_DOMAIN_API = {
    "host": "api.v2.example.com",
    "input": "example.com",
    "source": "crtsh",
}

_DOMAIN_APEX = {
    "host": "example.com",
    "input": "example.com",
    "source": "hackertarget",
}

SINGLE_NDJSON = json.dumps(_DOMAIN_WWW) + "\n"

MULTI_NDJSON = "\n".join([
    json.dumps(_DOMAIN_WWW),
    json.dumps(_DOMAIN_MAIL),
    json.dumps(_DOMAIN_API),
]) + "\n"

APEX_NDJSON = json.dumps(_DOMAIN_APEX) + "\n"


# ---------------------------------------------------------------------------
# _parse_domain — unit tests
# ---------------------------------------------------------------------------


class TestParseSubfinderDomainBasic:
    def setup_method(self):
        self.domain = _parse_domain(_DOMAIN_WWW)

    def test_host(self):
        assert self.domain.host == "www.example.com"

    def test_input(self):
        assert self.domain.input == "example.com"

    def test_source(self):
        assert self.domain.source == "certspotter"


class TestParseSubfinderDomainDefaults:
    def test_empty_dict_does_not_raise(self):
        domain = _parse_domain({})
        assert domain.host == ""
        assert domain.input == ""
        assert domain.source == ""

    def test_missing_input_defaults_to_empty(self):
        domain = _parse_domain({"host": "sub.example.com"})
        assert domain.input == ""

    def test_missing_source_defaults_to_empty(self):
        domain = _parse_domain({"host": "sub.example.com", "input": "example.com"})
        assert domain.source == ""

    def test_non_string_host_coerced(self):
        domain = _parse_domain({"host": 12345, "input": "example.com"})
        assert domain.host == "12345"

    def test_non_string_source_coerced(self):
        domain = _parse_domain({"host": "sub.example.com", "source": 99})
        assert domain.source == "99"


# ---------------------------------------------------------------------------
# _parse_ndjson — unit tests
# ---------------------------------------------------------------------------


class TestParseNdjsonSingle:
    def setup_method(self):
        self.result = _parse_ndjson(SINGLE_NDJSON, "subfinder -silent -json -d example.com")

    def test_success(self):
        assert self.result.success

    def test_error_is_none(self):
        assert self.result.error is None

    def test_one_domain(self):
        assert len(self.result.domains) == 1

    def test_command_preserved(self):
        assert self.result.command == "subfinder -silent -json -d example.com"

    def test_domain_host(self):
        assert self.result.domains[0].host == "www.example.com"


class TestParseNdjsonMultiple:
    def setup_method(self):
        self.result = _parse_ndjson(MULTI_NDJSON, "cmd")

    def test_three_domains(self):
        assert len(self.result.domains) == 3

    def test_count_property(self):
        assert self.result.count == 3

    def test_all_hosts_present(self):
        hosts = {d.host for d in self.result.domains}
        assert "www.example.com" in hosts
        assert "mail.example.com" in hosts
        assert "api.v2.example.com" in hosts

    def test_sources_distinct(self):
        sources = {d.source for d in self.result.domains}
        assert len(sources) == 3


class TestParseNdjsonBlankLines:
    def test_blank_lines_skipped(self):
        output = "\n\n" + json.dumps(_DOMAIN_WWW) + "\n\n"
        result = _parse_ndjson(output, "cmd")
        assert len(result.domains) == 1

    def test_whitespace_only_returns_error(self):
        result = _parse_ndjson("   \n\n  ", "cmd")
        assert not result.success
        assert result.error is not None

    def test_empty_string_returns_error(self):
        result = _parse_ndjson("", "cmd")
        assert not result.success
        assert "no parseable output" in (result.error or "")


class TestParseNdjsonMalformed:
    def test_entirely_bad_json_returns_error(self):
        result = _parse_ndjson("{not valid}\n{also bad}", "cmd")
        assert not result.success
        assert "JSON decode error" in (result.error or "")

    def test_one_bad_line_among_good_lines_is_dropped(self):
        output = (
            json.dumps(_DOMAIN_WWW) + "\n"
            "{bad json}\n"
            + json.dumps(_DOMAIN_MAIL) + "\n"
        )
        result = _parse_ndjson(output, "cmd")
        assert result.success
        assert len(result.domains) == 2

    def test_non_object_json_line_dropped(self):
        output = json.dumps(_DOMAIN_WWW) + "\n[1, 2, 3]\n"
        result = _parse_ndjson(output, "cmd")
        assert result.success
        assert len(result.domains) == 1

    def test_all_non_object_lines_returns_error(self):
        output = "[1,2,3]\n[4,5,6]\n"
        result = _parse_ndjson(output, "cmd")
        assert not result.success

    def test_mixed_valid_and_non_object_keeps_valid(self):
        output = json.dumps(_DOMAIN_API) + "\nnull\n"
        result = _parse_ndjson(output, "cmd")
        assert result.success
        assert len(result.domains) == 1


# ---------------------------------------------------------------------------
# SubfinderTool — subprocess integration tests (mocked)
# ---------------------------------------------------------------------------


def _make_proc(stdout: str = "", stderr: str = "", returncode: int = 0) -> MagicMock:
    proc = MagicMock()
    proc.stdout = stdout
    proc.stderr = stderr
    proc.returncode = returncode
    return proc


class TestSubfinderToolScan:
    def test_happy_path(self):
        tool = SubfinderTool()
        with patch("tools.subfinder.subfinder_tool.subprocess.run",
                   return_value=_make_proc(stdout=SINGLE_NDJSON)) as mock_run:
            result = tool.scan("example.com")

        assert result.success
        assert len(result.domains) == 1
        mock_run.assert_called_once()

    def test_base_flags_present(self):
        tool = SubfinderTool()
        with patch("tools.subfinder.subfinder_tool.subprocess.run",
                   return_value=_make_proc(stdout=SINGLE_NDJSON)) as mock_run:
            tool.scan("example.com")

        args = mock_run.call_args[0][0]
        assert "-silent" in args
        assert "-json" in args

    def test_domain_passed_with_dash_d(self):
        tool = SubfinderTool()
        with patch("tools.subfinder.subfinder_tool.subprocess.run",
                   return_value=_make_proc(stdout=SINGLE_NDJSON)) as mock_run:
            tool.scan("example.com")

        args = mock_run.call_args[0][0]
        assert "-d" in args
        idx = args.index("-d")
        assert args[idx + 1] == "example.com"

    def test_extra_args_forwarded(self):
        tool = SubfinderTool()
        with patch("tools.subfinder.subfinder_tool.subprocess.run",
                   return_value=_make_proc(stdout=SINGLE_NDJSON)) as mock_run:
            tool.scan("example.com", extra_args=["-timeout", "30", "-all"])

        args = mock_run.call_args[0][0]
        assert "-timeout" in args
        assert "30" in args
        assert "-all" in args

    def test_command_string_includes_domain(self):
        tool = SubfinderTool()
        with patch("tools.subfinder.subfinder_tool.subprocess.run",
                   return_value=_make_proc(stdout=SINGLE_NDJSON)):
            result = tool.scan("example.com")

        assert "example.com" in result.command

    def test_command_string_includes_base_flags(self):
        tool = SubfinderTool()
        with patch("tools.subfinder.subfinder_tool.subprocess.run",
                   return_value=_make_proc(stdout=SINGLE_NDJSON)):
            result = tool.scan("example.com")

        assert "-silent" in result.command
        assert "-json" in result.command

    def test_multiple_domains_returned(self):
        tool = SubfinderTool()
        with patch("tools.subfinder.subfinder_tool.subprocess.run",
                   return_value=_make_proc(stdout=MULTI_NDJSON)):
            result = tool.scan("example.com")

        assert result.count == 3

    def test_custom_binary_path_used(self):
        tool = SubfinderTool(subfinder_path="/opt/tools/subfinder")
        with patch("tools.subfinder.subfinder_tool.subprocess.run",
                   return_value=_make_proc(stdout=SINGLE_NDJSON)) as mock_run:
            tool.scan("example.com")

        args = mock_run.call_args[0][0]
        assert args[0] == "/opt/tools/subfinder"


class TestSubfinderToolErrors:
    def test_binary_not_found(self):
        tool = SubfinderTool(subfinder_path="/nonexistent/subfinder")
        with patch("tools.subfinder.subfinder_tool.subprocess.run",
                   side_effect=FileNotFoundError):
            result = tool.scan("example.com")

        assert not result.success
        assert "not found" in (result.error or "").lower()

    def test_timeout(self):
        tool = SubfinderTool(timeout=30)
        with patch("tools.subfinder.subfinder_tool.subprocess.run",
                   side_effect=subprocess.TimeoutExpired(cmd="subfinder", timeout=30)):
            result = tool.scan("example.com")

        assert not result.success
        assert "timed out" in (result.error or "").lower()
        assert "30s" in (result.error or "")

    def test_nonzero_exit_code(self):
        tool = SubfinderTool()
        with patch("tools.subfinder.subfinder_tool.subprocess.run",
                   return_value=_make_proc(returncode=1, stderr="invalid config")):
            result = tool.scan("example.com")

        assert not result.success
        assert "invalid config" in (result.error or "")
        assert "1" in (result.error or "")

    def test_nonzero_no_stderr_shows_placeholder(self):
        tool = SubfinderTool()
        with patch("tools.subfinder.subfinder_tool.subprocess.run",
                   return_value=_make_proc(returncode=2, stderr="")):
            result = tool.scan("example.com")

        assert not result.success
        assert "(no stderr)" in (result.error or "")

    def test_empty_stdout_returns_error(self):
        tool = SubfinderTool()
        with patch("tools.subfinder.subfinder_tool.subprocess.run",
                   return_value=_make_proc(stdout="")):
            result = tool.scan("example.com")

        assert not result.success
        assert "no output" in (result.error or "").lower()

    def test_whitespace_only_stdout_returns_error(self):
        tool = SubfinderTool()
        with patch("tools.subfinder.subfinder_tool.subprocess.run",
                   return_value=_make_proc(stdout="   \n  ")):
            result = tool.scan("example.com")

        assert not result.success

    def test_malformed_json_output_returns_error(self):
        tool = SubfinderTool()
        with patch("tools.subfinder.subfinder_tool.subprocess.run",
                   return_value=_make_proc(stdout="{not valid json}")):
            result = tool.scan("example.com")

        assert not result.success
        assert "JSON decode error" in (result.error or "")

    def test_os_error_returns_error(self):
        tool = SubfinderTool()
        with patch("tools.subfinder.subfinder_tool.subprocess.run",
                   side_effect=OSError("Permission denied")):
            result = tool.scan("example.com")

        assert not result.success
        assert "OS error" in (result.error or "")

    def test_command_recorded_on_all_errors(self):
        tool = SubfinderTool(subfinder_path="subfinder")
        with patch("tools.subfinder.subfinder_tool.subprocess.run",
                   side_effect=FileNotFoundError):
            result = tool.scan("target.org")

        assert "target.org" in result.command
        assert "-d" in result.command


# ---------------------------------------------------------------------------
# SubfinderDomain.root_domain property tests
# ---------------------------------------------------------------------------


class TestSubfinderDomainRootDomain:
    @pytest.mark.parametrize("host,expected", [
        ("www.example.com", "example.com"),
        ("mail.example.com", "example.com"),
        ("api.v2.example.com", "example.com"),
        ("deep.nested.sub.example.com", "example.com"),
        ("example.com", "example.com"),
        ("localhost", "localhost"),
    ])
    def test_root_domain_extraction(self, host: str, expected: str):
        domain = SubfinderDomain(host=host, input="example.com", source="test")
        assert domain.root_domain == expected

    def test_empty_host_returns_empty(self):
        domain = SubfinderDomain(host="", input="example.com", source="test")
        assert domain.root_domain == ""

    def test_single_label_returns_itself(self):
        domain = SubfinderDomain(host="intranet", input="intranet", source="test")
        assert domain.root_domain == "intranet"

    def test_root_domain_is_read_only_property(self):
        domain = SubfinderDomain(host="www.example.com", input="example.com", source="test")
        with pytest.raises(AttributeError):
            domain.root_domain = "other.com"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SubfinderScanResult property tests
# ---------------------------------------------------------------------------


class TestSubfinderScanResultSuccess:
    def test_success_when_no_error(self):
        r = SubfinderScanResult(command="cmd")
        assert r.success is True

    def test_not_success_when_error_set(self):
        r = SubfinderScanResult(command="cmd", error="something went wrong")
        assert r.success is False


class TestSubfinderScanResultCount:
    def test_count_matches_domain_list_length(self):
        r = SubfinderScanResult(
            command="cmd",
            domains=[
                SubfinderDomain(host="a.example.com", input="example.com", source="x"),
                SubfinderDomain(host="b.example.com", input="example.com", source="y"),
                SubfinderDomain(host="c.example.com", input="example.com", source="z"),
            ],
        )
        assert r.count == 3

    def test_count_zero_when_no_domains(self):
        r = SubfinderScanResult(command="cmd")
        assert r.count == 0

    def test_count_zero_on_error_result(self):
        r = SubfinderScanResult(command="cmd", error="failed")
        assert r.count == 0

    def test_domains_default_to_empty_list(self):
        r = SubfinderScanResult(command="cmd")
        assert r.domains == []
