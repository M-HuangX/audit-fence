"""Tests for audit_fence.snapshot — core capture, formatting, redaction."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path

import pytest

from audit_fence.snapshot import (
    Snapshot,
    ToolCallRecord,
    _AgentWriter,
    _apply_builtin_redaction,
    _flatten_dict,
    _format_table,
    _safe_name,
    default_format,
)


# ---------------------------------------------------------------------------
# _safe_name
# ---------------------------------------------------------------------------

class TestSafeName:
    def test_simple_name(self):
        assert _safe_name("get_stock_price") == "get_stock_price"

    def test_namespace_slashes(self):
        assert _safe_name("langchain/tools/tavily") == "langchain_tools_tavily"

    def test_colons(self):
        assert _safe_name("mcp::tool_name") == "mcp__tool_name"

    def test_dots_and_spaces(self):
        assert _safe_name("my.tool name") == "my_tool_name"

    def test_unicode(self):
        result = _safe_name("搜索工具")
        assert all(c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" or c == "_" for c in result)

    def test_truncation(self):
        long_name = "a" * 200
        assert len(_safe_name(long_name)) == 100

    def test_empty_string(self):
        assert _safe_name("") == "unknown"

    def test_all_special_chars(self):
        assert _safe_name("///") == "unknown"

    def test_leading_trailing_underscores(self):
        assert _safe_name("__tool__") == "tool"


# ---------------------------------------------------------------------------
# Builtin redaction
# ---------------------------------------------------------------------------

class TestBuiltinRedaction:
    def test_openai_key(self):
        text = "key is sk-abcdefghijklmnopqrstuvwxyz"
        result = _apply_builtin_redaction(text)
        assert "sk-" not in result
        assert "***API_KEY***" in result

    def test_google_key(self):
        text = "key is AIzaSyA" + "x" * 32
        result = _apply_builtin_redaction(text)
        assert "AIza" not in result
        assert "***API_KEY***" in result

    def test_tavily_key(self):
        text = "key is tvly-abcdefghijklmnopqrstuvwxyz"
        result = _apply_builtin_redaction(text)
        assert "tvly-" not in result

    def test_bearer_token(self):
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abc"
        result = _apply_builtin_redaction(text)
        assert "Bearer ***TOKEN***" in result

    def test_no_false_positives(self):
        text = "revenue: 394328000000, ticker: AAPL"
        assert _apply_builtin_redaction(text) == text


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

class TestFlattenDict:
    def test_simple(self):
        result = _flatten_dict({"price": 150.25, "ticker": "AAPL"})
        assert "price: 150.25" in result
        assert "ticker: AAPL" in result

    def test_nested(self):
        result = _flatten_dict({"a": {"b": {"c": 42}}})
        assert "a.b.c: 42" in result

    def test_max_depth(self):
        result = _flatten_dict({"a": {"b": {"c": {"d": 1}}}}, max_depth=2)
        assert "a.b.c" not in result or "{'d': 1}" in result

    def test_list_of_dicts(self):
        result = _flatten_dict({"items": [{"x": 1}, {"x": 2}]})
        assert "items[0].x: 1" in result
        assert "items[1].x: 2" in result


class TestFormatTable:
    def test_basic_table(self):
        rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        result = _format_table(rows)
        assert "(2 rows)" in result
        assert "a" in result
        assert "b" in result

    def test_empty(self):
        assert _format_table([]) == "(empty)"


class TestDefaultFormat:
    def test_none(self):
        assert default_format("t", {}, None) == "(no output)"

    def test_string(self):
        assert default_format("t", {}, "hello") == "hello"

    def test_dict(self):
        result = default_format("t", {}, {"k": "v"})
        assert "k: v" in result

    def test_list_of_dicts(self):
        result = default_format("t", {}, [{"a": 1}, {"a": 2}])
        assert "(2 rows)" in result

    def test_list_of_strings(self):
        result = default_format("t", {}, ["one", "two"])
        assert "[0] one" in result
        assert "[1] two" in result

    def test_mixed_list_fallback(self):
        result = default_format("t", {}, [1, "two", None])
        # Should fall through to JSON
        parsed = json.loads(result)
        assert parsed == [1, "two", None]

    def test_number_fallback(self):
        result = default_format("t", {}, 42)
        assert "42" in result


# ---------------------------------------------------------------------------
# _AgentWriter
# ---------------------------------------------------------------------------

class TestAgentWriter:
    def test_basic_record(self, tmp_path):
        writer = _AgentWriter(
            tmp_path / "agent", "test_agent",
            formatter=None, redact=None, max_output_bytes=10_000_000,
        )
        writer.record(
            tool_name="get_price",
            input_data={"ticker": "AAPL"},
            output_data={"price": 150.25},
            duration_ms=100.0,
            status="ok",
        )

        # Check flat text file exists
        calls = list((tmp_path / "agent" / "calls").iterdir())
        assert len(calls) == 1
        assert calls[0].name == "0000_get_price.txt"
        content = calls[0].read_text()
        assert "[tool_call #0] get_price" in content
        assert "agent: test_agent" in content
        assert "price: 150.25" in content

        # Check JSONL
        jsonl = (tmp_path / "agent" / "tool_calls.jsonl").read_text()
        entry = json.loads(jsonl.strip())
        assert entry["seq"] == 0
        assert entry["tool"] == "get_price"
        assert entry["status"] == "ok"
        assert entry["line_range"][0] == 1

    def test_error_record(self, tmp_path):
        writer = _AgentWriter(
            tmp_path / "agent", "test_agent",
            formatter=None, redact=None, max_output_bytes=10_000_000,
        )
        writer.record(
            tool_name="bad_tool",
            input_data={"q": "test"},
            output_data=None,
            duration_ms=50.0,
            status="error",
            error="ConnectionError: timeout",
        )

        calls = list((tmp_path / "agent" / "calls").iterdir())
        content = calls[0].read_text()
        assert "status: ERROR" in content
        assert "ConnectionError: timeout" in content

        stats = writer.stats()
        assert stats["errors"] == 1
        assert stats["tool_calls"] == 1

    def test_multiple_records_sequential(self, tmp_path):
        writer = _AgentWriter(
            tmp_path / "agent", "test_agent",
            formatter=None, redact=None, max_output_bytes=10_000_000,
        )
        for i in range(5):
            writer.record(
                tool_name=f"tool_{i}",
                input_data={"i": i},
                output_data=f"result_{i}",
                duration_ms=10.0,
                status="ok",
            )

        stats = writer.stats()
        assert stats["tool_calls"] == 5
        assert len(stats["tools_used"]) == 5

        # Check JSONL has 5 lines
        jsonl = (tmp_path / "agent" / "tool_calls.jsonl").read_text().strip()
        assert len(jsonl.split("\n")) == 5

    def test_redaction_disabled(self, tmp_path):
        writer = _AgentWriter(
            tmp_path / "agent", "test_agent",
            formatter=None, redact=False, max_output_bytes=10_000_000,
        )
        api_key = "sk-" + "a" * 30
        writer.record(
            tool_name="test",
            input_data={"key": api_key},
            output_data="ok",
            duration_ms=10.0,
            status="ok",
        )

        content = list((tmp_path / "agent" / "calls").iterdir())[0].read_text()
        assert api_key in content

    def test_redaction_builtin(self, tmp_path):
        writer = _AgentWriter(
            tmp_path / "agent", "test_agent",
            formatter=None, redact=None, max_output_bytes=10_000_000,
        )
        api_key = "sk-" + "a" * 30
        writer.record(
            tool_name="test",
            input_data={"key": api_key},
            output_data=f"response with {api_key}",
            duration_ms=10.0,
            status="ok",
        )

        content = list((tmp_path / "agent" / "calls").iterdir())[0].read_text()
        assert api_key not in content
        assert "***API_KEY***" in content

    def test_custom_redaction(self, tmp_path):
        def my_redact(tool_name, inp, out):
            if isinstance(inp, dict) and "password" in inp:
                inp = {**inp, "password": "***"}
            return inp, out

        writer = _AgentWriter(
            tmp_path / "agent", "test_agent",
            formatter=None, redact=my_redact, max_output_bytes=10_000_000,
        )
        writer.record(
            tool_name="login",
            input_data={"user": "admin", "password": "secret123"},
            output_data="ok",
            duration_ms=10.0,
            status="ok",
        )

        content = list((tmp_path / "agent" / "calls").iterdir())[0].read_text()
        assert "secret123" not in content
        assert "***" in content

    def test_custom_formatter(self, tmp_path):
        def my_formatter(tool_name, input_data, output_data):
            return f"CUSTOM: {output_data}"

        writer = _AgentWriter(
            tmp_path / "agent", "test_agent",
            formatter=my_formatter, redact=None, max_output_bytes=10_000_000,
        )
        writer.record(
            tool_name="test",
            input_data={},
            output_data="data",
            duration_ms=10.0,
            status="ok",
        )

        content = list((tmp_path / "agent" / "calls").iterdir())[0].read_text()
        assert "CUSTOM: data" in content

    def test_formatter_returns_none_falls_through(self, tmp_path):
        def selective_formatter(tool_name, input_data, output_data):
            return None  # Fall through to default

        writer = _AgentWriter(
            tmp_path / "agent", "test_agent",
            formatter=selective_formatter, redact=None, max_output_bytes=10_000_000,
        )
        writer.record(
            tool_name="test",
            input_data={},
            output_data={"k": "v"},
            duration_ms=10.0,
            status="ok",
        )

        content = list((tmp_path / "agent" / "calls").iterdir())[0].read_text()
        assert "k: v" in content

    def test_truncation(self, tmp_path):
        writer = _AgentWriter(
            tmp_path / "agent", "test_agent",
            formatter=None, redact=False, max_output_bytes=500,
        )
        writer.record(
            tool_name="test",
            input_data={},
            output_data="x" * 2000,
            duration_ms=10.0,
            status="ok",
        )

        content = list((tmp_path / "agent" / "calls").iterdir())[0].read_text()
        assert "[TRUNCATED:" in content

    def test_binary_output(self, tmp_path):
        writer = _AgentWriter(
            tmp_path / "agent", "test_agent",
            formatter=None, redact=None, max_output_bytes=10_000_000,
        )
        writer.record(
            tool_name="download",
            input_data={"url": "test.pdf"},
            output_data=b"\x89PNG\r\n" + b"\x00" * 100,
            duration_ms=50.0,
            status="ok",
        )

        calls_dir = tmp_path / "agent" / "calls"
        txt_files = [f for f in calls_dir.iterdir() if f.suffix == ".txt"]
        bin_files = [f for f in calls_dir.iterdir() if f.suffix == ".bin"]
        assert len(txt_files) == 1
        assert len(bin_files) == 1
        assert "[binary output:" in txt_files[0].read_text()

    def test_thread_safety(self, tmp_path):
        writer = _AgentWriter(
            tmp_path / "agent", "test_agent",
            formatter=None, redact=False, max_output_bytes=10_000_000,
        )
        errors = []

        def write_calls(start_idx):
            try:
                for i in range(20):
                    writer.record(
                        tool_name=f"tool_{start_idx}_{i}",
                        input_data={"i": start_idx * 20 + i},
                        output_data=f"result_{start_idx}_{i}",
                        duration_ms=1.0,
                        status="ok",
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_calls, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        stats = writer.stats()
        assert stats["tool_calls"] == 100

        # Check JSONL has 100 lines
        jsonl = (tmp_path / "agent" / "tool_calls.jsonl").read_text().strip()
        lines = jsonl.split("\n")
        assert len(lines) == 100

        # Check all seq numbers are unique
        seqs = [json.loads(line)["seq"] for line in lines]
        assert len(set(seqs)) == 100

    def test_stats(self, tmp_path):
        writer = _AgentWriter(
            tmp_path / "agent", "test_agent",
            formatter=None, redact=None, max_output_bytes=10_000_000,
        )
        writer.record("search", {"q": "a"}, "r1", 10, "ok")
        writer.record("search", {"q": "b"}, "r2", 10, "ok")
        writer.record("record", {"c": "x"}, "r3", 10, "ok")
        writer.record("fail", {}, None, 10, "error", error="boom")
        writer.add_artifact("report.md")

        stats = writer.stats()
        assert stats["tool_calls"] == 4
        assert stats["tool_counts"] == {"search": 2, "record": 1, "fail": 1}
        assert stats["errors"] == 1
        assert stats["artifacts"] == ["report.md"]
        assert stats["trace_dir"] == "test_agent/"


# ---------------------------------------------------------------------------
# Snapshot — lifecycle & query methods
# ---------------------------------------------------------------------------

class TestSnapshot:
    def test_basic_lifecycle(self, tmp_path):
        trace_dir = tmp_path / "trace"
        snap = Snapshot(str(trace_dir), session_id="test_session")

        writer = snap._get_writer("research")
        writer.record("search", {"q": "AAPL"}, "found", 50, "ok")
        writer.record("record", {"c": "rev"}, "ok", 10, "ok")

        manifest_path = snap.finalize()
        assert manifest_path.exists()

        manifest = snap.load_manifest()
        assert manifest["session_id"] == "test_session"
        assert manifest["agents"]["research"]["tool_calls"] == 2
        assert manifest["totals"]["tool_calls"] == 2
        assert manifest["totals"]["agents"] == 1

    def test_multi_agent(self, tmp_path):
        trace_dir = tmp_path / "trace"
        snap = Snapshot(str(trace_dir))

        snap._get_writer("research").record("search", {}, "r1", 10, "ok")
        snap._get_writer("analysis").record("calc", {}, "r2", 10, "ok")
        snap._get_writer("writer").record("format", {}, "r3", 10, "ok")

        snap.declare_dependency("analysis", upstream="research")
        snap.declare_dependency("writer", upstream="analysis")

        snap.finalize()
        manifest = snap.load_manifest()

        assert len(manifest["agents"]) == 3
        assert manifest["dependencies"] == {
            "analysis": ["research"],
            "writer": ["analysis"],
        }
        assert manifest["totals"]["tool_calls"] == 3

    def test_save_artifact(self, tmp_path):
        snap = Snapshot(str(tmp_path / "trace"))
        path = snap.save_artifact("research", "# Analysis\nRevenue is up.", "analysis.md")

        assert path.exists()
        assert path.read_text() == "# Analysis\nRevenue is up."
        assert snap._get_writer("research")._artifacts == ["analysis.md"]

    def test_declare_dependency_list(self, tmp_path):
        snap = Snapshot(str(tmp_path / "trace"))
        snap.declare_dependency("merger", upstream=["research", "analysis"])
        assert snap._dependencies == {"merger": ["research", "analysis"]}

    def test_declare_dependency_idempotent(self, tmp_path):
        snap = Snapshot(str(tmp_path / "trace"))
        snap.declare_dependency("b", upstream="a")
        snap.declare_dependency("b", upstream="a")
        assert snap._dependencies == {"b": ["a"]}

    def test_trace_dir_property(self, tmp_path):
        trace_dir = tmp_path / "trace"
        snap = Snapshot(str(trace_dir))
        assert snap.trace_dir == str(trace_dir)

    def test_agent_dir(self, tmp_path):
        snap = Snapshot(str(tmp_path / "trace"))
        assert snap.agent_dir("research") == str(tmp_path / "trace" / "research")

    def test_finalize_idempotent(self, tmp_path):
        snap = Snapshot(str(tmp_path / "trace"))
        p1 = snap.finalize()
        p2 = snap.finalize()
        assert p1 == p2

    def test_session_id_auto_generated(self, tmp_path):
        snap = Snapshot(str(tmp_path / "trace"))
        assert snap._session_id
        assert len(snap._session_id) > 10

    def test_context_manager_normal(self, tmp_path):
        with Snapshot(str(tmp_path / "trace"), session_id="ctx") as snap:
            snap._get_writer("a").record("t", {}, "ok", 10, "ok")

        manifest = json.loads((tmp_path / "trace" / "manifest.json").read_text())
        assert manifest["session_id"] == "ctx"
        assert "incomplete" not in manifest

    def test_context_manager_on_exception(self, tmp_path):
        with pytest.raises(ValueError):
            with Snapshot(str(tmp_path / "trace"), session_id="err") as snap:
                snap._get_writer("a").record("t", {}, "ok", 10, "ok")
                raise ValueError("boom")

        manifest = json.loads((tmp_path / "trace" / "manifest.json").read_text())
        assert manifest["incomplete"] is True

    def test_repr(self, tmp_path):
        snap = Snapshot(str(tmp_path / "trace"), session_id="abc")
        snap._get_writer("research")
        r = repr(snap)
        assert "Snapshot(" in r
        assert "abc" in r
        assert "research" in r


# ---------------------------------------------------------------------------
# Snapshot — build_index & resolve_ref
# ---------------------------------------------------------------------------

class TestSnapshotIndex:
    def _make_snapshot(self, tmp_path):
        snap = Snapshot(str(tmp_path / "trace"))
        w = snap._get_writer("research")
        w.record("search", {"q": "AAPL"}, "revenue: 394B", 50, "ok")
        w.record("search", {"q": "MSFT"}, "revenue: 245B", 40, "ok")
        w.record("record", {"c": "rev"}, "logged", 10, "ok")
        snap.finalize()
        return snap

    def test_build_index(self, tmp_path):
        snap = self._make_snapshot(tmp_path)
        index = snap.build_index()
        assert len(index) == 3
        assert "research:0" in index
        assert "research:1" in index
        assert "research:2" in index
        rec = index["research:0"]
        assert isinstance(rec, ToolCallRecord)
        assert rec.tool == "search"
        assert rec.agent == "research"

    def test_resolve_ref_relative(self, tmp_path):
        snap = self._make_snapshot(tmp_path)
        rec = snap.resolve_ref("research/calls/0000_search.txt", 5)
        assert rec is not None
        assert rec.tool == "search"
        assert rec.seq == 0

    def test_resolve_ref_absolute(self, tmp_path):
        snap = self._make_snapshot(tmp_path)
        abs_path = str(tmp_path / "trace" / "research" / "calls" / "0001_search.txt")
        rec = snap.resolve_ref(abs_path, 3)
        assert rec is not None
        assert rec.seq == 1

    def test_resolve_ref_not_found(self, tmp_path):
        snap = self._make_snapshot(tmp_path)
        assert snap.resolve_ref("nonexistent/calls/0000_x.txt", 1) is None

    def test_resolve_ref_wrong_line(self, tmp_path):
        snap = self._make_snapshot(tmp_path)
        rec = snap.resolve_ref("research/calls/0000_search.txt", 99999)
        assert rec is None

    def test_resolve_ref_non_call_path(self, tmp_path):
        snap = self._make_snapshot(tmp_path)
        assert snap.resolve_ref("research/analysis.md", 1) is None

    def test_build_index_skips_malformed_jsonl(self, tmp_path):
        snap = self._make_snapshot(tmp_path)
        # Append a bad line
        jsonl = tmp_path / "trace" / "research" / "tool_calls.jsonl"
        with open(jsonl, "a") as f:
            f.write("THIS IS NOT JSON\n")

        index = snap.build_index()
        assert len(index) == 3  # Still got the 3 valid entries


# ---------------------------------------------------------------------------
# Snapshot — decorator & wrap
# ---------------------------------------------------------------------------

class TestSnapshotCapture:
    def test_sync_decorator(self, tmp_path):
        snap = Snapshot(str(tmp_path / "trace"))

        @snap.capture(agent="research")
        def add(a: int, b: int) -> int:
            return a + b

        result = add(a=3, b=4)
        assert result == 7

        stats = snap._get_writer("research").stats()
        assert stats["tool_calls"] == 1
        assert stats["tools_used"] == ["add"]

    def test_async_decorator(self, tmp_path):
        snap = Snapshot(str(tmp_path / "trace"))

        @snap.capture(agent="research")
        async def fetch(url: str) -> str:
            return f"response from {url}"

        result = asyncio.run(fetch(url="http://test"))
        assert result == "response from http://test"

        stats = snap._get_writer("research").stats()
        assert stats["tool_calls"] == 1

    def test_decorator_captures_error(self, tmp_path):
        snap = Snapshot(str(tmp_path / "trace"))

        @snap.capture(agent="research")
        def failing():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            failing()

        stats = snap._get_writer("research").stats()
        assert stats["tool_calls"] == 1
        assert stats["errors"] == 1

    def test_wrap(self, tmp_path):
        snap = Snapshot(str(tmp_path / "trace"))

        def tool_a(x):
            return x * 2

        def tool_b(x):
            return x + 1

        wrapped = snap.wrap([tool_a, tool_b], agent="research")
        assert len(wrapped) == 2
        assert wrapped[0](5) == 10
        assert wrapped[1](5) == 6

        stats = snap._get_writer("research").stats()
        assert stats["tool_calls"] == 2


# ---------------------------------------------------------------------------
# ToolCallRecord
# ---------------------------------------------------------------------------

class TestToolCallRecord:
    def test_from_dict(self):
        data = {
            "seq": 0,
            "tool": "search",
            "agent": "research",
            "input": {"q": "test"},
            "output_file": "calls/0000_search.txt",
            "output_bytes": 1234,
            "timestamp": 1717000000.0,
            "duration_ms": 542.1,
            "status": "ok",
            "line_range": [1, 18],
        }
        rec = ToolCallRecord(**data)
        assert rec.seq == 0
        assert rec.tool == "search"
        assert rec.line_range == [1, 18]
        assert rec.error is None

    def test_with_error(self):
        rec = ToolCallRecord(
            seq=1, tool="bad", agent="a",
            input={}, output_file="x", output_bytes=0,
            timestamp=0, duration_ms=0, status="error",
            error="timeout",
        )
        assert rec.error == "timeout"
