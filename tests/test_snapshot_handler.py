"""Tests for audit_fence.snapshot_handler — LangGraph callback handler."""

from __future__ import annotations

import json
import time
from pathlib import Path
from uuid import uuid4

import pytest

langchain_core = pytest.importorskip("langchain_core")

from audit_fence.snapshot import Snapshot, _AgentWriter
from audit_fence.snapshot_handler import SnapshotHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_writer(tmp_path: Path, agent: str = "test") -> _AgentWriter:
    return _AgentWriter(
        tmp_path / agent, agent,
        formatter=None, redact=None, max_output_bytes=10_000_000,
    )


def _serialized(name: str) -> dict:
    """Minimal serialized dict as LangGraph passes to on_tool_start."""
    return {"name": name, "id": ["langchain", "tools", name]}


# ---------------------------------------------------------------------------
# on_tool_start + on_tool_end
# ---------------------------------------------------------------------------

class TestToolStartEnd:
    def test_basic_capture(self, tmp_path):
        writer = _make_writer(tmp_path)
        handler = SnapshotHandler(writer=writer, agent="test")
        rid = uuid4()

        handler.on_tool_start(
            serialized=_serialized("search"),
            input_str="query text",
            run_id=rid,
            inputs={"query": "AAPL revenue"},
        )
        handler.on_tool_end(
            output="revenue: $394B",
            run_id=rid,
        )

        stats = writer.stats()
        assert stats["tool_calls"] == 1
        assert stats["tools_used"] == ["search"]

        # Verify flat text content
        calls = list((tmp_path / "test" / "calls").iterdir())
        assert len(calls) == 1
        content = calls[0].read_text()
        assert "[tool_call #0] search" in content
        assert "revenue: $394B" in content

    def test_inputs_preferred_over_input_str(self, tmp_path):
        writer = _make_writer(tmp_path)
        handler = SnapshotHandler(writer=writer, agent="test")
        rid = uuid4()

        handler.on_tool_start(
            serialized=_serialized("search"),
            input_str="fallback text",
            run_id=rid,
            inputs={"query": "preferred input"},
        )
        handler.on_tool_end(output="result", run_id=rid)

        jsonl = (tmp_path / "test" / "tool_calls.jsonl").read_text().strip()
        entry = json.loads(jsonl)
        assert entry["input"] == {"query": "preferred input"}

    def test_input_str_fallback(self, tmp_path):
        writer = _make_writer(tmp_path)
        handler = SnapshotHandler(writer=writer, agent="test")
        rid = uuid4()

        handler.on_tool_start(
            serialized=_serialized("search"),
            input_str="fallback text",
            run_id=rid,
            inputs=None,
        )
        handler.on_tool_end(output="result", run_id=rid)

        jsonl = (tmp_path / "test" / "tool_calls.jsonl").read_text().strip()
        entry = json.loads(jsonl)
        assert entry["input"] == "fallback text"

    def test_tool_name_from_serialized_name(self, tmp_path):
        writer = _make_writer(tmp_path)
        handler = SnapshotHandler(writer=writer, agent="test")
        rid = uuid4()

        handler.on_tool_start(
            serialized={"name": "my_tool", "id": ["other"]},
            input_str="",
            run_id=rid,
        )
        handler.on_tool_end(output="ok", run_id=rid)

        jsonl = (tmp_path / "test" / "tool_calls.jsonl").read_text().strip()
        assert json.loads(jsonl)["tool"] == "my_tool"

    def test_tool_name_fallback_to_id(self, tmp_path):
        writer = _make_writer(tmp_path)
        handler = SnapshotHandler(writer=writer, agent="test")
        rid = uuid4()

        handler.on_tool_start(
            serialized={"name": "", "id": ["langchain", "tools", "fallback_tool"]},
            input_str="",
            run_id=rid,
        )
        handler.on_tool_end(output="ok", run_id=rid)

        jsonl = (tmp_path / "test" / "tool_calls.jsonl").read_text().strip()
        assert json.loads(jsonl)["tool"] == "fallback_tool"

    def test_tool_name_unknown_fallback(self, tmp_path):
        writer = _make_writer(tmp_path)
        handler = SnapshotHandler(writer=writer, agent="test")
        rid = uuid4()

        handler.on_tool_start(
            serialized={"id": []},
            input_str="",
            run_id=rid,
        )
        handler.on_tool_end(output="ok", run_id=rid)

        jsonl = (tmp_path / "test" / "tool_calls.jsonl").read_text().strip()
        assert json.loads(jsonl)["tool"] == "unknown"

    def test_output_with_content_attribute(self, tmp_path):
        """ToolMessage-like objects should have .content extracted."""
        writer = _make_writer(tmp_path)
        handler = SnapshotHandler(writer=writer, agent="test")
        rid = uuid4()

        handler.on_tool_start(
            serialized=_serialized("search"),
            input_str="",
            run_id=rid,
        )

        class FakeToolMessage:
            content = "extracted content"

        handler.on_tool_end(output=FakeToolMessage(), run_id=rid)

        calls = list((tmp_path / "test" / "calls").iterdir())
        content = calls[0].read_text()
        assert "extracted content" in content

    def test_duration_tracking(self, tmp_path):
        writer = _make_writer(tmp_path)
        handler = SnapshotHandler(writer=writer, agent="test")
        rid = uuid4()

        handler.on_tool_start(
            serialized=_serialized("slow"),
            input_str="",
            run_id=rid,
        )
        time.sleep(0.05)
        handler.on_tool_end(output="done", run_id=rid)

        jsonl = (tmp_path / "test" / "tool_calls.jsonl").read_text().strip()
        entry = json.loads(jsonl)
        assert entry["duration_ms"] >= 40  # At least 40ms


# ---------------------------------------------------------------------------
# on_tool_error
# ---------------------------------------------------------------------------

class TestToolError:
    def test_error_capture(self, tmp_path):
        writer = _make_writer(tmp_path)
        handler = SnapshotHandler(writer=writer, agent="test")
        rid = uuid4()

        handler.on_tool_start(
            serialized=_serialized("bad_api"),
            input_str="",
            run_id=rid,
        )
        handler.on_tool_error(
            error=RuntimeError("connection refused"),
            run_id=rid,
        )

        stats = writer.stats()
        assert stats["tool_calls"] == 1
        assert stats["errors"] == 1

        calls = list((tmp_path / "test" / "calls").iterdir())
        content = calls[0].read_text()
        assert "status: ERROR" in content
        assert "connection refused" in content

    def test_error_jsonl_entry(self, tmp_path):
        writer = _make_writer(tmp_path)
        handler = SnapshotHandler(writer=writer, agent="test")
        rid = uuid4()

        handler.on_tool_start(
            serialized=_serialized("fail"),
            input_str="",
            run_id=rid,
        )
        handler.on_tool_error(
            error=ValueError("bad input"),
            run_id=rid,
        )

        jsonl = (tmp_path / "test" / "tool_calls.jsonl").read_text().strip()
        entry = json.loads(jsonl)
        assert entry["status"] == "error"
        assert entry["error"] == "bad input"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_orphaned_end_ignored(self, tmp_path):
        """on_tool_end without matching on_tool_start is silently ignored."""
        writer = _make_writer(tmp_path)
        handler = SnapshotHandler(writer=writer, agent="test")

        handler.on_tool_end(output="orphan", run_id=uuid4())

        assert writer.stats()["tool_calls"] == 0

    def test_orphaned_error_ignored(self, tmp_path):
        """on_tool_error without matching on_tool_start is silently ignored."""
        writer = _make_writer(tmp_path)
        handler = SnapshotHandler(writer=writer, agent="test")

        handler.on_tool_error(error=RuntimeError("orphan"), run_id=uuid4())

        assert writer.stats()["tool_calls"] == 0

    def test_pending_cleanup(self, tmp_path):
        """Completed calls are removed from _pending."""
        writer = _make_writer(tmp_path)
        handler = SnapshotHandler(writer=writer, agent="test")
        rid = uuid4()

        handler.on_tool_start(
            serialized=_serialized("search"),
            input_str="",
            run_id=rid,
        )
        assert len(handler._pending) == 1

        handler.on_tool_end(output="ok", run_id=rid)
        assert len(handler._pending) == 0

    def test_multiple_concurrent_calls(self, tmp_path):
        """Multiple tool calls in flight simultaneously."""
        writer = _make_writer(tmp_path)
        handler = SnapshotHandler(writer=writer, agent="test")

        rid1, rid2, rid3 = uuid4(), uuid4(), uuid4()

        handler.on_tool_start(serialized=_serialized("a"), input_str="", run_id=rid1)
        handler.on_tool_start(serialized=_serialized("b"), input_str="", run_id=rid2)
        handler.on_tool_start(serialized=_serialized("c"), input_str="", run_id=rid3)

        assert len(handler._pending) == 3

        handler.on_tool_end(output="b_result", run_id=rid2)
        handler.on_tool_error(error=RuntimeError("c_fail"), run_id=rid3)
        handler.on_tool_end(output="a_result", run_id=rid1)

        assert len(handler._pending) == 0
        assert writer.stats()["tool_calls"] == 3
        assert writer.stats()["errors"] == 1


# ---------------------------------------------------------------------------
# Integration: SnapshotHandler via Snapshot.config() / Snapshot.handler()
# ---------------------------------------------------------------------------

class TestSnapshotIntegration:
    def test_config_returns_handler(self, tmp_path):
        snap = Snapshot(str(tmp_path / "trace"))
        config = snap.config(agent="research")

        assert "callbacks" in config
        assert len(config["callbacks"]) == 1
        assert isinstance(config["callbacks"][0], SnapshotHandler)

    def test_handler_returns_handler(self, tmp_path):
        snap = Snapshot(str(tmp_path / "trace"))
        h = snap.handler(agent="research")

        assert isinstance(h, SnapshotHandler)

    def test_config_creates_writer(self, tmp_path):
        snap = Snapshot(str(tmp_path / "trace"))
        snap.config(agent="research")

        assert "research" in snap._writers

    def test_full_flow_via_handler(self, tmp_path):
        """Simulate a complete capture flow using handler directly."""
        snap = Snapshot(str(tmp_path / "trace"), session_id="handler_test")
        handler = snap.handler(agent="research")

        # Simulate 2 tool calls
        rid1, rid2 = uuid4(), uuid4()

        handler.on_tool_start(
            serialized=_serialized("get_income"),
            input_str="",
            run_id=rid1,
            inputs={"ticker": "AAPL"},
        )
        handler.on_tool_end(
            output={"revenue": 394328000000, "net_income": 96995000000},
            run_id=rid1,
        )

        handler.on_tool_start(
            serialized=_serialized("search_news"),
            input_str="",
            run_id=rid2,
            inputs={"query": "AAPL Q4"},
        )
        handler.on_tool_end(output="Strong Q4 results...", run_id=rid2)

        snap.finalize()

        manifest = snap.load_manifest()
        assert manifest["agents"]["research"]["tool_calls"] == 2
        assert "get_income" in manifest["agents"]["research"]["tools_used"]
        assert "search_news" in manifest["agents"]["research"]["tools_used"]

        # Verify trace files exist
        calls_dir = tmp_path / "trace" / "research" / "calls"
        assert len(list(calls_dir.iterdir())) == 2
