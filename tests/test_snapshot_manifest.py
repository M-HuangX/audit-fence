"""Tests for manifest generation, format_manifest, and FenceGroup.from_snapshot_manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from audit_fence import FenceGroup, Snapshot
from audit_fence.snapshot import ToolCallRecord
from audit_fence.prompts import format_manifest


# ---------------------------------------------------------------------------
# format_manifest (prompts.py)
# ---------------------------------------------------------------------------

class TestFormatManifest:
    def test_basic(self):
        manifest = {
            "agents": {
                "research": {
                    "tool_calls": 15,
                    "tool_counts": {"search": 10, "record": 5},
                    "artifacts": ["analysis.md"],
                    "trace_dir": "research/",
                },
            },
            "dependencies": {},
        }
        result = format_manifest(manifest)
        assert "research/" in result
        assert "15 tool calls" in result
        assert "search x10" in result
        assert "record x5" in result
        assert "Artifact: research/analysis.md" in result

    def test_pipeline_flow(self):
        manifest = {
            "agents": {
                "research": {"tool_calls": 5, "tool_counts": {}, "artifacts": [], "trace_dir": "research/"},
                "analysis": {"tool_calls": 3, "tool_counts": {}, "artifacts": [], "trace_dir": "analysis/"},
                "writer": {"tool_calls": 2, "tool_counts": {}, "artifacts": [], "trace_dir": "writer/"},
            },
            "dependencies": {
                "analysis": ["research"],
                "writer": ["analysis"],
            },
        }
        result = format_manifest(manifest)
        assert "research -> analysis -> writer" in result

    def test_no_dependencies(self):
        manifest = {
            "agents": {
                "a": {"tool_calls": 1, "tool_counts": {}, "artifacts": [], "trace_dir": "a/"},
                "b": {"tool_calls": 1, "tool_counts": {}, "artifacts": [], "trace_dir": "b/"},
            },
            "dependencies": {},
        }
        result = format_manifest(manifest)
        assert "Pipeline flow:" in result

    def test_empty_manifest(self):
        manifest = {"agents": {}, "dependencies": {}}
        result = format_manifest(manifest)
        assert "Pipeline flow:" in result


# ---------------------------------------------------------------------------
# Snapshot manifest generation (via finalize)
# ---------------------------------------------------------------------------

class TestManifestGeneration:
    def test_manifest_structure(self, tmp_path):
        snap = Snapshot(str(tmp_path / "trace"), session_id="test123")

        w = snap._get_writer("research")
        w.record("search", {"q": "AAPL"}, "found", 50, "ok")
        w.record("search", {"q": "MSFT"}, "found", 40, "ok")
        w.record("record", {"c": "rev"}, "ok", 10, "ok")

        snap.save_artifact("research", "# Report", "report.md")
        snap.declare_dependency("analysis", upstream="research")

        snap.finalize()
        manifest = snap.load_manifest()

        assert manifest["version"] == 1
        assert manifest["session_id"] == "test123"
        assert "created" in manifest
        assert "finalized" in manifest
        assert manifest["duration_seconds"] >= 0

        agent = manifest["agents"]["research"]
        assert agent["tool_calls"] == 3
        assert agent["tool_counts"] == {"search": 2, "record": 1}
        assert agent["artifacts"] == ["report.md"]
        assert agent["errors"] == 0
        assert agent["trace_dir"] == "research/"
        assert agent["trace_bytes"] > 0

        assert manifest["dependencies"] == {"analysis": ["research"]}

        totals = manifest["totals"]
        assert totals["agents"] == 1
        assert totals["tool_calls"] == 3
        assert totals["artifacts"] == 1
        assert totals["errors"] == 0

    def test_manifest_with_errors(self, tmp_path):
        snap = Snapshot(str(tmp_path / "trace"))
        w = snap._get_writer("agent")
        w.record("ok_tool", {}, "r", 10, "ok")
        w.record("bad_tool", {}, None, 10, "error", error="boom")
        snap.finalize()

        manifest = snap.load_manifest()
        assert manifest["agents"]["agent"]["errors"] == 1
        assert manifest["totals"]["errors"] == 1

    def test_manifest_incomplete(self, tmp_path):
        snap = Snapshot(str(tmp_path / "trace"))
        snap.finalize(incomplete=True)
        manifest = snap.load_manifest()
        assert manifest["incomplete"] is True


# ---------------------------------------------------------------------------
# FenceGroup.from_snapshot_manifest
# ---------------------------------------------------------------------------

class TestFromSnapshotManifest:
    def _make_trace(self, tmp_path):
        """Create a trace directory with 2 agents and dependencies."""
        snap = Snapshot(str(tmp_path / "trace"))

        w1 = snap._get_writer("research")
        w1.record("search", {"q": "AAPL"}, "revenue: 394B", 50, "ok")
        snap.save_artifact("research", "# Research\nAAPL revenue is $394B.", "analysis.md")

        w2 = snap._get_writer("writer")
        w2.record("format", {}, "formatted", 10, "ok")

        snap.declare_dependency("writer", upstream="research")
        snap.finalize()
        return snap

    def test_creates_fences_per_agent(self, tmp_path):
        snap = self._make_trace(tmp_path)
        manifest = snap.load_manifest()

        group = FenceGroup.from_snapshot_manifest(
            manifest,
            document="Final report text.",
            trace_dir=snap.trace_dir,
        )

        assert "research" in group
        assert "writer" in group

    def test_document_priority_artifact(self, tmp_path):
        snap = self._make_trace(tmp_path)
        manifest = snap.load_manifest()

        group = FenceGroup.from_snapshot_manifest(
            manifest,
            document="Final report.",
            trace_dir=snap.trace_dir,
        )

        # Research has an artifact → its document should be the artifact content
        research_fence = group["research"]
        assert "AAPL revenue" in research_fence._resolved_document

        # Writer has no artifact → its document should be the fallback
        writer_fence = group["writer"]
        assert writer_fence._resolved_document == "Final report."

    def test_document_priority_override(self, tmp_path):
        snap = self._make_trace(tmp_path)
        manifest = snap.load_manifest()

        group = FenceGroup.from_snapshot_manifest(
            manifest,
            document="Final report.",
            trace_dir=snap.trace_dir,
            per_agent_documents={"research": "Custom doc for research."},
        )

        research_fence = group["research"]
        assert research_fence._resolved_document == "Custom doc for research."

    def test_link_topology(self, tmp_path):
        snap = self._make_trace(tmp_path)
        manifest = snap.load_manifest()

        group = FenceGroup.from_snapshot_manifest(
            manifest,
            document="Report.",
            trace_dir=snap.trace_dir,
        )

        writer_fence = group["writer"]
        assert len(writer_fence._upstream) == 1
        assert writer_fence._upstream[0] is group["research"]

    def test_source_set(self, tmp_path):
        snap = self._make_trace(tmp_path)
        manifest = snap.load_manifest()

        group = FenceGroup.from_snapshot_manifest(
            manifest,
            document="Report.",
            trace_dir=snap.trace_dir,
        )

        # Each fence should have search capability via set_source
        research_fence = group["research"]
        assert research_fence._search_fn is not None


# ---------------------------------------------------------------------------
# End-to-end: Snapshot → manifest → format_manifest → audit prompt
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_full_flow(self, tmp_path):
        """Snapshot capture → finalize → load manifest → format for audit."""
        snap = Snapshot(str(tmp_path / "trace"), session_id="e2e")

        # Simulate 2-stage pipeline
        w1 = snap._get_writer("research")
        w1.record("get_income", {"ticker": "AAPL"}, {"revenue": 394328000000}, 100, "ok")
        w1.record("search_news", {"q": "AAPL earnings"}, "Strong Q4...", 200, "ok")
        snap.save_artifact("research", "Analysis: AAPL revenue is $394B.", "analysis.md")

        w2 = snap._get_writer("writer")
        w2.record("format_report", {}, "Final report.", 50, "ok")

        snap.declare_dependency("writer", upstream="research")
        snap.finalize()

        # Load and format
        manifest = snap.load_manifest()
        prompt_section = format_manifest(manifest)

        assert "research/" in prompt_section
        assert "2 tool calls" in prompt_section
        assert "writer/" in prompt_section
        assert "research -> writer" in prompt_section

        # build_index
        index = snap.build_index()
        assert len(index) == 3
        rec = index["research:0"]
        assert rec.tool == "get_income"

        # resolve_ref
        ref = snap.resolve_ref("research/calls/0000_get_income.txt", 5)
        assert ref is not None
        assert ref.tool == "get_income"
