"""LangGraph callback handler for Snapshot capture.

Intercepts ``on_tool_start`` / ``on_tool_end`` / ``on_tool_error`` events
and delegates to :class:`~audit_fence.snapshot._AgentWriter` for trace file
output.

Requires ``langchain-core``::

    pip install langchain-core

The handler is instantiated via :meth:`Snapshot.config` or
:meth:`Snapshot.handler` — users rarely need to import this module directly.
"""

from __future__ import annotations

import time
from typing import Any

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:
    raise ImportError(
        "SnapshotHandler requires langchain-core. "
        "Install with: pip install langchain-core"
    )

from uuid import UUID


class SnapshotHandler(BaseCallbackHandler):
    """LangGraph callback handler that captures tool call I/O.

    Designed to be passed via ``config={"callbacks": [handler]}``.
    LangGraph automatically propagates callbacks to all tool invocations
    within the graph, so no tool modification is needed.

    Usage (via Snapshot — recommended)::

        snap = Snapshot("trace/")
        result = await agent.ainvoke(
            input, config=snap.config(agent="research")
        )

    Usage (manual config assembly)::

        handler = SnapshotHandler(writer=writer, agent="research")
        config = {"callbacks": [handler], "recursion_limit": 50}
    """

    def __init__(self, *, writer: Any, agent: str):
        """
        Args:
            writer: An :class:`~audit_fence.snapshot._AgentWriter` instance.
            agent: Agent name (for logging / identification).
        """
        super().__init__()
        self._writer = writer
        self._agent = agent
        self._pending: dict[str, dict] = {}

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool starts executing."""
        tool_name = serialized.get("name") or ""
        if not tool_name:
            id_parts = serialized.get("id", [])
            tool_name = id_parts[-1] if id_parts else "unknown"

        self._pending[str(run_id)] = {
            "tool": tool_name,
            "input": inputs if inputs is not None else input_str,
            "start_time": time.time(),
        }

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool finishes successfully."""
        rid = str(run_id)
        entry = self._pending.pop(rid, None)
        if entry is None:
            return

        # LangChain ToolMessage wraps output — extract the content
        output_data = output
        if hasattr(output, "content"):
            output_data = output.content

        duration_ms = (time.time() - entry["start_time"]) * 1000
        self._writer.record(
            tool_name=entry["tool"],
            input_data=entry["input"],
            output_data=output_data,
            duration_ms=duration_ms,
            status="ok",
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool raises an exception."""
        rid = str(run_id)
        entry = self._pending.pop(rid, None)
        if entry is None:
            return

        duration_ms = (time.time() - entry["start_time"]) * 1000
        self._writer.record(
            tool_name=entry["tool"],
            input_data=entry["input"],
            output_data=None,
            duration_ms=duration_ms,
            status="error",
            error=str(error),
        )
