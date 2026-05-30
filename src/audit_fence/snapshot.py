"""Production-side tool call capture for post-hoc audit.

Captures every tool call made by LangGraph agents and saves grep-friendly
trace files for audit-fence verification.

Quick start (LangGraph — zero-touch)::

    from audit_fence import Fence, Snapshot

    # 1. Capture production tool calls
    snap = Snapshot("trace/")
    result = await agent.ainvoke(input, config=snap.config(agent="research"))
    snap.finalize()

    # 2. Audit against captured data
    fence = Fence()
    fence.set_document(final_report)
    fence.set_source(snap.trace_dir)
    audit = await fence.audit(llm=llm, manifest=snap.load_manifest())

Alternative integration paths (non-LangGraph)::

    # Decorator
    @snap.capture(agent="research")
    def my_tool(query: str) -> dict: ...

    # Wrap existing tools
    wrapped = snap.wrap(tools, agent="research")
"""

from __future__ import annotations

import inspect
import itertools
import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# ToolCallRecord — structured representation of a captured tool call
# ---------------------------------------------------------------------------

@dataclass
class ToolCallRecord:
    """A recorded tool call from JSONL trace data.

    Fields match the JSONL schema written by :class:`_AgentWriter`.
    Returned by :meth:`Snapshot.build_index` and :meth:`Snapshot.resolve_ref`.
    """

    seq: int
    tool: str
    agent: str
    input: Any
    output_file: str
    output_bytes: int
    timestamp: float
    duration_ms: float
    status: str
    line_range: list[int] = field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------------------
# Filename sanitisation
# ---------------------------------------------------------------------------

def _safe_name(name: str) -> str:
    """Sanitise a tool name for use as a filename component.

    Replaces any character that is not ``[a-zA-Z0-9_-]`` with ``_``,
    strips leading/trailing underscores, and truncates to 100 chars.
    """
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    cleaned = cleaned.strip("_")
    return cleaned[:100] or "unknown"


# ---------------------------------------------------------------------------
# Built-in redaction
# ---------------------------------------------------------------------------

_BUILTIN_REDACT_PATTERNS = [
    (re.compile(r"sk-[a-zA-Z0-9]{20,}"), "***API_KEY***"),
    (re.compile(r"AIza[a-zA-Z0-9_-]{35}"), "***API_KEY***"),
    (re.compile(r"tvly-[a-zA-Z0-9]{20,}"), "***API_KEY***"),
    (re.compile(r"Bearer [a-zA-Z0-9._-]{20,}"), "Bearer ***TOKEN***"),
]


def _apply_builtin_redaction(text: str) -> str:
    """Apply built-in API key redaction patterns to *text*."""
    for pattern, replacement in _BUILTIN_REDACT_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _redact_value(value: Any) -> Any:
    """Recursively apply built-in redaction to a value."""
    if isinstance(value, str):
        return _apply_builtin_redaction(value)
    if isinstance(value, dict):
        return {k: _redact_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# Output formatting helpers
# ---------------------------------------------------------------------------

def _flatten_dict(d: dict, max_depth: int = 3, prefix: str = "") -> str:
    """Flatten a dict to ``key: value`` lines (recursive)."""
    lines: list[str] = []
    for key, value in d.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict) and max_depth > 1:
            lines.append(_flatten_dict(value, max_depth - 1, full_key))
        elif (
            isinstance(value, list)
            and value
            and isinstance(value[0], dict)
            and max_depth > 1
        ):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    lines.append(
                        _flatten_dict(item, max_depth - 1, f"{full_key}[{i}]")
                    )
                else:
                    lines.append(f"{full_key}[{i}]: {item}")
        else:
            lines.append(f"{full_key}: {value}")
    return "\n".join(lines)


def _format_table(rows: list[dict]) -> str:
    """Format a list of dicts as an ASCII table."""
    if not rows:
        return "(empty)"
    cols = list(rows[0].keys())
    widths = {c: len(str(c)) for c in cols}
    for row in rows:
        for c in cols:
            widths[c] = max(widths[c], len(str(row.get(c, ""))))

    header = " | ".join(str(c).ljust(widths[c]) for c in cols)
    sep = "-|-".join("-" * widths[c] for c in cols)
    body = "\n".join(
        " | ".join(str(row.get(c, "")).ljust(widths[c]) for c in cols)
        for row in rows
    )
    return f"({len(rows)} rows)\n{header}\n{sep}\n{body}"


def default_format(tool_name: str, input_data: Any, output_data: Any) -> str:
    """Smart flattening for grep-friendly text output.

    Handles the common shapes returned by LLM agent tools:

    - ``str`` → preserved as-is
    - ``dict`` → recursive key-value flattening
    - ``list[dict]`` → ASCII table (SQL results, RAG chunks)
    - ``list[str]`` → numbered lines
    - everything else → ``json.dumps``
    """
    if output_data is None:
        return "(no output)"

    if isinstance(output_data, str):
        return output_data

    if isinstance(output_data, dict):
        return _flatten_dict(output_data, max_depth=3)

    if isinstance(output_data, list) and output_data:
        if all(isinstance(item, dict) for item in output_data):
            return _format_table(output_data)
        if all(isinstance(item, str) for item in output_data):
            return "\n".join(f"[{i}] {item}" for i, item in enumerate(output_data))

    return json.dumps(output_data, indent=2, default=str)


def _format_input(input_data: Any) -> str:
    """Format tool input for the flat text header line."""
    if isinstance(input_data, dict):
        return ", ".join(f"{k}={v}" for k, v in input_data.items())
    if isinstance(input_data, str):
        return input_data
    return str(input_data)


def _serialize(data: Any) -> Any:
    """Make *data* JSON-serialisable (recursive)."""
    if isinstance(data, dict):
        return {k: _serialize(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_serialize(item) for item in data]
    if isinstance(data, (str, int, float, bool, type(None))):
        return data
    return str(data)


# ---------------------------------------------------------------------------
# _AgentWriter — per-agent, thread-safe trace writer
# ---------------------------------------------------------------------------

class _AgentWriter:
    """Thread-safe writer for a single agent's trace files.

    Writes two parallel representations of each tool call:

    1. **JSONL** (``tool_calls.jsonl``) — structured index for machine reading
    2. **Flat text** (``calls/NNNN_tool.txt``) — grep-friendly content
    """

    def __init__(
        self,
        agent_dir: Path,
        agent_name: str,
        *,
        formatter: Callable | None,
        redact: Callable | bool | None,
        max_output_bytes: int,
    ):
        self._agent_dir = agent_dir
        self._agent_name = agent_name
        self._calls_dir = agent_dir / "calls"
        self._jsonl_path = agent_dir / "tool_calls.jsonl"
        self._formatter = formatter
        self._redact = redact
        self._max_output_bytes = max_output_bytes
        self._lock = threading.Lock()
        self._seq = itertools.count()
        self._tool_counts: dict[str, int] = {}
        self._errors = 0
        self._artifacts: list[str] = []
        self._total_bytes = 0

        self._calls_dir.mkdir(parents=True, exist_ok=True)

    # ---- public API ----

    def record(
        self,
        tool_name: str,
        input_data: Any,
        output_data: Any,
        duration_ms: float,
        status: str,
        error: str | None = None,
    ) -> None:
        """Record a single tool call (thread-safe).

        The entire method body runs under a lock so that the sequence
        counter, flat-text file, and JSONL entry stay consistent.
        """
        with self._lock:
            seq = next(self._seq)
            now = time.time()

            # -- redaction --
            clean_input, clean_output = self._apply_redaction(
                tool_name, input_data, output_data
            )

            # -- format body --
            if error:
                body = f"error:\n{error}"
                if self._redact is not False:
                    body = _apply_builtin_redaction(body)
            elif isinstance(clean_output, bytes):
                bin_path = self._calls_dir / f"{seq:04d}_{_safe_name(tool_name)}.bin"
                bin_data = clean_output[: self._max_output_bytes]
                bin_path.write_bytes(bin_data)
                body = (
                    f"[binary output: {len(clean_output)} bytes]\n"
                    f"saved to: {bin_path.name}"
                )
            else:
                body = None
                if self._formatter:
                    body = self._formatter(tool_name, clean_input, clean_output)
                if body is None:
                    body = default_format(tool_name, clean_input, clean_output)

            # -- build flat text file --
            ts_str = datetime.fromtimestamp(now, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            header = (
                f"[tool_call #{seq}] {tool_name}\n"
                f"agent: {self._agent_name}\n"
                f"timestamp: {ts_str}\n"
                f"duration: {int(duration_ms)}ms\n"
            )
            if status != "ok":
                header += f"status: {status.upper()}\n"
            header += f"input: {_format_input(clean_input)}\n\n"

            if error:
                full_text = header + body
            else:
                full_text = header + "output:\n" + body

            # -- truncation --
            text_bytes = full_text.encode("utf-8", errors="replace")
            if len(text_bytes) > self._max_output_bytes:
                full_text = text_bytes[: self._max_output_bytes].decode(
                    "utf-8", errors="replace"
                )
                full_text += (
                    f"\n\n[TRUNCATED: output was {len(text_bytes)} bytes, "
                    f"showing first {self._max_output_bytes}]"
                )

            # -- write flat text --
            flat_path = self._calls_dir / f"{seq:04d}_{_safe_name(tool_name)}.txt"
            flat_path.write_text(full_text, encoding="utf-8")

            output_bytes = flat_path.stat().st_size
            line_count = full_text.count("\n") + 1

            # -- append JSONL entry --
            entry = {
                "seq": seq,
                "tool": tool_name,
                "agent": self._agent_name,
                "input": _serialize(clean_input),
                "output_file": f"calls/{flat_path.name}",
                "output_bytes": output_bytes,
                "timestamp": now,
                "duration_ms": round(duration_ms, 1),
                "status": status,
                "line_range": [1, line_count],
            }
            if error:
                entry["error"] = error

            with open(self._jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")

            # -- bookkeeping --
            self._tool_counts[tool_name] = self._tool_counts.get(tool_name, 0) + 1
            if status == "error":
                self._errors += 1
            self._total_bytes += output_bytes

    def add_artifact(self, filename: str) -> None:
        """Register an artifact for this agent."""
        if filename not in self._artifacts:
            self._artifacts.append(filename)

    def stats(self) -> dict:
        """Return stats for manifest generation."""
        with self._lock:
            total_calls = sum(self._tool_counts.values())
            return {
                "tool_calls": total_calls,
                "tools_used": list(self._tool_counts.keys()),
                "tool_counts": dict(self._tool_counts),
                "artifacts": list(self._artifacts),
                "errors": self._errors,
                "trace_dir": f"{self._agent_name}/",
                "trace_bytes": self._total_bytes,
            }

    # ---- internal ----

    def _apply_redaction(
        self, tool_name: str, input_data: Any, output_data: Any
    ) -> tuple[Any, Any]:
        """Apply custom callback then built-in pattern redaction."""
        clean_input = input_data
        clean_output = output_data

        if callable(self._redact):
            clean_input, clean_output = self._redact(
                tool_name, clean_input, clean_output
            )

        if self._redact is not False:
            clean_input = _redact_value(clean_input)
            clean_output = _redact_value(clean_output)

        return clean_input, clean_output


# ---------------------------------------------------------------------------
# Snapshot — top-level capture coordinator
# ---------------------------------------------------------------------------

class Snapshot:
    """Production-side tool call capture for post-hoc audit.

    Captures every tool call made by LangGraph agents and saves
    grep-friendly trace files for audit-fence verification.

    Primary usage (LangGraph — zero-touch)::

        snap = Snapshot("trace/")
        result = await agent.ainvoke(
            input, config=snap.config(agent="research")
        )
        snap.finalize()

    The trace directory can then be passed to ``fence.set_source()``
    for audit.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        session_id: str | None = None,
        formatter: Callable | None = None,
        redact: Callable | bool | None = None,
        max_output_bytes: int = 10_000_000,
    ):
        """
        Args:
            output_dir: Root directory for trace files.
            session_id: Unique session identifier.  Auto-generated as
                ``YYYYMMDD_HHMMSS_{8hex}`` (UTC) if not provided.
            formatter: Custom callable
                ``(tool_name, input, output) -> str | None``.
                Return formatted string, or ``None`` to fall through to
                :func:`default_format`.
            redact: Sanitisation callback
                ``(tool_name, input_data, output_data) -> (clean_input,
                clean_output)``.  Set to ``False`` to disable built-in
                API-key redaction.  ``None`` (default) enables built-in
                patterns only.
            max_output_bytes: Maximum bytes for a single tool call output
                file.  Outputs exceeding this are truncated with a note.
        """
        self._output_dir = Path(output_dir)
        self._session_id = session_id or self._generate_session_id()
        self._formatter = formatter
        self._redact = redact
        self._max_output_bytes = max_output_bytes
        self._writers: dict[str, _AgentWriter] = {}
        self._writers_lock = threading.Lock()
        self._dependencies: dict[str, list[str]] = {}
        self._created = time.time()
        self._finalized = False

        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ---- LangGraph integration (primary path) ----

    def config(self, *, agent: str = "default") -> dict:
        """Return a LangGraph-compatible ``RunnableConfig``.

        Pass the returned dict as the ``config`` parameter to any
        LangGraph agent invocation::

            result = await agent.ainvoke(
                input, config=snap.config(agent="research")
            )

        Args:
            agent: Name for this agent's trace directory.

        Returns:
            ``{"callbacks": [SnapshotHandler(...)]}``
        """
        return {"callbacks": [self.handler(agent=agent)]}

    def handler(self, *, agent: str = "default"):
        """Return the raw :class:`SnapshotHandler` for manual config assembly.

        Use this when you need to combine with other callbacks::

            config = {
                "callbacks": [snap.handler(agent="research"), my_logger],
                "recursion_limit": 50,
            }

        Args:
            agent: Name for this agent's trace directory.
        """
        from .snapshot_handler import SnapshotHandler

        writer = self._get_writer(agent)
        return SnapshotHandler(writer=writer, agent=agent)

    # ---- artifact management ----

    def save_artifact(
        self,
        agent: str,
        content: str,
        filename: str,
    ) -> Path:
        """Save an agent's intermediate output as a searchable text file.

        The artifact is saved under ``trace/{agent}/{filename}`` and is
        searchable via ripgrep during audit.

        Args:
            agent: Agent name (determines directory).
            content: Text content to save.
            filename: Output filename (e.g., ``"analysis.md"``).

        Returns:
            Absolute path to the saved file.
        """
        writer = self._get_writer(agent)
        artifact_path = self._output_dir / agent / filename
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(content, encoding="utf-8")
        writer.add_artifact(filename)
        return artifact_path

    def declare_dependency(
        self,
        agent: str,
        *,
        upstream: str | list[str],
    ) -> None:
        """Declare that one agent depends on another's output.

        Recorded in ``manifest.json`` and used by
        :meth:`FenceGroup.from_snapshot_manifest` to build fence link
        topology for multi-agent audit.

        Args:
            agent: The downstream agent name.
            upstream: Name(s) of upstream agent(s).
        """
        if isinstance(upstream, str):
            upstream = [upstream]
        existing = self._dependencies.get(agent, [])
        for u in upstream:
            if u not in existing:
                existing.append(u)
        self._dependencies[agent] = existing

    # ---- lifecycle ----

    def finalize(self, *, incomplete: bool = False) -> Path:
        """Write ``manifest.json`` and mark the snapshot as complete.

        Args:
            incomplete: If ``True``, marks the manifest as incomplete
                (set automatically when exiting a ``with`` block via
                exception).

        Returns:
            Path to the written ``manifest.json``.
        """
        if self._finalized:
            return self._output_dir / "manifest.json"

        finalized_time = time.time()

        agents: dict[str, dict] = {}
        total_calls = 0
        total_artifacts = 0
        total_errors = 0
        total_bytes = 0

        for agent_name, writer in self._writers.items():
            stats = writer.stats()
            agents[agent_name] = stats
            total_calls += stats["tool_calls"]
            total_artifacts += len(stats["artifacts"])
            total_errors += stats["errors"]
            total_bytes += stats["trace_bytes"]

        manifest = {
            "version": 1,
            "session_id": self._session_id,
            "created": datetime.fromtimestamp(
                self._created, tz=timezone.utc
            ).isoformat(),
            "finalized": datetime.fromtimestamp(
                finalized_time, tz=timezone.utc
            ).isoformat(),
            "duration_seconds": round(finalized_time - self._created, 1),
            "agents": agents,
            "dependencies": dict(self._dependencies),
            "totals": {
                "agents": len(agents),
                "tool_calls": total_calls,
                "artifacts": total_artifacts,
                "errors": total_errors,
                "trace_bytes": total_bytes,
            },
        }

        if incomplete:
            manifest["incomplete"] = True

        manifest_path = self._output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        self._finalized = True
        return manifest_path

    # ---- query methods ----

    @property
    def trace_dir(self) -> str:
        """Root trace directory path.  Pass to ``fence.set_source()``."""
        return str(self._output_dir)

    def agent_dir(self, agent: str) -> str:
        """Trace directory for a specific agent."""
        return str(self._output_dir / agent)

    def load_manifest(self) -> dict:
        """Load and return the manifest (after :meth:`finalize`).

        Raises:
            FileNotFoundError: If ``finalize()`` has not been called.
        """
        manifest_path = self._output_dir / "manifest.json"
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def build_index(self) -> dict[str, ToolCallRecord]:
        """Build tool call index from JSONL files.

        Returns:
            A dict mapping ``"{agent}:{seq}"`` to :class:`ToolCallRecord`
            for all captured calls across all agents.
        """
        index: dict[str, ToolCallRecord] = {}
        for agent_name in self._list_agents():
            jsonl_path = self._output_dir / agent_name / "tool_calls.jsonl"
            if not jsonl_path.exists():
                continue
            for line in jsonl_path.read_text(encoding="utf-8").strip().split("\n"):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = f"{agent_name}:{entry['seq']}"
                index[key] = ToolCallRecord(**{
                    k: v for k, v in entry.items()
                    if k in ToolCallRecord.__dataclass_fields__
                })
        return index

    def resolve_ref(
        self,
        search_file: str,
        search_line: int,
    ) -> ToolCallRecord | None:
        """Resolve grep coordinates to a :class:`ToolCallRecord`.

        Given a file path and line number from a grep result,
        identifies which tool call produced that output.

        Args:
            search_file: Relative or absolute path to a trace file.
            search_line: Line number within that file.

        Returns:
            The matching :class:`ToolCallRecord`, or ``None``.
        """
        # Normalise to a path relative to the trace directory.
        if os.path.isabs(search_file):
            try:
                rel_path = os.path.relpath(search_file, self._output_dir)
            except ValueError:
                return None
        else:
            rel_path = search_file

        parts = Path(rel_path).parts
        if len(parts) < 3 or parts[1] != "calls":
            return None

        agent = parts[0]
        target_file = str(Path(*parts[1:]))

        jsonl_path = self._output_dir / agent / "tool_calls.jsonl"
        if not jsonl_path.exists():
            return None

        for line in jsonl_path.read_text(encoding="utf-8").strip().split("\n"):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("output_file") == target_file:
                lr = entry.get("line_range", [])
                if len(lr) == 2 and lr[0] <= search_line <= lr[1]:
                    return ToolCallRecord(**{
                        k: v for k, v in entry.items()
                        if k in ToolCallRecord.__dataclass_fields__
                    })

        return None

    # ---- alternative integration paths ----

    def capture(self, *, agent: str = "default"):
        """Decorator to capture tool calls.

        Supports both sync and async functions::

            @snap.capture(agent="research")
            def get_stock_price(ticker: str) -> dict:
                return api.get(f"/stock/{ticker}")

            @snap.capture(agent="research")
            async def search_docs(query: str) -> list:
                return await vector_db.search(query)
        """

        def decorator(fn: Callable) -> Callable:
            writer = self._get_writer(agent)
            name = getattr(fn, "__name__", None) or getattr(fn, "name", str(fn))

            if inspect.iscoroutinefunction(fn):

                @wraps(fn)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    start = time.time()
                    input_data = kwargs or _positional_input(args)
                    try:
                        result = await fn(*args, **kwargs)
                        writer.record(
                            tool_name=name,
                            input_data=input_data,
                            output_data=result,
                            duration_ms=(time.time() - start) * 1000,
                            status="ok",
                        )
                        return result
                    except Exception as e:
                        writer.record(
                            tool_name=name,
                            input_data=input_data,
                            output_data=None,
                            duration_ms=(time.time() - start) * 1000,
                            status="error",
                            error=str(e),
                        )
                        raise

                return async_wrapper
            else:

                @wraps(fn)
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    start = time.time()
                    input_data = kwargs or _positional_input(args)
                    try:
                        result = fn(*args, **kwargs)
                        writer.record(
                            tool_name=name,
                            input_data=input_data,
                            output_data=result,
                            duration_ms=(time.time() - start) * 1000,
                            status="ok",
                        )
                        return result
                    except Exception as e:
                        writer.record(
                            tool_name=name,
                            input_data=input_data,
                            output_data=None,
                            duration_ms=(time.time() - start) * 1000,
                            status="error",
                            error=str(e),
                        )
                        raise

                return sync_wrapper

        return decorator

    def wrap(self, tools: list, *, agent: str = "default") -> list:
        """Wrap a list of callables with snapshot capture.

        Returns a new list of wrapped callables (originals untouched).

        Args:
            tools: List of callable tools.
            agent: Agent name for the trace directory.
        """
        return [self.capture(agent=agent)(tool) for tool in tools]

    # ---- context manager ----

    def __enter__(self) -> Snapshot:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.finalize(incomplete=exc_type is not None)

    def __repr__(self) -> str:
        agents = list(self._writers.keys())
        return (
            f"Snapshot(output_dir={str(self._output_dir)!r}, "
            f"session_id={self._session_id!r}, agents={agents})"
        )

    # ---- internal helpers ----

    @staticmethod
    def _generate_session_id() -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        rand = uuid.uuid4().hex[:8]
        return f"{ts}_{rand}"

    def _get_writer(self, agent: str) -> _AgentWriter:
        # Fast path — no lock needed if writer already exists.
        if agent in self._writers:
            return self._writers[agent]
        with self._writers_lock:
            # Re-check under lock (double-checked locking).
            if agent not in self._writers:
                agent_dir = self._output_dir / agent
                self._writers[agent] = _AgentWriter(
                    agent_dir,
                    agent,
                    formatter=self._formatter,
                    redact=self._redact,
                    max_output_bytes=self._max_output_bytes,
                )
            return self._writers[agent]

    def _list_agents(self) -> list[str]:
        """List agent names from writers or directory scan."""
        if self._writers:
            return list(self._writers.keys())
        agents = []
        if self._output_dir.exists():
            for p in sorted(self._output_dir.iterdir()):
                if p.is_dir() and (p / "tool_calls.jsonl").exists():
                    agents.append(p.name)
        return agents


def _positional_input(args: tuple) -> Any:
    """Convert positional args to a serialisable input representation."""
    if len(args) == 1:
        return args[0]
    return list(args) if args else {}
