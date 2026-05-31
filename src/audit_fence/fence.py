"""Core enforcement primitives: Fence, SearchRecord, validation logic."""

from __future__ import annotations

import fnmatch
import inspect
import itertools
import json
import time
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .workflow import ClaimRecord

from .matching import (
    _normalize,
    _number_match,
)


@dataclass
class SearchRecord:
    """A recorded search result."""

    query: str
    result_text: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)
    source: str = ""
    tool_name: str = ""
    """Which tool produced this result (e.g. 'get_stock_info')."""
    file_path: str = ""
    """File where result was found (e.g. grep result path)."""


class Fence:
    """Programmatic enforcement for LLM agent evidence submission.

    Forces agents to search before submitting evidence, then validates
    that submitted evidence actually matches real search results.

    Usage::

        fence = Fence()

        @fence.track
        def search(query: str) -> str:
            return my_search_backend(query)

        @fence.enforce
        def record_citation(claim: str, evidence: str) -> dict:
            return {"claim": claim, "status": "recorded"}

    The ``@fence.track`` decorator records every search result.
    The ``@fence.enforce`` decorator validates that the ``evidence``
    parameter matches a recent search result before allowing execution.

    Async functions are transparently supported::

        @fence.track
        async def search(query: str) -> str:
            return await my_async_backend(query)

        @fence.enforce
        async def record_citation(claim: str, evidence: str) -> dict:
            return {"claim": claim, "status": "recorded"}
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        min_evidence_length: int = 20,
        history_window: int = 20,
        history_limit: int | None = None,
        context: dict | None = None,
        track_all: bool = False,
    ):
        self._name = name
        self._history: list[SearchRecord] = []
        self._rejections: list[dict] = []
        self._min_evidence_length = min_evidence_length
        self._history_window = history_window
        self._history_limit = history_limit
        self._context: dict = context or {}
        self._search_fns: list[Callable] = []
        self._submit_fns: list[Callable] = []
        self._track_all = track_all
        self._upstream: list[Fence] = []
        # Workflow layer attributes
        self._document: str | Callable[[], str] | None = None
        self._output_path: str | None = None
        self._claims: list[ClaimRecord] = []
        self._search_fn: Callable | None = None  # set by set_source()
        self._next_claim_id: Callable[[], int] = itertools.count(1).__next__

    # -- Public properties ---------------------------------------------------

    @property
    def name(self) -> str | None:
        """The fence's name, or None if unnamed."""
        return self._name

    @property
    def history(self) -> list[SearchRecord]:
        """All recorded search results."""
        return list(self._history)

    @property
    def rejections(self) -> list[dict]:
        """All recorded enforcement rejections."""
        return list(self._rejections)

    @property
    def tools(self) -> list[Callable]:
        """All registered tools (search + submit), in registration order."""
        return self._search_fns + self._submit_fns

    @property
    def claims(self) -> list:
        """All recorded ClaimRecords."""
        return list(self._claims)

    # -- Document & output ---------------------------------------------------

    def set_document(self, text: str | Callable[[], str]) -> None:
        """Set the document being audited.

        When a document is set, any ``@fence.enforce`` tool whose function
        has a ``claim_in_document`` parameter will automatically verify
        that the parameter value is a normalized substring of this
        document.

        Args:
            text: The document text, or a callable that returns it
                (for dynamic content).
        """
        self._document = text

    def set_source(self, path: str, **kwargs: Any) -> None:
        """Set the source data directory for the audit agent to search.

        Creates a :class:`~audit_fence.tools.RipgrepBackend` pointed at
        *path* and registers it as a tracked search tool.  The resulting
        search function is available as :attr:`search`.

        Args:
            path: Directory containing source data / trace files.
            **kwargs: Forwarded to :class:`RipgrepBackend` (e.g.
                ``max_matches``, ``rg_path``).
        """
        from .tools import RipgrepBackend

        grep = RipgrepBackend(root=path, **kwargs)
        self._search_fn = self.wrap_tool(grep, role="search")

    @property
    def search(self) -> Callable:
        """The search tool created by :meth:`set_source`.

        Raises:
            RuntimeError: If :meth:`set_source` has not been called.
        """
        if self._search_fn is None:
            raise RuntimeError(
                "No search tool configured. Call fence.set_source(path) first."
            )
        return self._search_fn

    async def audit(self, llm: Any, **kwargs: Any) -> Any:
        """Run a complete audit using a pre-built ReAct agent.

        This is the simplest way to run an audit — one method call.
        Requires ``langgraph`` and a LangChain-compatible chat model.

        Args:
            llm: Any LangChain-compatible chat model
                (``ChatOpenAI``, ``ChatAnthropic``, etc.).
            **kwargs: Forwarded to :func:`~audit_fence.agent.run_audit`
                (``max_rounds``, ``timeout``, ``extra_fields``,
                ``manifest``, etc.).

        Returns:
            :class:`~audit_fence.agent.AuditResult` with claims,
            rejections, and summary.

        Example::

            from langchain_openai import ChatOpenAI
            result = await fence.audit(llm=ChatOpenAI(model="gpt-4o"))

            # With manifest for guided navigation:
            result = await fence.audit(
                llm=llm, manifest=snapshot.load_manifest()
            )
        """
        from .agent import run_audit
        return await run_audit(self, llm, **kwargs)

    def record_tool(self, **kwargs: Any) -> Callable:
        """Create an enforcement-checked record tool bound to this fence.

        Convenience method equivalent to
        ``create_record_tool(fence, ...)``.  See
        :func:`~audit_fence.workflow.create_record_tool` for full
        parameter documentation.

        Returns:
            A callable record tool with fence enforcement.
        """
        from .workflow import create_record_tool
        return create_record_tool(self, **kwargs)

    def set_output(self, path: str) -> None:
        """Set the JSONL output file path for claim persistence.

        When set, every successful record call that produces a
        :class:`~audit_fence.workflow.ClaimRecord` will auto-append to
        this file.

        Args:
            path: File path for JSONL output.
        """
        self._output_path = path

    def save_claims(self, path: str | None = None) -> None:
        """Write all recorded claims to a JSONL file.

        Args:
            path: Output file path. If None, uses the path set by
                :meth:`set_output`. Raises ValueError if neither is set.
        """
        target = path or self._output_path
        if target is None:
            raise ValueError(
                "No output path set. Call set_output() or pass a path."
            )
        p = Path(target)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for claim in self._claims:
                f.write(
                    json.dumps(claim.to_dict(), ensure_ascii=False) + "\n"
                )

    @property
    def _resolved_document(self) -> str | None:
        """Resolve the document text (handles callables)."""
        if self._document is None:
            return None
        if callable(self._document):
            return self._document()
        return self._document

    def _check_claim_in_document(self, claim_text: str) -> str | None:
        """Verify claim_in_document is a substring of the audited document.

        Returns an ERROR string if the check fails, or None if it passes
        (or if no document is set).
        """
        doc = self._resolved_document
        if doc is None:
            return None  # No document set, skip check
        if not claim_text:
            return None  # Empty claim, skip check

        norm_claim = _normalize(claim_text)
        norm_doc = _normalize(doc)

        if norm_claim in norm_doc:
            return None  # Match found

        err = (
            "Claim text not found in the audited document. "
            "Copy the EXACT text."
        )
        self._log_rejection("claim_in_document", claim_text, err)
        return f"ERROR: {err}"

    # -- Linking -------------------------------------------------------------

    def link(self, *upstreams: Fence) -> Fence:
        """Declare that this fence can cite evidence from upstream fences.

        When this fence validates evidence (via @enforce or wrap(submit=...)),
        it checks its OWN search history plus the history
        of all linked upstream fences, transitively.

        Can be called multiple times to add more upstreams.
        Duplicate links are silently ignored.

        Args:
            *upstreams: One or more Fence instances.

        Returns:
            self (for method chaining).

        Raises:
            TypeError: If any argument is not a Fence instance.

        Example::

            worker = Fence(name="worker")
            manager = Fence(name="manager")
            manager.link(worker)

            # Chaining:
            manager.link(worker_a).link(worker_b)

            # Multiple at once:
            manager.link(worker_a, worker_b, worker_c)
        """
        for upstream in upstreams:
            if not isinstance(upstream, Fence):
                raise TypeError(
                    f"link() requires Fence instances, got {type(upstream).__name__}"
                )
            if upstream not in self._upstream:
                self._upstream.append(upstream)
        return self

    # -- Decorators ----------------------------------------------------------

    def track(self, fn: Callable) -> Callable:
        """Decorator: register a search tool whose results are tracked.

        Every call's return value is appended to the search history.
        Evidence submitted via ``@fence.enforce`` must match a recent
        tracked result.

        Works with both sync and async functions.
        """
        return self._wrap_as_search(fn)

    def enforce(
        self,
        fn: Callable | None = None,
        *,
        evidence_param: str = "evidence",
        min_length: int | None = None,
        context: dict | None = None,
    ) -> Callable:
        """Decorator: register a submit tool with enforcement guards.

        Before the decorated function executes, validates:

        1. The ``claim_in_document`` parameter (if present) exists in the
           document set via :meth:`set_document`.
        2. At least one search has been recorded (via ``@fence.track``).
        3. The ``evidence_param`` value is a substring of a recent search
           result (not semantic — exact character match).

        If any check fails, the function is **not called**. Instead an
        ``"ERROR: ..."`` string is returned and the rejection is logged.

        Can be used with or without arguments::

            @fence.enforce
            def submit(claim, evidence): ...

            @fence.enforce(evidence_param="grep_output", min_length=30)
            def submit(claim, grep_output): ...

        The optional ``context`` dict is merged with the fence-level context
        and attached to every rejection from this tool.
        """

        def decorator(func: Callable) -> Callable:
            return self._wrap_as_submit(
                func,
                evidence_param,
                min_length=min_length,
                context=context,
            )

        # Support both @fence.enforce and @fence.enforce(...)
        if fn is not None:
            return decorator(fn)
        return decorator

    # -- History collection ---------------------------------------------------

    def _collect_history(self) -> list[SearchRecord]:
        """Collect search history from this fence and all linked upstream fences.

        Performs breadth-first traversal of the upstream DAG.
        Each fence contributes its most recent ``history_window`` records.
        Cycle-safe via visited set.

        Returns:
            Combined list of SearchRecord from this fence and all upstream
            fences.  Not sorted -- caller should sort by timestamp if needed.
        """
        visited: set[int] = set()
        records: list[SearchRecord] = []
        queue: list[Fence] = [self]

        while queue:
            current = queue.pop(0)
            current_id = id(current)
            if current_id in visited:
                continue
            visited.add(current_id)

            # Each fence contributes its own window
            window = current._history[-current._history_window:]
            records.extend(window)

            # Enqueue upstream fences
            for upstream in current._upstream:
                if id(upstream) not in visited:
                    queue.append(upstream)

        return records

    # -- Core enforcement checks ---------------------------------------------

    def _check_evidence(
        self,
        evidence: str,
        tool_name: str,
        *,
        min_length: int | None = None,
        context: dict | None = None,
    ) -> str | None:
        """Run the three core enforcement checks on evidence.

        1. Search history must exist.
        2. Evidence must meet minimum length.
        3. Evidence must match a recent search result.

        Returns the raw error reason on failure (and logs the rejection),
        or ``None`` on success.
        """
        if not self._collect_history():
            err = (
                "No search calls recorded. You must call a search "
                "tool first to find evidence before submitting."
            )
            self._log_rejection(tool_name, evidence, err, context)
            return err

        eff_min = (
            min_length
            if min_length is not None
            else self._min_evidence_length
        )
        if len(evidence.strip()) < eff_min:
            err = (
                f"Evidence too short (got {len(evidence.strip())} chars, "
                f"min {eff_min}). Paste actual search output."
            )
            self._log_rejection(tool_name, evidence, err, context)
            return err

        ok, err = self._verify_search_match(evidence)
        if not ok:
            self._log_rejection(tool_name, evidence, err, context)
            return err

        return None

    # -- Validation ----------------------------------------------------------

    def _verify_search_match(self, evidence: str) -> tuple[bool, str]:
        """Verify that evidence text matches a recent search result.

        Checks this fence's recent search results (and all linked upstream
        fences' results) for a substring match.  This is intentionally
        **not** semantic similarity — it is exact character-level
        containment, which prevents the LLM from paraphrasing or
        fabricating evidence.

        Falls back to number-format matching for lines that contain numeric
        values (e.g. "5.1B" matches "5100000000").
        """
        recent = self._collect_history()
        all_text = "\n".join(r.result_text for r in recent)

        lines = [l.strip() for l in evidence.strip().split("\n") if l.strip()]
        if not lines:
            return False, "Evidence is empty."

        for line in lines[:5]:
            normalized = line.lstrip("> ").strip()
            if len(normalized) < 10:
                continue

            # Direct substring match
            if normalized in all_text:
                return True, ""

            # Try stripping file:line: prefix (common grep output format)
            parts = normalized.split(":", 2)
            if len(parts) >= 3 and parts[1].strip().lstrip("-").isdigit():
                content = parts[2].strip()
                if len(content) >= 15 and content in all_text:
                    return True, ""

            # Fallback: number-format matching
            if _number_match(normalized, all_text):
                return True, ""

        return False, (
            "Evidence does not match any recent search result. "
            "Call a search tool first, then paste the matching "
            "output into the evidence field."
        )

    # -- Logging -------------------------------------------------------------

    def _log_rejection(
        self,
        tool_name: str,
        content: str,
        reason: str,
        per_tool_context: dict | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "tool": tool_name,
            "content": content[:200],
            "reason": reason,
            "timestamp": time.time(),
        }
        # Merge fence-level context, then per-tool context (per-tool wins)
        merged = {**self._context}
        if per_tool_context:
            merged.update(per_tool_context)
        if merged:
            entry["context"] = merged
        self._rejections.append(entry)

    def save_log(self, path: str | Path) -> None:
        """Write all rejection records to a JSONL file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for entry in self._rejections:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def reset(self) -> None:
        """Clear search history, rejection log, and recorded claims."""
        self._history.clear()
        self._rejections.clear()
        self._claims.clear()

    # -- Serialization --------------------------------------------------------

    def _snapshot(self) -> dict:
        """Serialize this fence's state (internal, used by FenceGroup)."""
        return {
            "name": self._name,
            "history": [
                {
                    "query": r.query,
                    "result_text": r.result_text,
                    "timestamp": r.timestamp,
                    "metadata": r.metadata,
                    "source": r.source,
                    "tool_name": r.tool_name,
                    "file_path": r.file_path,
                }
                for r in self._history
            ],
            "rejections": list(self._rejections),
            "config": {
                "min_evidence_length": self._min_evidence_length,
                "history_window": self._history_window,
                "history_limit": self._history_limit,
                "context": self._context,
                "track_all": self._track_all,
            },
        }

    def _restore(self, data: dict) -> None:
        """Restore this fence's state from a snapshot (internal)."""
        self._history = [
            SearchRecord(**r) for r in data.get("history", [])
        ]
        self._rejections = data.get("rejections", [])
        config = data.get("config", {})
        if "min_evidence_length" in config:
            self._min_evidence_length = config["min_evidence_length"]
        if "history_window" in config:
            self._history_window = config["history_window"]
        if "history_limit" in config:
            self._history_limit = config["history_limit"]
        if "context" in config:
            self._context = config["context"]
        if "track_all" in config:
            self._track_all = config["track_all"]

    def snapshot(self) -> dict:
        """Serialize this fence's state for persistence.

        Returns a JSON-serializable dict.  Restore with
        :meth:`Fence.restore`.
        """
        return self._snapshot()

    @classmethod
    def restore(cls, data: dict) -> Fence:
        """Restore a Fence from a snapshot dict.

        Args:
            data: A dict previously returned by :meth:`snapshot`.

        Returns:
            A new Fence instance with the restored state.
        """
        fence = cls(name=data.get("name"))
        fence._restore(data)
        return fence

    # -- inject / drop_last ---------------------------------------------------

    def inject(self, record: SearchRecord) -> None:
        """Manually add a search record to this fence's history.

        Useful for human-in-the-loop workflows where a human approves
        evidence that should be available for enforcement.

        Args:
            record: A :class:`SearchRecord` to add.
        """
        self._history.append(record)
        self._trim_history()

    def drop_last(self, n: int = 1) -> None:
        """Remove the last *n* entries from this fence's history.

        Useful for retry/rollback scenarios where stale or incorrect
        search results should be discarded.

        Args:
            n: Number of entries to remove (default 1).
        """
        if n >= len(self._history):
            self._history.clear()
        else:
            del self._history[-n:]

    # -- Memory management ---------------------------------------------------

    def _trim_history(self) -> None:
        """Trim history to ``history_limit`` if set."""
        if self._history_limit is not None and len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit:]

    # -- wrap_tools / wrap_tool ------------------------------------------------

    def wrap_tools(
        self,
        tools: list[Callable],
        *,
        search: list[str | Callable] | None = None,
        submit: list[str | Callable] | None = None,
        evidence_param: str = "evidence",
    ) -> list[Callable]:
        """Wrap a list of tools with audit enforcement.

        Tools matching *search* patterns get :meth:`track` behaviour (results
        are recorded in search history).  Tools matching *submit* patterns
        get :meth:`enforce` behaviour (evidence is validated before execution).
        Unmatched tools pass through unchanged.

        When the fence was created with ``track_all=True`` and no *search* or
        *submit* lists are provided, **every** tool is tracked (equivalent to
        ``search=["*"]``).

        Args:
            tools: Existing tool functions.
            search: Glob pattern strings (matched against ``fn.__name__``) or
                direct function references identifying search tools.  May be
                mixed.
            submit: Same format as *search*, identifying submit tools.
            evidence_param: Name of the parameter containing evidence text on
                submit tools (default ``"evidence"``).

        Returns:
            A new list of (possibly wrapped) callables in the same order as
            *tools*.  Originals are not mutated.
        """
        search_specs = search or []
        submit_specs = submit or []
        result: list[Callable] = []

        for tool in tools:
            # Already wrapped by this fence — pass through (no double-wrap)
            if getattr(tool, "_fence_role", None) is not None:
                result.append(tool)
                continue

            name = _get_tool_name(tool)

            if _matches_spec(tool, name, search_specs):
                result.append(self._wrap_as_search(tool))
            elif _matches_spec(tool, name, submit_specs):
                result.append(self._wrap_as_submit(tool, evidence_param))
            elif self._track_all and not search_specs and not submit_specs:
                # track_all mode with no explicit patterns → track everything
                result.append(self._wrap_as_search(tool))
            else:
                result.append(tool)

        return result

    def wrap_tool(
        self,
        fn: Callable,
        role: str,
        *,
        evidence_param: str = "evidence",
    ) -> Callable:
        """Wrap a single tool programmatically (without decorators).

        Args:
            fn: The tool function to wrap.
            role: ``"search"`` or ``"submit"``.
            evidence_param: For submit tools, the parameter name containing
                evidence (default ``"evidence"``).

        Returns:
            A wrapped callable with the appropriate fence behaviour.

        Raises:
            ValueError: If *role* is not ``"search"`` or ``"submit"``.
        """
        if role == "search":
            return self._wrap_as_search(fn)
        elif role == "submit":
            return self._wrap_as_submit(fn, evidence_param)
        else:
            raise ValueError(f"role must be 'search' or 'submit', got {role!r}")

    # -- Internal wrapping helpers ---------------------------------------------

    def _wrap_as_search(self, fn: Callable) -> Callable:
        """Wrap *fn* with search-tracking behaviour."""
        tool_name = _get_tool_name(fn)

        def _record(result: Any, args: tuple, kwargs: dict) -> None:
            parts = [str(a) for a in args]
            parts.extend(f"{k}={v}" for k, v in kwargs.items())
            self._history.append(
                SearchRecord(
                    query=" ".join(parts) if parts else "(no args)",
                    result_text=str(result),
                    source=self._name or "",
                    tool_name=tool_name,
                )
            )
            self._trim_history()

        if inspect.iscoroutinefunction(fn):

            @wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                result = await fn(*args, **kwargs)
                _record(result, args, kwargs)
                return result

            async_wrapper._fence_role = "search"  # type: ignore[attr-defined]
            self._search_fns.append(async_wrapper)
            return async_wrapper

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = fn(*args, **kwargs)
            _record(result, args, kwargs)
            return result

        wrapper._fence_role = "search"  # type: ignore[attr-defined]
        self._search_fns.append(wrapper)
        return wrapper

    def _wrap_as_submit(
        self,
        fn: Callable,
        evidence_param: str = "evidence",
        *,
        min_length: int | None = None,
        context: dict | None = None,
    ) -> Callable:
        """Wrap *fn* with enforcement behaviour.

        Used internally by both :meth:`enforce` (decorator API) and
        :meth:`wrap_tool` / :meth:`wrap_tools` (programmatic API).
        """
        tool_name = _get_tool_name(fn)

        def _validate(args: tuple, kwargs: dict) -> str | None:
            # 1. Claim-in-document check (cheap text match)
            cid = _resolve_param(fn, "claim_in_document", args, kwargs)
            if cid:
                doc_err = self._check_claim_in_document(cid)
                if doc_err is not None:
                    return doc_err

            # 2. Evidence enforcement (history + length + search match)
            evidence = _resolve_param(fn, evidence_param, args, kwargs)

            err = self._check_evidence(
                evidence, tool_name, min_length=min_length, context=context,
            )
            if err is not None:
                return f"ERROR: {err}"

            return None

        if inspect.iscoroutinefunction(fn):

            @wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                error = _validate(args, kwargs)
                if error is not None:
                    return error
                return await fn(*args, **kwargs)

            async_wrapper._fence_role = "submit"  # type: ignore[attr-defined]
            self._submit_fns.append(async_wrapper)
            return async_wrapper

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            error = _validate(args, kwargs)
            if error is not None:
                return error
            return fn(*args, **kwargs)

        wrapper._fence_role = "submit"  # type: ignore[attr-defined]
        self._submit_fns.append(wrapper)
        return wrapper


# -- Module-level helpers (stateless) ----------------------------------------


def _resolve_param(
    func: Callable, param_name: str, args: tuple, kwargs: dict
) -> str:
    """Extract a named parameter from args/kwargs."""
    if param_name in kwargs:
        return str(kwargs[param_name])
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    if param_name in params:
        idx = params.index(param_name)
        if idx < len(args):
            return str(args[idx])
    return ""


def _get_tool_name(tool: Any) -> str:
    """Extract a human-readable name from a tool.

    Checks (in order): ``__name__`` (plain functions), ``.name`` (LangChain
    ``StructuredTool``), then falls back to ``str(tool)``.
    """
    if hasattr(tool, "__name__"):
        return tool.__name__
    if hasattr(tool, "name"):
        return tool.name
    return str(tool)


def _matches_spec(
    tool: Callable, name: str, specs: list[str | Callable]
) -> bool:
    """Check whether *tool* matches any entry in *specs*.

    Each spec can be:
    - A string glob pattern matched against *name* (e.g. ``"search_*"``).
    - A direct function reference (identity comparison with *tool*).
    """
    for spec in specs:
        if callable(spec) and not isinstance(spec, str):
            if spec is tool:
                return True
        elif isinstance(spec, str):
            if fnmatch.fnmatch(name, spec):
                return True
    return False
