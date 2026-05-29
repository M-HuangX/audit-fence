"""Core enforcement primitives: Fence, SearchRecord, validation logic."""

from __future__ import annotations

import fnmatch
import inspect
import json
import re
import time
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable


@dataclass
class SearchRecord:
    """A recorded search result."""

    query: str
    result_text: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)
    source: str = ""


@dataclass
class ValidationResult:
    """Result of validating output text against search history.

    Returned by :meth:`Fence.validate_output`.
    """

    found: list[str]
    """Quoted passages that matched search history."""

    not_found: list[str]
    """Quoted passages that did NOT match search history."""

    @property
    def total(self) -> int:
        """Total number of quoted passages examined."""
        return len(self.found) + len(self.not_found)

    @property
    def coverage(self) -> float:
        """Fraction of quoted passages that matched (0.0 to 1.0)."""
        if self.total == 0:
            return 1.0
        return len(self.found) / self.total

    @property
    def ok(self) -> bool:
        """True if all quoted passages matched search history."""
        return len(self.not_found) == 0


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

    # -- Linking -------------------------------------------------------------

    def link(self, *upstreams: Fence) -> Fence:
        """Declare that this fence can cite evidence from upstream fences.

        When this fence validates evidence (via @enforce, wrap(submit=...), or
        validate_output()), it checks its OWN search history plus the history
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

        def _record_result(result: Any, args: tuple, kwargs: dict) -> None:
            """Shared logic: append search result to history."""
            parts = [str(a) for a in args]
            parts.extend(f"{k}={v}" for k, v in kwargs.items())
            self._history.append(
                SearchRecord(
                    query=" ".join(parts) if parts else "(no args)",
                    result_text=str(result),
                    source=self._name or "",
                )
            )
            self._trim_history()

        if inspect.iscoroutinefunction(fn):

            @wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                result = await fn(*args, **kwargs)
                _record_result(result, args, kwargs)
                return result

            async_wrapper._fence_role = "search"  # type: ignore[attr-defined]
            self._search_fns.append(async_wrapper)
            return async_wrapper

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = fn(*args, **kwargs)
            _record_result(result, args, kwargs)
            return result

        wrapper._fence_role = "search"  # type: ignore[attr-defined]
        self._search_fns.append(wrapper)
        return wrapper

    def enforce(
        self,
        fn: Callable | None = None,
        *,
        evidence_param: str = "evidence",
        claim_param: str | None = None,
        source_text: str | Callable[[], str] | None = None,
        min_length: int | None = None,
        context: dict | None = None,
    ) -> Callable:
        """Decorator: register a submit tool with enforcement guards.

        Before the decorated function executes, validates:

        1. At least one search has been recorded (via ``@fence.track``).
        2. The ``evidence_param`` value is a substring of a recent search
           result (not semantic — exact character match).
        3. *(Optional)* The ``claim_param`` value exists in ``source_text``.

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
        per_tool_context = context

        def decorator(func: Callable) -> Callable:

            def _validate(args: tuple, kwargs: dict) -> str | None:
                """Shared validation logic. Returns error string or None."""
                evidence = _resolve_param(func, evidence_param, args, kwargs)

                # Check 1: search history must not be empty
                # (checks own history + upstream via _collect_history)
                if not self._collect_history():
                    err = (
                        "No search calls recorded. You must call a search "
                        "tool first to find evidence before submitting."
                    )
                    self._log_rejection(
                        func.__name__, evidence, err, per_tool_context
                    )
                    return f"ERROR: {err}"

                # Check 2: minimum evidence length
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
                    self._log_rejection(
                        func.__name__, evidence, err, per_tool_context
                    )
                    return f"ERROR: {err}"

                # Check 3: evidence must match a recent search result
                ok, err = self._verify_search_match(evidence)
                if not ok:
                    self._log_rejection(
                        func.__name__, evidence, err, per_tool_context
                    )
                    return f"ERROR: {err}"

                # Check 4 (optional): claim must exist in source text
                if claim_param is not None and source_text is not None:
                    claim = _resolve_param(func, claim_param, args, kwargs)
                    if claim:
                        src = (
                            source_text() if callable(source_text) else source_text
                        )
                        ok, err = _verify_source_match(claim, src)
                        if not ok:
                            self._log_rejection(
                                func.__name__, claim, err, per_tool_context
                            )
                            return f"ERROR: {err}"

                return None  # All checks passed

            if inspect.iscoroutinefunction(func):

                @wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    error = _validate(args, kwargs)
                    if error is not None:
                        return error
                    return await func(*args, **kwargs)

                async_wrapper._fence_role = "submit"  # type: ignore[attr-defined]
                self._submit_fns.append(async_wrapper)
                return async_wrapper

            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                error = _validate(args, kwargs)
                if error is not None:
                    return error
                return func(*args, **kwargs)

            wrapper._fence_role = "submit"  # type: ignore[attr-defined]
            self._submit_fns.append(wrapper)
            return wrapper

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
        """Append all rejection records to a JSONL file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for entry in self._rejections:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def reset(self) -> None:
        """Clear search history and rejection log."""
        self._history.clear()
        self._rejections.clear()

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

    # -- wrap / wrap_one -------------------------------------------------------

    def wrap(
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

    def wrap_one(
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

    # -- validate_output -------------------------------------------------------

    def validate_output(self, text: str) -> ValidationResult:
        """Validate output text against search history (soft enforcement).

        Extracts quoted/cited passages from *text* and checks each against
        the search history.  Returns a :class:`ValidationResult` with
        ``found``, ``not_found``, and ``coverage`` stats.

        This enables the "soft enforcement" pattern for agents that produce
        a final report without explicit submit tools.

        Args:
            text: The agent's output text (e.g. a final report).

        Returns:
            A :class:`ValidationResult` with match details.
        """
        quotes = _extract_quotes(text)
        all_records = self._collect_history()
        all_text = "\n".join(r.result_text for r in all_records)

        found: list[str] = []
        not_found: list[str] = []

        for quote in quotes:
            normalized = quote.strip()
            if len(normalized) < 10:
                # Too short to be meaningful — skip
                continue
            if normalized in all_text:
                found.append(quote)
            elif _number_match(normalized, all_text):
                found.append(quote)
            else:
                not_found.append(quote)

        return ValidationResult(found=found, not_found=not_found)

    # -- Internal wrapping helpers ---------------------------------------------

    def _wrap_as_search(self, fn: Callable) -> Callable:
        """Wrap *fn* with search-tracking behaviour."""

        def _record(result: Any, args: tuple, kwargs: dict) -> None:
            parts = [str(a) for a in args]
            parts.extend(f"{k}={v}" for k, v in kwargs.items())
            self._history.append(
                SearchRecord(
                    query=" ".join(parts) if parts else "(no args)",
                    result_text=str(result),
                    source=self._name or "",
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

    def _wrap_as_submit(self, fn: Callable, evidence_param: str) -> Callable:
        """Wrap *fn* with enforcement behaviour."""
        tool_name = _get_tool_name(fn)

        def _validate(args: tuple, kwargs: dict) -> str | None:
            evidence = _resolve_param(fn, evidence_param, args, kwargs)

            if not self._collect_history():
                err = (
                    "No search calls recorded. You must call a search "
                    "tool first to find evidence before submitting."
                )
                self._log_rejection(tool_name, evidence, err)
                return f"ERROR: {err}"

            if len(evidence.strip()) < self._min_evidence_length:
                err = (
                    f"Evidence too short (got {len(evidence.strip())} chars, "
                    f"min {self._min_evidence_length}). Paste actual search output."
                )
                self._log_rejection(tool_name, evidence, err)
                return f"ERROR: {err}"

            ok, err = self._verify_search_match(evidence)
            if not ok:
                self._log_rejection(tool_name, evidence, err)
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


# Regex to extract quoted passages: "..." or '...' (at least 10 chars inside)
_QUOTE_PATTERN = re.compile(r'["\u201c]([^"\u201d]{10,})["\u201d]')


def _extract_quotes(text: str) -> list[str]:
    """Extract quoted passages from text for validation.

    Looks for double-quoted strings (ASCII ``"`` or Unicode curly quotes)
    that are at least 10 characters long.
    """
    return [m.group(1) for m in _QUOTE_PATTERN.finditer(text)]


def _verify_source_match(claim: str, source_text: str) -> tuple[bool, str]:
    """Verify that claim text exists as a substring in the source document."""
    if not claim or not source_text:
        return True, ""

    norm_claim = _normalize(claim)
    norm_source = _normalize(source_text)

    if norm_claim in norm_source:
        return True, ""

    return False, (
        "Claim text not found in the source document. "
        "Copy the EXACT text from the source."
    )


def _normalize(text: str) -> str:
    """Normalize text for substring comparison: strip markdown, collapse whitespace, lowercase."""
    text = re.sub(r"\*\*|__|\*|_|`", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


# -- Number format matching --------------------------------------------------

# Pattern: optional sign, digits with optional commas, optional decimal, optional suffix
_NUM_PATTERN = re.compile(
    r"[-+]?"
    r"\d[\d,]*"           # integer part (may have commas)
    r"(?:\.\d+)?"         # optional decimal
    r"[%KMBTkmbt]?"        # optional suffix (must be adjacent, no whitespace)
)

_SUFFIX_MULTIPLIERS: dict[str, float] = {
    "k": 1e3,
    "m": 1e6,
    "b": 1e9,
    "t": 1e12,
}


def normalize_number(text: str) -> float | None:
    """Parse a human-readable number string into a float.

    Handles:
    - Comma-separated numbers: ``"5,098,000,000"`` -> ``5098000000.0``
    - Suffix abbreviations: ``"5.1B"`` -> ``5100000000.0``
    - Percentages: ``"26.2%"`` -> ``0.262``
    - Plain numbers: ``"18.923"`` -> ``18.923``

    Returns ``None`` if the string does not look like a number.
    """
    text = text.strip()
    if not text:
        return None

    # Remove leading currency signs
    text = text.lstrip("$")

    # Check for percentage
    is_pct = text.endswith("%")
    if is_pct:
        text = text[:-1].strip()

    # Remove commas
    text = text.replace(",", "")

    # Extract suffix
    suffix = ""
    if text and text[-1].lower() in _SUFFIX_MULTIPLIERS:
        suffix = text[-1].lower()
        text = text[:-1].strip()

    try:
        value = float(text)
    except (ValueError, TypeError):
        return None

    if suffix:
        value *= _SUFFIX_MULTIPLIERS[suffix]
    if is_pct:
        value /= 100.0

    return value


def extract_numbers(text: str) -> list[float]:
    """Extract all numeric values from a text string.

    Finds numbers with optional K/M/B/T suffixes, commas, and percentage
    signs, then normalizes each to a float.

    Example::

        >>> extract_numbers("Revenue $5.1B, up 26.2% YoY")
        [5100000000.0, 0.262]
    """
    results: list[float] = []
    for match in _NUM_PATTERN.finditer(text):
        token = match.group()
        val = normalize_number(token)
        if val is not None:
            results.append(val)
    return results


def _numbers_overlap(nums_a: list[float], nums_b: list[float]) -> bool:
    """Check if any number from list A matches any number from list B.

    Two numbers match if they are within 0.1% relative tolerance of each
    other, handling float imprecision from suffix expansion.
    """
    for a in nums_a:
        for b in nums_b:
            if a == 0 and b == 0:
                return True
            if a == 0 or b == 0:
                continue
            rel = abs(a - b) / max(abs(a), abs(b))
            if rel < 0.001:
                return True
    return False


def _number_match(evidence_line: str, search_text: str) -> bool:
    """Fallback matching: check if a line's numbers match numbers in search results.

    Only activates when the evidence line contains at least one number.
    Requires at least one number overlap AND some non-numeric text overlap
    (at least 3 words in common) to prevent pure-number false positives.
    """
    ev_nums = extract_numbers(evidence_line)
    if not ev_nums:
        return False

    sr_nums = extract_numbers(search_text)
    if not sr_nums:
        return False

    if not _numbers_overlap(ev_nums, sr_nums):
        return False

    # Require some non-numeric textual overlap to prevent false positives
    ev_words = set(re.sub(r"[^a-zA-Z\s]", "", evidence_line.lower()).split())
    sr_words = set(re.sub(r"[^a-zA-Z\s]", "", search_text.lower()).split())
    # Remove very common words
    stopwords = {"the", "a", "an", "in", "of", "to", "and", "or", "is", "was", "for", "on", "at", "by", "with", "from"}
    ev_words -= stopwords
    sr_words -= stopwords
    common = ev_words & sr_words
    return len(common) >= 2
