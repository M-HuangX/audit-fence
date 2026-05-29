"""Core enforcement primitives: Fence, SearchRecord, validation logic."""

from __future__ import annotations

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
    """

    def __init__(
        self,
        *,
        min_evidence_length: int = 20,
        history_window: int = 20,
    ):
        self._history: list[SearchRecord] = []
        self._rejections: list[dict] = []
        self._min_evidence_length = min_evidence_length
        self._history_window = history_window
        self._search_fns: list[Callable] = []
        self._submit_fns: list[Callable] = []

    # -- Public properties ---------------------------------------------------

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

    # -- Decorators ----------------------------------------------------------

    def track(self, fn: Callable) -> Callable:
        """Decorator: register a search tool whose results are tracked.

        Every call's return value is appended to the search history.
        Evidence submitted via ``@fence.enforce`` must match a recent
        tracked result.
        """

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = fn(*args, **kwargs)
            # Build query description from arguments
            parts = [str(a) for a in args]
            parts.extend(f"{k}={v}" for k, v in kwargs.items())
            self._history.append(
                SearchRecord(
                    query=" ".join(parts) if parts else "(no args)",
                    result_text=str(result),
                )
            )
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
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                evidence = _resolve_param(func, evidence_param, args, kwargs)

                # Check 1: search history must not be empty
                if not self._history:
                    err = (
                        "No search calls recorded. You must call a search "
                        "tool first to find evidence before submitting."
                    )
                    self._log_rejection(func.__name__, evidence, err)
                    return f"ERROR: {err}"

                # Check 2: minimum evidence length
                eff_min = min_length if min_length is not None else self._min_evidence_length
                if len(evidence.strip()) < eff_min:
                    err = (
                        f"Evidence too short (got {len(evidence.strip())} chars, "
                        f"min {eff_min}). Paste actual search output."
                    )
                    self._log_rejection(func.__name__, evidence, err)
                    return f"ERROR: {err}"

                # Check 3: evidence must match a recent search result
                ok, err = self._verify_search_match(evidence)
                if not ok:
                    self._log_rejection(func.__name__, evidence, err)
                    return f"ERROR: {err}"

                # Check 4 (optional): claim must exist in source text
                if claim_param is not None and source_text is not None:
                    claim = _resolve_param(func, claim_param, args, kwargs)
                    if claim:
                        src = source_text() if callable(source_text) else source_text
                        ok, err = _verify_source_match(claim, src)
                        if not ok:
                            self._log_rejection(func.__name__, claim, err)
                            return f"ERROR: {err}"

                return func(*args, **kwargs)

            wrapper._fence_role = "submit"  # type: ignore[attr-defined]
            self._submit_fns.append(wrapper)
            return wrapper

        # Support both @fence.enforce and @fence.enforce(...)
        if fn is not None:
            return decorator(fn)
        return decorator

    # -- Validation ----------------------------------------------------------

    def _verify_search_match(self, evidence: str) -> tuple[bool, str]:
        """Verify that evidence text matches a recent search result.

        Checks the last ``history_window`` search results for a substring
        match. This is intentionally **not** semantic similarity — it is
        exact character-level containment, which prevents the LLM from
        paraphrasing or fabricating evidence.
        """
        recent = self._history[-self._history_window :]
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

        return False, (
            "Evidence does not match any recent search result. "
            "Call a search tool first, then paste the matching "
            "output into the evidence field."
        )

    # -- Logging -------------------------------------------------------------

    def _log_rejection(self, tool_name: str, content: str, reason: str) -> None:
        self._rejections.append(
            {
                "tool": tool_name,
                "content": content[:200],
                "reason": reason,
                "timestamp": time.time(),
            }
        )

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
