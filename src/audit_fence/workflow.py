"""Workflow layer: ClaimRecord, record tool factory, JSONL persistence."""

from __future__ import annotations

import inspect
import json
import time
from dataclasses import asdict, dataclass, field
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .fence import Fence

# Global auto-increment counter for ClaimRecord IDs
_claim_id_counter = 0


def _next_claim_id() -> int:
    """Return the next auto-incrementing claim ID."""
    global _claim_id_counter
    _claim_id_counter += 1
    return _claim_id_counter


def reset_claim_ids() -> None:
    """Reset the global claim ID counter (useful for tests)."""
    global _claim_id_counter
    _claim_id_counter = 0


@dataclass
class ClaimRecord:
    """A recorded audit claim linking a document statement to source evidence.

    Fields are deliberately generic -- users define their own verdict
    taxonomy and source_type values.
    """

    claim: str
    """The factual claim being verified (natural language)."""

    claim_in_document: str
    """VERBATIM text from the document being audited. Must be an exact
    substring -- enforcement rejects paraphrases."""

    evidence: str
    """Search output that supports (or contradicts) the claim."""

    # Source provenance
    source_agent: str = ""
    """Which agent/system produced the source data."""

    source_tool: str = ""
    """Which tool call the evidence came from."""

    source_index: int = -1
    """Tool call index for precise location (0-based)."""

    raw_value: str = ""
    """Exact value as it appears in source data."""

    # Search coordinates
    search_file: str = ""
    """File where evidence was found."""

    search_line: int = -1
    """Line number in file."""

    # Classification
    verdict: str = ""
    """User-defined verdict (no built-in taxonomy)."""

    source_type: str = "standard"
    """Evidence type: 'standard', 'kb', 'web', 'computation', or user-defined."""

    # Metadata
    metadata: dict = field(default_factory=dict)
    """Extensible metadata for domain-specific fields."""

    id: int = field(default_factory=_next_claim_id)
    """Auto-incrementing claim ID."""

    timestamp: float = field(default_factory=time.time)
    """Creation timestamp."""

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return asdict(self)


def create_record_tool(
    fence: Fence,
    name: str = "record_claim",
    doc: str = "Record an audited claim with evidence.",
    extra_fields: list[str] | None = None,
    require_search: bool = True,
    require_claim_in_document: bool = True,
    skip_search_for: list[str] | None = None,
) -> Callable:
    """Create an enforcement-checked record tool that produces ClaimRecords.

    The returned callable:
    - Is decorated with ``@fence.enforce``
    - Accepts claim, claim_in_document, evidence + extra_fields as params
    - Returns a :class:`ClaimRecord`
    - Auto-appends to fence's claim list and JSONL output

    Args:
        fence: The Fence instance to attach enforcement to.
        name: Function name for the returned tool.
        doc: Docstring for the returned tool.
        extra_fields: Additional ClaimRecord field names to accept as
            parameters (e.g. ``["source_tool", "raw_value", "verdict"]``).
        require_search: If True (default), evidence must match search
            history (standard fence enforcement).
        require_claim_in_document: If True (default), claim_in_document
            must be found in the fence's document (if set).
        skip_search_for: List of verdict values that bypass the search
            requirement (e.g. ``["not-found", "derived"]``).

    Returns:
        A callable record tool with fence enforcement.
    """
    extra = extra_fields or []
    skip_verdicts = set(skip_search_for or [])

    # Build the base function signature dynamically
    # Core params are always present; extras are optional kwargs
    def _make_record(
        claim: str,
        claim_in_document: str,
        evidence: str,
        **kwargs: Any,
    ) -> ClaimRecord:
        record_kwargs: dict[str, Any] = {
            "claim": claim,
            "claim_in_document": claim_in_document,
            "evidence": evidence,
        }
        for f in extra:
            if f in kwargs:
                record_kwargs[f] = kwargs[f]
        record = ClaimRecord(**record_kwargs)

        # Append to fence claims list
        fence._claims.append(record)

        # Auto-append to JSONL if output path is set
        if fence._output_path is not None:
            _append_jsonl(fence._output_path, record)

        return record

    _make_record.__name__ = name
    _make_record.__qualname__ = name
    _make_record.__doc__ = doc

    if not require_search and not skip_verdicts:
        # No enforcement needed at all -- just track as submit
        @wraps(_make_record)
        def unguarded(
            claim: str = "",
            claim_in_document: str = "",
            evidence: str = "",
            **kwargs: Any,
        ) -> Any:
            # Still do claim_in_document check if required
            if require_claim_in_document and claim_in_document:
                err = fence._check_claim_in_document(claim_in_document)
                if err is not None:
                    return err

            return _make_record(
                claim=claim,
                claim_in_document=claim_in_document,
                evidence=evidence,
                **kwargs,
            )

        unguarded.__name__ = name
        unguarded.__qualname__ = name
        unguarded.__doc__ = doc
        unguarded._fence_role = "submit"  # type: ignore[attr-defined]
        fence._submit_fns.append(unguarded)
        return unguarded

    # With enforcement: we need a wrapper that conditionally skips
    # search enforcement for certain verdicts
    @wraps(_make_record)
    def guarded(
        claim: str = "",
        claim_in_document: str = "",
        evidence: str = "",
        **kwargs: Any,
    ) -> Any:
        verdict_val = kwargs.get("verdict", "")

        # Check claim_in_document if required
        if require_claim_in_document and claim_in_document:
            err = fence._check_claim_in_document(claim_in_document)
            if err is not None:
                return err

        # Skip search enforcement for certain verdicts
        if require_search and verdict_val not in skip_verdicts:
            # Check 1: search history must exist
            if not fence._collect_history():
                err = (
                    "No search calls recorded. You must call a search "
                    "tool first to find evidence before submitting."
                )
                fence._log_rejection(name, evidence, err)
                return f"ERROR: {err}"

            # Check 2: min evidence length
            if len(evidence.strip()) < fence._min_evidence_length:
                err = (
                    f"Evidence too short (got {len(evidence.strip())} chars, "
                    f"min {fence._min_evidence_length}). Paste actual search output."
                )
                fence._log_rejection(name, evidence, err)
                return f"ERROR: {err}"

            # Check 3: evidence must match search history
            ok, err = fence._verify_search_match(evidence)
            if not ok:
                fence._log_rejection(name, evidence, err)
                return f"ERROR: {err}"

        return _make_record(
            claim=claim,
            claim_in_document=claim_in_document,
            evidence=evidence,
            **kwargs,
        )

    guarded.__name__ = name
    guarded.__qualname__ = name
    guarded.__doc__ = doc
    guarded._fence_role = "submit"  # type: ignore[attr-defined]
    fence._submit_fns.append(guarded)
    return guarded


def _append_jsonl(path: str, record: ClaimRecord) -> None:
    """Append a single ClaimRecord as a JSON line to a file."""
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
