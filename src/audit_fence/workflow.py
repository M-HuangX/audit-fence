"""Workflow layer: ClaimRecord, record tool factory, JSONL persistence."""

from __future__ import annotations

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

    # Evidence chain (optional)
    upstream_id: int = -1
    """ID of an upstream claim this record traces back to (-1 = none)."""

    upstream_fence: str = ""
    """Name of the fence that holds the upstream claim."""

    id: int = field(default_factory=_next_claim_id)
    """Auto-incrementing claim ID."""

    timestamp: float = field(default_factory=time.time)
    """Creation timestamp."""

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return asdict(self)


_CLAIM_FIELDS = set(ClaimRecord.__dataclass_fields__.keys())


def create_record_tool(
    fence: Fence,
    name: str = "record_claim",
    doc: str = "Record an audited claim with evidence.",
    extra_fields: list[str] | None = None,
    require_search: bool = True,
    require_claim_in_document: bool = True,
    skip_enforcement: dict[str, Any] | Callable[[dict], bool] | None = None,
    enrich: Callable[[ClaimRecord], ClaimRecord | None] | None = None,
    on_record: Callable[[ClaimRecord], None] | None = None,
    on_reject: Callable[[str, str, str], None] | None = None,
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
        extra_fields: Additional field names to accept as parameters.
            Fields that match :class:`ClaimRecord` attributes (e.g.
            ``"verdict"``, ``"source_tool"``) are set directly.
            Unrecognized fields are stored in the record's ``metadata``
            dict, so domain-specific data is preserved without schema
            changes.
        require_search: If True (default), evidence must match search
            history (standard fence enforcement).
        require_claim_in_document: If True (default), claim_in_document
            must be found in the fence's document (if set).
        skip_enforcement: Conditions under which search enforcement is
            skipped.  Two forms are accepted:

            * **dict** — ``{field_name: [values]}``.  Search enforcement
              is skipped when *any* field in the record kwargs matches one
              of the listed values.  Example::

                  skip_enforcement={"verdict": ["not-found", "derived"]}
                  skip_enforcement={"source_type": ["kb", "web", "derived"]}

            * **callable** — ``fn(kwargs) -> bool``.  Receives the full
              keyword arguments dict and returns ``True`` to skip.
              Example::

                  skip_enforcement=lambda kw: kw.get("confidence", 0) > 0.9

        enrich: Optional callback invoked after a :class:`ClaimRecord` is
            created but *before* it is persisted.  Use this to resolve
            source coordinates, link upstream claims, or compute derived
            fields.  Return the (possibly modified) record, or ``None``
            to reject::

                def resolve(r: ClaimRecord) -> ClaimRecord | None:
                    r.source_tool = lookup(r.search_file, r.search_line)
                    if not r.source_tool:
                        return None  # reject — can't resolve source
                    return r

        on_record: Optional callback fired after a record is successfully
            created, enriched, and persisted.  Receives the final
            :class:`ClaimRecord`.  Use for real-time event emission,
            logging, or UI updates::

                on_record=lambda r: emit_sse("claim_recorded", r.to_dict())

        on_reject: Optional callback fired when a record is rejected
            (enforcement failure, document mismatch, or enrich rejection).
            Receives ``(tool_name, content, reason)``::

                on_reject=lambda t, c, r: log.warning(f"[{t}] {r}")

    Returns:
        A callable record tool with fence enforcement.
    """
    extra = extra_fields or []
    _enrich = enrich
    _on_record = on_record
    _on_reject = on_reject

    # Build the skip predicate from the user-provided spec.
    if skip_enforcement is None:
        _should_skip: Callable[[dict], bool] = lambda kw: False  # noqa: E731
    elif callable(skip_enforcement) and not isinstance(skip_enforcement, dict):
        _should_skip = skip_enforcement
    else:
        # dict form: {field: [values]}
        _skip_map: dict[str, Any] = skip_enforcement  # type: ignore[assignment]

        def _should_skip(kw: dict) -> bool:
            for field_name, values in _skip_map.items():
                if kw.get(field_name, "") in values:
                    return True
            return False

    # Build the base function signature dynamically
    # Core params are always present; extras are optional kwargs
    def _make_record(
        claim: str,
        claim_in_document: str,
        evidence: str,
        **kwargs: Any,
    ) -> Any:
        record_kwargs: dict[str, Any] = {
            "claim": claim,
            "claim_in_document": claim_in_document,
            "evidence": evidence,
        }
        extra_metadata: dict[str, Any] = {}
        for f in extra:
            if f in kwargs:
                if f in _CLAIM_FIELDS:
                    record_kwargs[f] = kwargs[f]
                else:
                    extra_metadata[f] = kwargs[f]
        if extra_metadata:
            existing = record_kwargs.get("metadata", {})
            record_kwargs["metadata"] = {**existing, **extra_metadata}
        record = ClaimRecord(**record_kwargs)

        # Enrichment hook: resolve source, link upstream, etc.
        # Returning None signals rejection.
        if _enrich is not None:
            record = _enrich(record)
            if record is None:
                err = "Record rejected by enrich callback."
                fence._log_rejection(name, evidence, err)
                if _on_reject is not None:
                    _on_reject(name, evidence, err)
                return f"ERROR: {err}"

        # Append to fence claims list
        fence._claims.append(record)

        # Auto-append to JSONL if output path is set
        if fence._output_path is not None:
            _append_jsonl(fence._output_path, record)

        # Lifecycle callback
        if _on_record is not None:
            _on_record(record)

        return record

    _make_record.__name__ = name
    _make_record.__qualname__ = name
    _make_record.__doc__ = doc

    if not require_search and skip_enforcement is None:
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
                    if _on_reject is not None:
                        _on_reject(name, claim_in_document, err)
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

    # With enforcement: wrapper that conditionally skips via _should_skip
    @wraps(_make_record)
    def guarded(
        claim: str = "",
        claim_in_document: str = "",
        evidence: str = "",
        **kwargs: Any,
    ) -> Any:
        # Check claim_in_document if required
        if require_claim_in_document and claim_in_document:
            err = fence._check_claim_in_document(claim_in_document)
            if err is not None:
                if _on_reject is not None:
                    _on_reject(name, claim_in_document, err)
                return err

        # Skip search enforcement when the predicate says so
        if require_search and not _should_skip(kwargs):
            # Check 1: search history must exist
            if not fence._collect_history():
                err = (
                    "No search calls recorded. You must call a search "
                    "tool first to find evidence before submitting."
                )
                fence._log_rejection(name, evidence, err)
                if _on_reject is not None:
                    _on_reject(name, evidence, err)
                return f"ERROR: {err}"

            # Check 2: min evidence length
            if len(evidence.strip()) < fence._min_evidence_length:
                err = (
                    f"Evidence too short (got {len(evidence.strip())} chars, "
                    f"min {fence._min_evidence_length}). Paste actual search output."
                )
                fence._log_rejection(name, evidence, err)
                if _on_reject is not None:
                    _on_reject(name, evidence, err)
                return f"ERROR: {err}"

            # Check 3: evidence must match search history
            ok, err = fence._verify_search_match(evidence)
            if not ok:
                fence._log_rejection(name, evidence, err)
                if _on_reject is not None:
                    _on_reject(name, evidence, err)
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
