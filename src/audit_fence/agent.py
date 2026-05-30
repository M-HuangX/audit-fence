"""Pre-built audit agent — runs a complete audit with one method call.

Requires ``langgraph`` and a LangChain-compatible chat model::

    pip install langgraph langchain-openai
    # or: pip install langgraph langchain-anthropic

Usage::

    from audit_fence import Fence
    from langchain_openai import ChatOpenAI

    fence = Fence()
    fence.set_document(open("report.md").read())
    fence.set_source("./source_data/")
    fence.set_output("audit/citations.jsonl")

    result = await fence.audit(llm=ChatOpenAI(model="gpt-4o"))
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .fence import Fence

logger = logging.getLogger(__name__)


@dataclass
class AuditResult:
    """Result of a complete audit run."""

    claims: list = field(default_factory=list)
    """All ClaimRecords produced during the audit."""

    rejections: list = field(default_factory=list)
    """All enforcement rejections logged during the audit."""

    summary: dict = field(default_factory=dict)
    """Aggregated statistics: total, found, not_found counts."""

    messages: list = field(default_factory=list)
    """Raw agent message history (for debugging)."""


def _make_langchain_tools(fence: Fence, extra_fields: list[str]) -> list:
    """Create LangChain-compatible tools from a configured Fence.

    Returns a list of LangChain ``StructuredTool`` objects wrapping
    the fence's search and record functions.
    """
    try:
        from langchain_core.tools import StructuredTool
    except ImportError:
        raise ImportError(
            "audit() requires langchain-core. "
            "Install with: pip install langchain-core"
        )

    from .workflow import create_record_tool

    # Search tool
    search_fn = fence.search  # raises RuntimeError if set_source() not called

    def _search(query: str) -> str:
        """Search source data for evidence matching a claim."""
        return search_fn(query)

    search_tool = StructuredTool.from_function(
        func=_search,
        name="search",
        description=(
            "Search the source data directory for evidence. "
            "Call this BEFORE recording any claim. "
            "The query should target specific values, names, or facts."
        ),
    )

    # Record tool
    record_fn = create_record_tool(fence, extra_fields=extra_fields)

    # Build the record wrapper with explicit parameters for the agent
    def _record(
        claim: str = "",
        claim_in_document: str = "",
        evidence: str = "",
        finding: str = "",
        source_tool: str = "",
        raw_value: str = "",
    ) -> str:
        """Record an audited claim with evidence from a recent search.

        Args:
            claim: What is being verified (natural language).
            claim_in_document: VERBATIM text copied from the report.
            evidence: VERBATIM text copied from search output.
            finding: 'found' or 'not-found'.
            source_tool: Which tool produced the source data.
            raw_value: Exact value from source data.
        """
        kwargs: dict[str, Any] = {}
        if finding:
            kwargs["finding"] = finding
        if source_tool:
            kwargs["source_tool"] = source_tool
        if raw_value:
            kwargs["raw_value"] = raw_value
        result = record_fn(
            claim=claim,
            claim_in_document=claim_in_document,
            evidence=evidence,
            **kwargs,
        )
        if isinstance(result, str):
            return result  # ERROR string
        return f"Recorded: [{result.finding or 'found'}] {result.claim[:80]}"

    record_tool = StructuredTool.from_function(
        func=_record,
        name="record",
        description=(
            "Record a verified claim. You MUST call search() first. "
            "evidence must be verbatim text from a recent search result. "
            "claim_in_document must be verbatim text from the report."
        ),
    )

    return [search_tool, record_tool]


async def run_audit(
    fence: Fence,
    llm: Any,
    *,
    max_rounds: int = 200,
    extra_fields: list[str] | None = None,
    prompt_template: str | None = None,
    timeout: int = 900,
) -> AuditResult:
    """Run a complete audit using a LangGraph ReAct agent.

    Args:
        fence: A configured Fence (set_document, set_source, set_output).
        llm: Any LangChain-compatible chat model (ChatOpenAI, ChatAnthropic, etc.).
        max_rounds: Maximum agent reasoning rounds (default 200).
        extra_fields: Extra ClaimRecord fields the agent can fill.
        prompt_template: Override the default VERIFY_CLAIMS prompt.
        timeout: Timeout in seconds (default 900 = 15 minutes).

    Returns:
        AuditResult with claims, rejections, and summary.
    """
    try:
        from langchain_core.messages import HumanMessage
        from langgraph.prebuilt import create_react_agent
    except ImportError:
        raise ImportError(
            "audit() requires langgraph. "
            "Install with: pip install langgraph langchain-core"
        )

    from .prompts import VERIFY_CLAIMS

    # Validate fence is configured
    if fence._resolved_document is None:
        raise ValueError("No document set. Call fence.set_document() first.")
    if fence._search_fn is None:
        raise ValueError("No source set. Call fence.set_source() first.")

    # Build tools
    fields = extra_fields or ["finding", "source_tool", "raw_value"]
    tools = _make_langchain_tools(fence, fields)

    # System prompt
    template = prompt_template or VERIFY_CLAIMS
    system_prompt = template.format(
        document="the document provided below",
        data_source="the source data (use the search tool)",
    )

    # User message = the document to audit
    doc_text = fence._resolved_document
    user_message = (
        "Audit the following document. Verify every factual claim by "
        "searching the source data, then record each claim with its "
        "evidence.\n\n"
        "---\n\n"
        f"{doc_text}"
    )

    # Create agent
    agent = create_react_agent(llm, tools, prompt=system_prompt)
    recursion_limit = 4 * max_rounds + 1

    logger.info(
        "Starting audit: %d chars document, max_rounds=%d",
        len(doc_text), max_rounds,
    )

    # Run
    try:
        result = await asyncio.wait_for(
            agent.ainvoke(
                {"messages": [HumanMessage(content=user_message)]},
                config={"recursion_limit": recursion_limit},
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.error("Audit timed out after %ds", timeout)
        result = {"messages": []}
    except Exception:
        logger.error("Audit failed", exc_info=True)
        raise

    # Build summary
    claims = fence.claims
    finding_counts: dict[str, int] = {}
    for c in claims:
        f = getattr(c, "finding", "") or "unknown"
        finding_counts[f] = finding_counts.get(f, 0) + 1

    summary = {
        "total": len(claims),
        "rejections": len(fence.rejections),
        **finding_counts,
    }

    logger.info(
        "Audit complete: %d claims, %d rejections",
        len(claims), len(fence.rejections),
    )

    return AuditResult(
        claims=claims,
        rejections=fence.rejections,
        summary=summary,
        messages=result.get("messages", []),
    )
