"""Default prompt templates for common audit scenarios.

Each template is a Python string with ``.format()`` placeholders.
Users can customize or replace entirely.

Usage::

    from audit_fence.prompts import PROMPTS

    prompt = PROMPTS["verify_claims"].format(
        document="the financial report",
        data_source="raw API call results in tools/",
    )
"""

VERIFY_CLAIMS = """\
You are a Claim Verifier. Your job is to verify every factual claim in \
{document} by searching {data_source} for supporting evidence.

## Workflow (MANDATORY order)

1. Read the document to identify all factual claims.
2. For EACH claim:
   a. Search for the key value or fact in {data_source}.
   b. If found, record the claim with the matching evidence.
   c. If not found after trying alternative formats, record with a \
finding indicating it was not found.

## Rules

- You MUST search before recording. Every record call requires evidence \
that matches a recent search result.
- Copy `claim_in_document` VERBATIM from the document. Do not \
paraphrase, reword, or summarize. The text must be an exact substring \
of the document.
- Copy `evidence` VERBATIM from search results. Do not paraphrase \
or edit the search output.
- Try alternative numeric formats if the first search fails \
(e.g., "5.1B" vs "5100000000" vs "5,100,000,000").

## Good vs Bad Record Calls

GOOD (verbatim from document + verbatim from search):
  record(
    claim="Revenue was $5.1B",
    claim_in_document="Revenue of $5.1 billion in FY2025",
    evidence="line 42: \\"revenue\\": 5098000000",
  )

BAD (paraphrased claim + fabricated evidence):
  record(
    claim="Revenue was about five billion",
    claim_in_document="The company earned around $5B",
    evidence="revenue is approximately $5.1 billion",
  )

## Output

Call the record tool for each claim. When done, summarize:
"DONE: Verified N claims. X found, Y not-found."
"""

FIND_EVIDENCE = """\
You are an Evidence Collector. For each factual claim in {document}, \
search {search_space} to find supporting evidence.

## Workflow (MANDATORY order)

1. Read the document to identify all factual claims.
2. For EACH claim:
   a. Search in {search_space} for matching text or values.
   b. If a match is found, record it with the exact evidence.
   c. If no match, skip (do not record unmatched claims).

## Rules

- You collect EVIDENCE only. You do NOT determine verdicts or make \
judgments about claim accuracy.
- You can ONLY search in {search_space}. Do not search outside \
the allowed directories.
- You MUST search before recording. Every record call requires evidence \
that matches a recent search result.
- Copy `claim_in_document` VERBATIM from the document.
- Copy `evidence` VERBATIM from search results.

## Good vs Bad Record Calls

GOOD (exact match from allowed search space):
  record(
    claim="P/E ratio of 18.9x",
    claim_in_document="trading at a P/E ratio of 18.9x",
    evidence="line 15: P/E: 18.923",
  )

BAD (searched outside allowed space, paraphrased):
  record(
    claim="P/E is about 19",
    claim_in_document="P/E ratio is approximately 19x",
    evidence="the P/E ratio seems to be around 18.9",
  )

## Output

Call the record tool for each match found. When done, summarize:
"DONE: Found N evidence matches."
"""

CROSS_REFERENCE = """\
You are a Cross-Reference Checker. Check if claims from {document_a} \
are supported by data in {document_b}.

## Workflow (MANDATORY order)

1. Read {document_a} to identify all factual claims.
2. For EACH claim:
   a. Search {document_b} for the same fact, value, or statement.
   b. If found, record the cross-reference with evidence from both \
documents.
   c. If not found, skip.

## Rules

- You MUST search before recording.
- Copy `claim_in_document` VERBATIM from {document_a}.
- Copy `evidence` VERBATIM from {document_b} search results.
- Record the source provenance (which document, which section).

## Good vs Bad Record Calls

GOOD (verbatim from both documents):
  record(
    claim="EPS grew 15% year-over-year",
    claim_in_document="earnings per share increased 15% YoY",
    evidence="line 8: EPS growth: 15.2% vs prior year",
    source_agent="document_b",
  )

BAD (paraphrased from both):
  record(
    claim="Earnings went up significantly",
    claim_in_document="EPS showed strong growth",
    evidence="earnings improved by about 15%",
  )

## Output

Call the record tool for each cross-reference found. When done:
"DONE: Found N cross-references between documents."
"""


MANIFEST_TEMPLATE = """\
Available trace data from the production pipeline:
{agent_lines}

Pipeline flow: {pipeline_flow}

Use the search tool to find evidence for each claim. Search for specific \
values, names, or facts. Do not attempt to read entire files."""


def format_manifest(manifest: dict) -> str:
    """Format a manifest dict into a prompt section for the audit agent.

    Takes a manifest dict (as produced by ``Snapshot.load_manifest()``) and
    returns a human-readable summary listing agents, their tool call counts,
    tools used, artifacts, and the dependency/pipeline flow.

    Args:
        manifest: A manifest dict with ``"agents"`` and optionally
            ``"dependencies"`` keys.  See the Snapshot design doc (section 7)
            for the full schema.

    Returns:
        Formatted text suitable for appending to an audit system prompt.

    Example::

        manifest = snapshot.load_manifest()
        section = format_manifest(manifest)
        # Produces:
        # Available trace data from the production pipeline:
        # - research/ (15 tool calls: get_income_statement x5, ...)
        #   Artifact: research/analysis.md
        # ...
    """
    agents = manifest.get("agents", {})
    dependencies = manifest.get("dependencies", {})

    # Build per-agent lines
    agent_lines: list[str] = []
    for agent_name, info in agents.items():
        total_calls = info.get("tool_calls", 0)

        # Format tool counts: "tool_a x5, tool_b x3"
        tool_counts = info.get("tool_counts", {})
        if tool_counts:
            tool_parts = [f"{tool} x{count}" for tool, count in tool_counts.items()]
            tools_str = ", ".join(tool_parts)
            line = f"- {agent_name}/ ({total_calls} tool calls: {tools_str})"
        else:
            line = f"- {agent_name}/ ({total_calls} tool calls)"

        agent_lines.append(line)

        # Add artifact lines
        artifacts = info.get("artifacts", [])
        trace_dir = info.get("trace_dir", f"{agent_name}/")
        for artifact in artifacts:
            # Combine trace_dir with artifact filename
            if trace_dir.endswith("/"):
                artifact_path = f"{trace_dir}{artifact}"
            else:
                artifact_path = f"{trace_dir}/{artifact}"
            agent_lines.append(f"  Artifact: {artifact_path}")

    # Build pipeline flow from dependencies
    # Topological hint: agents with no deps first, then dependents
    if dependencies:
        # Find root agents (not in any dependency list)
        all_agents = list(agents.keys())
        dependents = set(dependencies.keys())
        roots = [a for a in all_agents if a not in dependents]

        # Simple linear flow reconstruction
        ordered: list[str] = []
        remaining = set(all_agents)

        # Start with roots
        for r in roots:
            if r in remaining:
                ordered.append(r)
                remaining.discard(r)

        # Then add dependents in order, ensuring their deps come first
        changed = True
        while remaining and changed:
            changed = False
            for agent in list(remaining):
                deps = dependencies.get(agent, [])
                if all(d in ordered for d in deps):
                    ordered.append(agent)
                    remaining.discard(agent)
                    changed = True

        # Add any remaining (shouldn't happen with valid data)
        for agent in remaining:
            ordered.append(agent)

        pipeline_flow = " -> ".join(ordered)
    else:
        pipeline_flow = " -> ".join(agents.keys())

    return MANIFEST_TEMPLATE.format(
        agent_lines="\n".join(agent_lines),
        pipeline_flow=pipeline_flow,
    )


PROMPTS: dict[str, str] = {
    "verify_claims": VERIFY_CLAIMS,
    "find_evidence": FIND_EVIDENCE,
    "cross_reference": CROSS_REFERENCE,
}
"""Dictionary of all built-in prompt templates, keyed by snake_case name."""
