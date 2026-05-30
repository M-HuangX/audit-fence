"""Multi-agent audit pipeline with tool call annotation.

Demonstrates how to build a Firn-style multi-stage audit:
- Round 1: Parallel specialist fidelity (verify specialist output → raw data)
- Round 2: Report evidence collection (verify report → specialist + source)
- Evidence chain traversal (R2 → R1 provenance)
- Tool call annotation pattern (grep_line → tool_name resolution)
- Lifecycle callbacks (on_record / on_reject)
- Enrich hook with rejection
"""

import json
import re

from audit_fence import (
    ClaimRecord,
    Fence,
    FenceGroup,
    SandboxedSearch,
    create_record_tool,
)

# ──────────────────────────────────────────────────────────────
# Simulated data (in production, these are real files on disk)
# ──────────────────────────────────────────────────────────────

SPECIALIST_OUTPUT = """\
Revenue reached $5.1 billion in FY2025, up 12% year-over-year.
Operating margin expanded to 32.1%, driven by cost reductions.
The trailing P/E ratio stands at 18.9x.
Management guided for 15-20% revenue growth in FY2026.
"""

REPORT = """\
The company reported revenue of $5.1 billion in FY2025, representing
12% year-over-year growth. Operating margin expanded to 32.1%.
The stock currently trades at a trailing P/E ratio of 18.9x, below
the sector median of 22.4x. Management guided for 15-20% revenue
growth in FY2026, though this target appears ambitious.
"""

# Simulated tool call data (JSON format, line numbers matter)
TOOL_CALLS_JSON = """\
{
  "agent": "fundamental",
  "tool_calls": [
    {
      "tool_name": "get_income_statement",
      "input": {"ticker": "ACME", "period": "annual"},
      "output": {"totalRevenue": 5098000000, "revenueGrowth": 0.122},
      "duration": 1.2
    },
    {
      "tool_name": "get_financial_metrics",
      "input": {"ticker": "ACME"},
      "output": {"operatingMargin": 0.321, "grossMargin": 0.681},
      "duration": 0.8
    },
    {
      "tool_name": "get_stock_info",
      "input": {"ticker": "ACME"},
      "output": {"trailingPE": 18.923, "forwardPE": 15.2},
      "duration": 0.5
    }
  ]
}"""


# ──────────────────────────────────────────────────────────────
# Tool call annotation pattern (reusable utility)
# ──────────────────────────────────────────────────────────────

def build_tool_call_map(tool_calls_text: str) -> dict[str, list[tuple[str, int, int]]]:
    """Build a line-number → tool-call lookup table from JSON.

    Returns {file_path: [(tool_name, index, start_line), ...]}.
    Assumes ``json.dumps(indent=2)`` formatting with fixed line counts
    per entry.  Adjust ``lines_per_entry`` for your format.

    This is the pattern Firn uses to annotate grep results with
    ``[@ tool_call #N: tool_name]`` so the LLM agent knows which
    tool call produced each matching line.
    """
    data = json.loads(tool_calls_text)
    calls = data.get("tool_calls", [])

    # With json.dumps(indent=2), each tool call entry occupies ~8 lines.
    # The array starts at line 4 (0-indexed) after the header.
    header_lines = 4  # lines before first tool call entry
    lines_per_entry = 8  # lines per tool call (adjust for your format)

    entries = []
    for i, tc in enumerate(calls):
        start = header_lines + i * lines_per_entry
        entries.append((tc.get("tool_name", ""), i, start))

    return {"tools/fundamental_tool_calls.json": entries}


def annotate_grep_line(
    line: str,
    tc_map: dict[str, list[tuple[str, int, int]]],
) -> str:
    """Annotate a grep output line with tool call info.

    Transforms:
      tools/data.json:10: "trailingPE": 18.923
    Into:
      tools/data.json:10: "trailingPE": 18.923  [@ tool_call #2: get_stock_info]
    """
    m = re.match(r"([^:]+):(\d+)[:-]", line)
    if not m:
        return line
    file_path = m.group(1)
    line_num = int(m.group(2))

    entries = tc_map.get(file_path, [])
    for tool_name, idx, start_line in entries:
        if start_line <= line_num <= start_line + 7:
            return f"{line}  [@ tool_call #{idx}: {tool_name}]"
    return line


# ──────────────────────────────────────────────────────────────
# Simulated search backend
# ──────────────────────────────────────────────────────────────

# Build tool call map for annotation
TC_MAP = build_tool_call_map(TOOL_CALLS_JSON)


def grep_backend(pattern: str, path: str = "", **kwargs) -> str:
    """Simulated grep backend (in production, use RipgrepBackend)."""
    data = {
        "5098|5.1": 'tools/fundamental_tool_calls.json:8: "totalRevenue": 5098000000',
        "0.122|12": 'tools/fundamental_tool_calls.json:8: "revenueGrowth": 0.122',
        "0.321|32.1": 'tools/fundamental_tool_calls.json:14: "operatingMargin": 0.321',
        "18.9|18.923": 'tools/fundamental_tool_calls.json:20: "trailingPE": 18.923',
    }
    raw = data.get(pattern, f"No matches for '{pattern}' in {path}")
    # Annotate with tool call info
    lines = raw.split("\n")
    annotated = [annotate_grep_line(line, TC_MAP) for line in lines]
    return "\n".join(annotated)


def specialist_grep(pattern: str, path: str = "", **kwargs) -> str:
    """Simulated grep in specialist outputs."""
    data = {
        "5.1 billion": 'trace/specialist_outputs/fundamental_output.md:1: Revenue reached $5.1 billion in FY2025, up 12% year-over-year.',
        "32.1%": 'trace/specialist_outputs/fundamental_output.md:2: Operating margin expanded to 32.1%, driven by cost reductions.',
        "18.9x": 'trace/specialist_outputs/fundamental_output.md:3: The trailing P/E ratio stands at 18.9x.',
    }
    return data.get(pattern, f"No matches for '{pattern}' in {path}")


# ──────────────────────────────────────────────────────────────
# Tool call resolution (enrich hook pattern)
# ──────────────────────────────────────────────────────────────

def resolve_tool_call_location(grep_file: str, grep_line: int) -> tuple[str, int]:
    """Resolve (tool_name, tool_call_index) from grep coordinates.

    This is the pattern for converting grep output coordinates back to
    the specific tool call that produced the data.  Adjust the formula
    for your JSON format.
    """
    tc_data = json.loads(TOOL_CALLS_JSON)
    calls = tc_data.get("tool_calls", [])

    # Formula for json.dumps(indent=2): index = (line - header) // lines_per_entry
    header_lines = 4
    lines_per_entry = 8
    index = (grep_line - header_lines) // lines_per_entry

    if 0 <= index < len(calls):
        return calls[index].get("tool_name", ""), index
    return "", -1


# ──────────────────────────────────────────────────────────────
# Demo execution
# ──────────────────────────────────────────────────────────────


def main():
    """Run the multi-agent audit demo."""

    # ── Round 1: Specialist Fidelity Audit ────────────────────

    print("=" * 60)
    print("ROUND 1: Specialist Fidelity Audit")
    print("=" * 60)

    group = FenceGroup()

    # R1 fence: verify specialist output against raw tool data
    r1 = group.create("r1_fundamental")
    r1.set_document(SPECIALIST_OUTPUT)
    r1.set_output("/tmp/audit_fence_r1_fundamental.jsonl")

    # Track search results
    search_r1 = r1.wrap_tool(grep_backend, role="search")

    # Enrich: resolve grep coordinates → tool call info.
    # Returns None to reject if resolution fails.
    def enrich_r1(record: ClaimRecord) -> ClaimRecord | None:
        file = record.metadata.get("grep_file", "")
        line = record.metadata.get("grep_line", -1)
        if record.finding in ("not-found", "derived"):
            return record  # no resolution needed
        tool_name, index = resolve_tool_call_location(file, line)
        if not tool_name:
            return None  # reject: can't resolve tool call
        record.source_tool = tool_name
        record.source_index = index
        return record

    # Record tool with:
    # - skip_enforcement for not-found/derived findings
    # - enrich hook for tool call resolution (can reject)
    # - on_record callback for real-time notification
    # - domain-specific extra fields routed to metadata
    record_r1 = create_record_tool(
        r1,
        name="record_specialist_claim",
        extra_fields=[
            "finding",       # → ClaimRecord.finding (known field)
            "raw_value",     # → ClaimRecord.raw_value (known field)
            "grep_file",     # → ClaimRecord.metadata["grep_file"] (unknown → metadata)
            "grep_line",     # → ClaimRecord.metadata["grep_line"] (unknown → metadata)
            "output_line",   # → ClaimRecord.metadata["output_line"] (unknown → metadata)
        ],
        skip_enforcement={"finding": ["not-found", "derived"]},
        enrich=enrich_r1,
        on_record=lambda r: print(f"  [R1] Recorded #{r.id}: [{r.finding}] {r.claim[:50]}"),
        on_reject=lambda t, c, reason: print(f"  [R1] REJECTED: {reason[:60]}"),
    )

    # Simulate R1 audit: verify each claim
    search_r1("5098|5.1", "tools/")
    record_r1(
        claim="Revenue reached $5.1 billion in FY2025",
        claim_in_document="Revenue reached $5.1 billion in FY2025",
        evidence='"totalRevenue": 5098000000  [@ tool_call #0: get_income_statement]',
        finding="found",
        raw_value="5098000000",
        grep_file="tools/fundamental_tool_calls.json",
        grep_line=8,
        output_line=1,
    )

    search_r1("0.321|32.1", "tools/")
    record_r1(
        claim="Operating margin expanded to 32.1%",
        claim_in_document="Operating margin expanded to 32.1%",
        evidence='"operatingMargin": 0.321  [@ tool_call #1: get_financial_metrics]',
        finding="found",
        raw_value="0.321",
        grep_file="tools/fundamental_tool_calls.json",
        grep_line=14,
        output_line=2,
    )

    search_r1("18.9|18.923", "tools/")
    record_r1(
        claim="The trailing P/E ratio stands at 18.9x",
        claim_in_document="The trailing P/E ratio stands at 18.9x",
        evidence='"trailingPE": 18.923  [@ tool_call #2: get_stock_info]',
        finding="found",
        raw_value="18.923",
        grep_file="tools/fundamental_tool_calls.json",
        grep_line=20,
        output_line=3,
    )

    # Not-found claim (skips search enforcement)
    record_r1(
        claim="Management guided for 15-20% revenue growth",
        claim_in_document="Management guided for 15-20% revenue growth in FY2026",
        evidence="",
        finding="not-found",
        raw_value="",
        grep_file="",
        grep_line=-1,
        output_line=4,
    )

    # ── Round 2a: Specialist Evidence Agent ───────────────────

    print()
    print("=" * 60)
    print("ROUND 2a: Specialist Evidence (report → specialist outputs)")
    print("=" * 60)

    r2a = group.create("r2a_specialist")
    r2a.set_document(REPORT)
    r2a.set_output("/tmp/audit_fence_r2a.jsonl")

    # Sandboxed search: can only search specialist outputs
    sandboxed_r2a = SandboxedSearch(
        backend=specialist_grep,
        allowed_dirs=["trace/specialist_outputs/"],
    )
    search_r2a = r2a.wrap_tool(sandboxed_r2a, role="search")

    # Enrich: cross-reference with R1 claims for provenance chain
    def enrich_r2a(record: ClaimRecord) -> ClaimRecord | None:
        specialist = record.metadata.get("specialist_agent", "")
        r1_fence_name = f"r1_{specialist}"
        r1_fence = group.get(r1_fence_name)
        if r1_fence is None:
            return record

        # Find nearest R1 claim (by text similarity — simplified here)
        for r1_claim in r1_fence.claims:
            # Simple word overlap check (production uses weighted scoring)
            r1_words = set(r1_claim.claim.lower().split())
            r2_words = set(record.claim.lower().split())
            overlap = len(r1_words & r2_words) / max(len(r1_words | r2_words), 1)
            if overlap > 0.4:
                record.upstream_id = r1_claim.id
                record.upstream_fence = r1_fence_name
                record.source_tool = r1_claim.source_tool
                record.source_index = r1_claim.source_index
                break

        return record

    record_r2a = create_record_tool(
        r2a,
        name="record_specialist_evidence",
        extra_fields=["specialist_agent", "specialist_excerpt", "source_type"],
        skip_enforcement={"source_type": ["derived", "kb", "web"]},
        enrich=enrich_r2a,
        on_record=lambda r: print(f"  [R2a] Recorded #{r.id}: {r.claim[:50]}"),
    )

    # Verify report claims against specialist outputs
    search_r2a("5.1 billion", "trace/specialist_outputs/")
    record_r2a(
        claim="Revenue of $5.1 billion",
        claim_in_document="revenue of $5.1 billion in FY2025",
        evidence="Revenue reached $5.1 billion in FY2025, up 12% year-over-year.",
        specialist_agent="fundamental",
        specialist_excerpt="Revenue reached $5.1 billion in FY2025",
        source_type="standard",
    )

    search_r2a("32.1%", "trace/specialist_outputs/")
    record_r2a(
        claim="Operating margin of 32.1%",
        claim_in_document="Operating margin expanded to 32.1%",
        evidence="Operating margin expanded to 32.1%, driven by cost reductions.",
        specialist_agent="fundamental",
        specialist_excerpt="Operating margin expanded to 32.1%",
        source_type="standard",
    )

    # Blocked: searching outside sandbox
    result = sandboxed_r2a("revenue", "tools/data.json")
    print(f"\n  Sandbox test: {result[:60]}...")

    # ── Evidence chain traversal ──────────────────────────────

    print()
    print("=" * 60)
    print("EVIDENCE CHAIN TRAVERSAL")
    print("=" * 60)

    # Trace a R2a claim back through R1 to the source tool call
    for claim in r2a.claims:
        chain = group.trace_chain(claim)
        print(f"\nClaim: {claim.claim}")
        for i, link in enumerate(chain):
            prefix = "  " * i + ("-> " if i > 0 else "")
            fence_name = link.upstream_fence or "(root)"
            tool_info = f"{link.source_tool}#{link.source_index}" if link.source_tool else "n/a"
            print(f"  {prefix}[{fence_name}] #{link.id}: {link.claim[:40]} | tool: {tool_info}")

    # ── Summary ───────────────────────────────────────────────

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"R1 claims: {len(group['r1_fundamental'].claims)}")
    print(f"R2a claims: {len(group['r2a_specialist'].claims)}")
    print(f"Total rejections: {len(group.all_rejections)}")

    for r in group.all_rejections:
        print(f"  [{r['tool']}] {r['reason'][:70]}")

    # Persistence: save and restore full topology
    state = group.snapshot()
    print(f"\nSnapshot: {len(state['fences'])} fences, "
          f"{len(state.get('links', {}))} link relationships")


if __name__ == "__main__":
    main()
