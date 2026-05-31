[← Back to README](../../README.md)

# Structured Claims

The core `@fence.track` / `@fence.enforce` decorators work with any function signature. For audit workflows that produce structured claim records — with findings, source provenance, and evidence chains — audit-fence provides a higher-level API.

### ClaimRecord

A dataclass that links a document statement to its source evidence:

```python
from audit_fence import ClaimRecord

record = ClaimRecord(
    claim="P/E ratio of 18.9x",                          # what's being verified
    claim_in_document="the stock trades at 18.9x P/E",   # verbatim text from audited document
    evidence="fundamental.json:18: pe_ratio: 18.923",    # search output that supports it
    source_tool="get_stock_info",                         # which tool produced the source data
    source_index=0,                                       # tool call index
    raw_value="18.923",                                   # exact value from source
    finding="found",                                      # agent-assigned (no built-in taxonomy)
    source_type="standard",                               # standard | kb | web | computation | custom
)
```

All fields except `claim`, `claim_in_document`, and `evidence` are optional. IDs auto-increment. Serialize with `record.to_dict()`.

### record_tool

Creates an enforcement-checked record tool — combines `@fence.enforce` with `ClaimRecord` creation and JSONL persistence:

```python
from audit_fence import Fence

fence = Fence()
fence.set_document(report_text)
fence.set_output("audit/citations.jsonl")   # auto-append each record

record = fence.record_tool(
    name="record_citation",
    extra_fields=["finding", "source_tool", "raw_value", "grep_file", "grep_line"],
)

# Usage: search first, then record
search("revenue")
result = record(
    claim="Revenue was $5.1B",
    claim_in_document="revenue of $5.1 billion",
    evidence="fundamental.json:42: totalRevenue: 5098000000",
    finding="found",               # known field -> ClaimRecord.finding
    source_tool="get_stock_info",   # known field -> ClaimRecord.source_tool
    raw_value="5098000000",         # known field -> ClaimRecord.raw_value
    grep_file="fundamental.json",   # unknown field -> ClaimRecord.metadata["grep_file"]
    grep_line=42,                   # unknown field -> ClaimRecord.metadata["grep_line"]
)
# result is a ClaimRecord instance
# also auto-appended to audit/citations.jsonl
```

`extra_fields` that match `ClaimRecord` attributes are set directly on the record. Unrecognized fields are routed to the record's `metadata` dict.

### Record enrichment

The `enrich` callback runs after a `ClaimRecord` is created but before it's persisted. Return the record to accept, or `None` to reject:

```python
def resolve_source(record: ClaimRecord) -> ClaimRecord | None:
    """Auto-resolve tool name from grep coordinates."""
    if record.search_file and record.search_line >= 0:
        tool, idx = my_trace_resolver(record.search_file, record.search_line)
        record.source_tool = tool
        record.source_index = idx
    else:
        return None  # reject — can't resolve source provenance
    return record

record = fence.record_tool(
    extra_fields=["search_file", "search_line"],
    enrich=resolve_source,
)
```

When `enrich` returns `None`, the record is not persisted. The rejection is logged and triggers `on_reject`.

### Lifecycle callbacks

```python
record = fence.record_tool(
    on_record=lambda r: print(f"Recorded #{r.id}: {r.finding} — {r.claim[:50]}"),
    on_reject=lambda tool, content, reason: log.warning(f"[{tool}] {reason}"),
)
```

`on_record` fires after successful creation, enrichment, and persistence. `on_reject` fires on any rejection. Both are optional.

### Evidence chain

In multi-stage pipelines, claims in later stages trace back to claims from earlier stages:

```python
from audit_fence import FenceGroup

group = FenceGroup()
r1 = group.create("r1_fundamental")
r2 = group.create("r2_specialist")

# R1 records a claim
rec_r1 = r1.record_tool(name="record_r1_claim", ...)
search_r1("pe_ratio")
r1_claim = rec_r1(claim="P/E of 18.9x", ...)

# R2 links back to R1 via enrich
def link_to_r1(record):
    best_match = find_nearest(r1.claims, record)
    if best_match:
        record.upstream_id = best_match.id
        record.upstream_fence = "r1_fundamental"
    return record

rec_r2 = r2.record_tool(name="record_r2_claim", enrich=link_to_r1, ...)

# Traverse the full chain
chain = group.trace_chain(r2_claim)
# [r2_claim, r1_claim] — from final claim back to source
```

`trace_chain()` is cycle-safe and works with arbitrarily long pipelines.
