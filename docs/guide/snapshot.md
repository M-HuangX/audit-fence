[← Back to README](../../README.md)

# Snapshot — Production Trace Capture

In production, your agent system generates reports using its own tools — API calls, database queries, search results. **Snapshot** captures every tool call output as grep-friendly trace files, so the audit agent can search and verify them.

```
Production pipeline                     Audit
┌──────────────────────────┐            ┌───────────────────────────┐
│  Agent calls tools       │            │  fence.set_source(trace/) │
│  snap captures output    │──trace/──> │  fence.audit(llm=...)     │
│  snap.finalize()         │            │  → grep trace files       │
└──────────────────────────┘            └───────────────────────────┘
```

### LangGraph (zero-config)

```python
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
```

`snap.config()` injects a LangGraph callback handler — no tool modification needed. Every tool call is saved as a flat text file under `trace/research/calls/` plus a structured JSONL index.

### Trace file format

Each tool call produces a grep-friendly text file:

```
trace/
├── research/
│   ├── calls/
│   │   ├── 0000_get_financials.txt    # flat text, one file per call
│   │   ├── 0001_search_filings.txt
│   │   └── 0002_analyze_data.txt
│   └── tool_calls.jsonl               # structured index
└── manifest.json                       # agent topology + tool stats
```

The flat text files are formatted for grep: `dict` → `key: value` lines, `list[dict]` → ASCII table, `str` → as-is. Output is capped at 10MB per file (configurable).

### Manifest-guided audit

`snap.finalize()` writes `manifest.json` with agent topology, tool statistics, and trace paths. Pass it to `fence.audit()` to inject this context into the audit agent's system prompt:

```python
audit = await fence.audit(llm=llm, manifest=snap.load_manifest())
```

### Multi-agent capture

```python
snap = Snapshot("trace/")

# Capture each agent separately
result_a = await agent_a.ainvoke(input, config=snap.config(agent="fundamental"))
result_b = await agent_b.ainvoke(input, config=snap.config(agent="technical"))

# Declare dependency graph
snap.declare_dependency("synthesizer", upstream=["fundamental", "technical"])
result_c = await agent_c.ainvoke(input, config=snap.config(agent="synthesizer"))

snap.finalize()

# Build audit topology from manifest
group = FenceGroup.from_snapshot_manifest(
    snap.load_manifest(),
    document=open("report.md").read(),
    trace_dir="trace/",
)
```

`FenceGroup.from_snapshot_manifest()` creates per-agent fences with links matching the declared dependencies.

### Alternative integration (non-LangGraph)

```python
# Decorator
@snap.capture(agent="research")
def get_stock_price(ticker: str) -> dict:
    return api.get(f"/stock/{ticker}")

# Wrap existing tools
wrapped = snap.wrap([tool_a, tool_b], agent="research")
```

### Built-in redaction

API keys matching common patterns (`sk-*`, `AIza*`, `tvly-*`, `Bearer *`) are automatically redacted. Add custom redaction:

```python
def my_redact(tool_name, input_data, output_data):
    # scrub PII, internal URLs, etc.
    return clean_input, clean_output

snap = Snapshot("trace/", redact=my_redact)
```

Set `redact=False` to disable all redaction.

### Context manager

```python
with Snapshot("trace/") as snap:
    result = await agent.ainvoke(input, config=snap.config(agent="research"))
# finalize() called automatically; marks incomplete on exception
```
