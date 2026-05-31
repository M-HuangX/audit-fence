[← Back to README](../../README.md)

# Integration

For the simplest path, use `fence.audit()` — see [Quick Start](../../README.md#quick-start). The sections below cover manual integration for custom agents and pipelines.

The core enforcement engine has no required dependencies. The pre-built audit agent (`fence.audit()`) uses LangGraph — install the model provider you prefer. For custom integration, audit-fence provides `wrap_tools()` for existing tool lists and decorators for new projects.

### Configuration

```python
fence = Fence(
    name="audit_r1",          # Optional identifier (for multi-agent, logging)
    min_evidence_length=20,   # Minimum chars for evidence (default: 20)
    history_window=20,        # How many recent searches to check against (default: 20)
    history_limit=100,        # Max total records kept in memory (default: unlimited)
    context={"ticker": "AAPL", "phase": "audit"},  # Attached to every rejection log entry
    track_all=False,          # When True, wrap_tools() tracks ALL tools (see Soft Enforcement)
)
```

#### Setup methods

After constructing a Fence, configure what it audits:

```python
fence.set_document(open("report.md").read())  # the report being audited
fence.set_source("./source_data/")            # where to search (creates a RipgrepBackend)
fence.set_output("audit/citations.jsonl")     # auto-persist every ClaimRecord
```

`set_document()` enables automatic `claim_in_document` verification. `set_source()` creates a tracked search tool (available as `fence.search`). `set_output()` auto-appends records to JSONL. All three accept callables for dynamic content — see individual sections below.

#### Custom parameter names

```python
@fence.enforce(evidence_param="grep_output")
def submit(claim: str, grep_output: str) -> dict: ...
```

#### Document enforcement

`set_document()` tells the fence which document is being audited. Any function with a `claim_in_document` parameter is automatically checked — the value must be a verbatim substring of the document (after markdown normalization):

```python
fence = Fence()
fence.set_document("The company reported revenue of $5.1 billion in FY2025.")

@fence.track
def search(query: str) -> str: ...

@fence.enforce
def record(claim: str, claim_in_document: str, evidence: str) -> dict:
    return {"claim": claim, "status": "ok"}

search("revenue")
record(
    claim="Revenue was $5.1B",
    claim_in_document="revenue of $5.1 billion",    # found in document
    evidence="<matching search output>",
)
record(
    claim="Revenue was $5.1B",
    claim_in_document="revenue of $8.2 billion",    # not in document
    evidence="<matching search output>",
)
# => ERROR: 'revenue of $8.2 billion' not found in the audited document
```

Callable documents are supported for dynamic content:

```python
fence.set_document(lambda: open("report.md").read())
```

#### Rejection logging

Every rejected submission is recorded:

```python
fence.rejections
# [{"tool": "record_citation", "content": "...", "reason": "...", "timestamp": ..., "context": {...}}]

fence.save_log("enforcement_log.jsonl")
```

### wrap_tools() — for existing codebases

If you already have tools defined, `wrap_tools()` adds enforcement without modifying any function definitions:

```python
from audit_fence import Fence

fence = Fence()

# Your existing tools — no changes needed
existing_tools = [search_web, get_financials, analyze_data, write_report]

# One call: classify by name pattern, get back enforced tools
protected_tools = fence.wrap_tools(
    existing_tools,
    search=["search_*", "get_*"],     # these get tracked
    submit=["write_*"],               # these get enforced
)

agent = create_react_agent(llm, protected_tools)
```

Tools matching `search` patterns are tracked. Tools matching `submit` patterns are enforced. Unmatched tools pass through unchanged. You can also match by function reference:

```python
protected_tools = fence.wrap_tools(
    existing_tools,
    search=[search_web, get_financials],
    submit=[write_report],
)
```

### Decorators — for new projects

When building tools from scratch, decorators express intent at the definition site:

```python
from langchain_core.tools import tool
from audit_fence import Fence

fence = Fence()

@tool
@fence.track
def search_evidence(query: str, path: str = "traces/") -> str:
    """Search trace files for evidence."""
    return subprocess.run(["rg", "-n", query, path], capture_output=True, text=True).stdout

@tool
@fence.enforce
def record_citation(claim: str, evidence: str) -> str:
    """Record a citation. evidence must match a recent search result."""
    return json.dumps({"claim": claim, "status": "recorded"})

agent = create_react_agent(llm, [search_evidence, record_citation])
```

When the agent submits invalid evidence, the tool returns `"ERROR: ..."`. The ReAct loop sees this as a failed tool call and retries — naturally driving the agent toward valid, search-backed evidence.

### OpenAI / Anthropic / Custom

The decorated functions are regular callables. Use them in your tool dispatch however your framework requires:

```python
# OpenAI function calling — use your framework's schema helper
tools_schema = [describe_function(fn) for fn in fence.tools]

# Any custom framework
for fn in fence.tools:
    register_tool(fn.__name__, fn, fn.__doc__)
```

### DeepSeek (reasoning models)

DeepSeek's reasoning models (R1, V3, V4 Flash) return a `reasoning_content` field that must be passed back in all subsequent API requests. LangChain's `ChatOpenAI` silently drops this field during serialization, causing 400 errors on multi-turn conversations. This is an [open upstream bug](https://github.com/langchain-ai/langchain/issues/34166) (6+ months unresolved as of May 2026).

audit-fence provides a drop-in replacement:

```python
from audit_fence.compat import ChatOpenAIDeepSeek

llm = ChatOpenAIDeepSeek(
    model="deepseek-reasoner",
    api_key="...",
    base_url="https://api.deepseek.com",
    extra_body={"thinking": {"type": "enabled"}},
)

# Works with fence.audit() and create_react_agent()
result = await fence.audit(llm=llm)
```

`ChatOpenAIDeepSeek` fixes two classes of 400 errors: missing `reasoning_content` round-trip and unmatched `tool_call_id`s from invalid tool call arguments. When LangChain merges an upstream fix, replace with plain `ChatOpenAI` — the interface is identical.
