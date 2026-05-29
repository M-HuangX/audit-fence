# audit-fence

**Programmatic enforcement for LLM agent evidence. Search-verified citations. Zero hallucinated evidence.**

<p>
  <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/dependencies-zero-brightgreen" alt="Zero Dependencies"/>
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"/>
</p>

---

## The Problem

AI agents are increasingly used to generate reports in finance, legal, medical, and other regulated industries. These reports contain factual claims — numbers, dates, metrics — and stakeholders need to know where each claim came from. The standard approach: let the LLM cite its own sources as it writes.

The problem is that **citations are self-declared**. The same model that might hallucinate "revenue grew 26%" is also the one claiming "Source: get_income_statement". When the context window is long and packed with data, the probability of hallucinated citations rises sharply — and you have no way to tell which ones are real.

You can't manually verify every report. The intuitive solution is to use a second AI — an **audit agent** — to independently verify the first agent's citations against raw data. But the audit agent is itself an LLM. It can fabricate evidence just as easily: "I checked line 42 and it says $5.1B" — did it actually check, or is it confabulating?

## The Solution

**audit-fence** enforces a simple rule at the code level: **no search, no evidence**.

```
Without audit-fence:
  Agent → "I found X in the data"  →  Record citation  →  ✓ accepted (unverified)

With audit-fence:
  Agent → search("X")              →  Result recorded to history
  Agent → submit(evidence="X...")  →  Verify evidence ∈ search history  →  ✓ or REJECTED
```

The agent's evidence submission tool **programmatically validates** that the submitted evidence is a character-level substring of an actual search result. Not semantic similarity — exact text match. If the agent didn't search for it, or if it paraphrases/fabricates the result, the submission is rejected and the rejection is logged.

This is not an arbitrary constraint. It exploits a known property of transformer attention: when a context window is packed with a report, source data, and reasoning traces, **information in the middle is most prone to hallucination** (the "lost in the middle" effect). By forcing a fresh search before each evidence submission, the relevant data is placed at the **tail of the context window** — where attention is strongest. The enforcement doesn't just validate; it structurally reduces the conditions under which hallucination occurs.

## Quick Start

```bash
pip install audit-fence
```

```python
from audit_fence import Fence

fence = Fence()

@fence.track
def search(query: str) -> str:
    """Your search tool — results are automatically tracked."""
    return my_search_backend(query)

@fence.enforce
def record_citation(claim: str, evidence: str) -> dict:
    """Submit evidence — must match a recent search result."""
    return {"claim": claim, "evidence": evidence, "status": "recorded"}

# Works: search first, then submit matching evidence
search("revenue")
record_citation(claim="Revenue $5.1B", evidence="<paste from search output>")

# Blocked: submit without searching
fence.reset()
record_citation(claim="Revenue $5.1B", evidence="anything")
# => ERROR: No search calls recorded. You must call a search tool first.

# Blocked: submit fabricated evidence
search("revenue")
record_citation(claim="Revenue $5.1B", evidence="fabricated text not in results")
# => ERROR: Evidence does not match any recent search result.
```

## How It Works

audit-fence has two decorators and three validation checks:

### `@fence.track` — Record search results

Wraps any search function. Every call's return value is appended to an internal history. The function's behavior is unchanged — the decorator only adds tracking.

### `@fence.enforce` — Validate before submission

Wraps any evidence submission function. Before the function executes, three checks run:

| Check | What it validates | Rejection message |
|-------|------------------|-------------------|
| **Search history** | At least one `@fence.track` call has been made | "No search calls recorded" |
| **Evidence match** | `evidence` parameter is a substring of a recent search result | "Evidence does not match any recent search result" |
| **Source text** *(optional)* | `claim` parameter exists in the source document | "Claim text not found in the source document" |

If any check fails, the function is **not called**. An `"ERROR: ..."` string is returned instead, and the rejection is logged with timestamp, tool name, and reason.

### Rejection logging

Every rejected submission is recorded:

```python
fence.rejections
# [{"tool": "record_citation", "content": "...", "reason": "...", "timestamp": ...}]

fence.save_log("enforcement_log.jsonl")
```

A compliance officer can review not just what was accepted, but what was rejected and why.

## Configuration

```python
fence = Fence(
    min_evidence_length=20,   # Minimum chars for evidence (default: 20)
    history_window=20,        # How many recent searches to check against (default: 20)
)
```

### Custom parameter names

```python
@fence.enforce(evidence_param="grep_output")
def submit(claim: str, grep_output: str) -> dict: ...
```

### Source text verification

Optionally verify that the claim exists in a source document (e.g., the report being audited):

```python
report = open("report.md").read()

@fence.enforce(claim_param="claim_in_report", source_text=report)
def submit(claim_in_report: str, evidence: str) -> dict: ...
```

`source_text` can also be a callable (for dynamic content):

```python
@fence.enforce(claim_param="claim", source_text=lambda: load_latest_report())
def submit(claim: str, evidence: str) -> dict: ...
```

## Framework Integration

audit-fence has **zero dependencies**. The decorators produce normal Python functions that work with any framework's tool system.

### LangGraph / LangChain

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

# Use directly with create_react_agent
agent = create_react_agent(llm, [search_evidence, record_citation])
```

When the agent submits invalid evidence, the tool returns `"ERROR: ..."`. The ReAct loop sees this as a failed tool call and retries — naturally driving the agent toward valid, search-backed evidence.

### OpenAI / Anthropic / Custom

The decorated functions are regular callables. Use them in your tool dispatch however your framework requires:

```python
# OpenAI function calling
tools_schema = [describe_function(fn) for fn in fence.tools]
# In your dispatch: call the function, it returns result or "ERROR: ..."

# Any custom framework
for fn in fence.tools:
    register_tool(fn.__name__, fn, fn.__doc__)
```

## Examples

| Example | Description |
|---------|-------------|
| [`minimal.py`](examples/minimal.py) | Core pattern in 15 lines |
| [`financial_report.py`](examples/financial_report.py) | Financial report audit with source text verification |
| [`langchain_agent.py`](examples/langchain_agent.py) | Integration with LangGraph / LangChain |

## Origin

audit-fence is extracted from [**Firn**](https://github.com/M-HuangX/Firn), a multi-agent financial analysis system with a 3-phase audit pipeline. In Firn, the enforcement mechanism is tightly integrated with a financial-domain audit workflow (specialist agents, trace directories, verdict merging). audit-fence isolates the core enforcement pattern — search tracking + evidence validation + rejection logging — as a standalone, domain-agnostic library.

The enforcement has been battle-tested in Firn across hundreds of audit runs, where it consistently prevents the audit agent from fabricating or paraphrasing evidence.

## License

[MIT](LICENSE) — use it anywhere, no restrictions.
