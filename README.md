# audit-fence

**Your AI writes a report. audit-fence traces every claim back to its source — and proves the trace is real.**

Programmatic traceability for AI-generated reports. Every step verified, every rejection logged. No search, no evidence.

<p>
  <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/dependencies-zero-brightgreen" alt="Zero Dependencies"/>
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"/>
</p>

---

<p align="center">
  <img src="docs/overview.png" alt="audit-fence: from self-declared citations to enforced audit" width="100%"/>
</p>

## Why This Exists

AI agents increasingly generate reports that humans act on — financial analysis, legal summaries, medical assessments, compliance reviews. These reports contain factual claims, and stakeholders in regulated industries need to know: **where exactly did each number come from?**

Everyone agrees reports need citations. The real question is: **can you trace each citation back to source data, and can you prove that trace is real?**

### Self-declared citations — no traceable chain

Most AI tools today — ChatGPT, Gemini, Deep Research, Perplexity — let the LLM cite its own sources as it writes. The model generates *"revenue grew 26% [Source: income\_statement]"* in a single pass. The same model that might hallucinate the number is also declaring the source.

There is **no traceable chain** from claim to source data. The citation is an assertion, not a verifiable link. For a blog post, that's fine. For a financial report that a compliance officer must sign off on, it's not.

### Generation-time validation — existence, not correspondence

Tools like [instructor](https://github.com/jxnl/instructor) improve on this with structured output validation: every quote the LLM produces must be a verified substring of the source text. If the citation doesn't exist in the original document, the model is asked to retry.

This is better — at least the quote is real. But it has two blind spots:

**Existence ≠ correspondence.** The source text may contain thousands of lines. A substring check proves the quote *exists* somewhere in the document. It cannot prove it's the *right* quote for *this* claim. The document might contain "26%" in five different contexts — a different metric, a different entity, a different time period. The model picks one. The substring check passes. The attribution is wrong. This failure mode is **invisible** to any system that only checks "does this string appear somewhere in the text."

**Writing and citing are separate cognitive tasks.** Asking an LLM to simultaneously compose analysis and attach precise citations degrades both. Independent engineering teams have found that separating claim generation from evidence retrieval — write first, then trace each claim back to its source — [reduces hallucination from ~30% to under 1%](https://medium.com/lets-code-future/how-to-make-llms-cite-their-sources-and-why-rag-isnt-enough-86a9b107feed). The model performs better on focused, single-purpose work: *"find evidence for this specific claim"* produces far more reliable results than *"write a report and cite everything as you go."*

These tools provide a form of citation quality at generation time, but they don't produce a **traceable audit chain** — there's no independent verification step, no record of what was searched, no log of what was rejected.

### The missing piece — who audits the auditor?

The right architecture is clear: let the writer write freely, then send an **independent audit agent** to verify every claim against the raw data after the fact.

But this just moves the problem. The audit agent is itself an LLM. When its context window is packed with the report, raw data, and reasoning traces, it can fabricate evidence just as easily: *"I checked line 42 and it says $5.1B."* Did it actually check, or is it confabulating from a 100K-token context?

Generation-time validators can't help here — the auditor isn't writing a report with structured output. It's searching for evidence and building a trace from each claim to its source. You need a constraint that makes **the tracing process itself trustworthy**: one that operates at the tool level, forcing the agent to prove it actually searched before it can record anything.

**This is what audit-fence does.**

---

## What audit-fence Does

audit-fence traces every claim in an AI-generated report back to its source data — and ensures the trace itself is trustworthy. One rule, enforced by code: **no search, no evidence.**

```
Without enforcement:
  Audit agent → "I found X in the data"  →  Record  →  ✓ accepted (unverified)

With audit-fence:
  Audit agent → search("X")              →  Result recorded to history
  Audit agent → submit(evidence="X...")  →  Verify evidence ∈ search history  →  ✓ or REJECTED
```

Two decorators. Three validation checks. Zero dependencies.

<p align="center">
  <img src="docs/mechanism.png" alt="How audit-fence enforces evidence verification — search, match, log" width="100%"/>
</p>

**`@fence.track`** wraps your search tool. Every search result is recorded in an internal history.

**`@fence.enforce`** wraps your evidence submission tool. Before the function executes, it validates:

| Check | What it validates | On failure |
|-------|------------------|------------|
| **Search history** | At least one search has been performed | Rejected: "No search calls recorded" |
| **Evidence match** | Submitted evidence is a character-level substring of a recent search result | Rejected: "Evidence does not match any recent search result" |
| **Source text** *(optional)* | Claim text exists in the source document being audited | Rejected: "Claim text not found in the source document" |

If any check fails, the function is **not called**. An `ERROR` string is returned (which ReAct agents naturally retry on), and the rejection is logged with timestamp, tool name, and reason — a compliance officer can inspect not just what was accepted, but what was rejected and why.

### Why this works — it's not just validation

Forcing a fresh search before each evidence submission isn't just a policy check. It exploits a known property of transformer attention.

When an audit agent's context window is packed with the report, source data, and reasoning traces, **information in the middle is most prone to hallucination** — the "[lost in the middle](https://arxiv.org/abs/2307.03172)" effect. If the evidence the agent needs sits in the middle of a long context, the model is more likely to misquote, misattribute, or fabricate.

By requiring a `search()` call before every `submit()`, audit-fence forces the relevant evidence to the **tail of the context window** — where attention is strongest and hallucination is least likely. The enforcement doesn't just catch fabrication after the fact; it **structurally reduces the conditions under which fabrication occurs**.

### How audit-fence compares

| | Self-declared | Generation-time validated | **Enforced tracing** |
|---|---|---|---|
| **Examples** | ChatGPT, Gemini, Deep Research | instructor, Pydantic validators | **audit-fence** |
| **Traceability** | None — citation is an assertion | Partial — quote exists in source | Claim → search result → source data |
| **Who is constrained** | Nobody | The writer | The agent doing the tracing |
| **When** | During generation | During generation | After the report exists |
| **Prevents fabricated evidence** | No | Partially (existence only) | Yes (must prove search happened) |
| **Reduces wrong attribution** | No | No | Yes (targeted search per claim) |
| **Audit trail** | None | Retry silently | Full rejection log (JSONL) |

These approaches are complementary. Generation-time validation improves output quality; audit-fence provides the traceable evidence chain after the fact. Different stages, different guarantees.

---

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
    return my_search_backend(query)  # ripgrep, SQL, API call, etc.

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

See [`examples/`](examples/) for complete, runnable scripts including a [financial report audit](examples/financial_report.py) and [LangGraph integration](examples/langchain_agent.py).

---

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

Optionally verify that the claim text exists in the source document being audited:

```python
report = open("report.md").read()

@fence.enforce(claim_param="claim_in_report", source_text=report)
def submit(claim_in_report: str, evidence: str) -> dict: ...
```

`source_text` can be a callable for dynamic content:

```python
@fence.enforce(claim_param="claim", source_text=lambda: load_latest_report())
def submit(claim: str, evidence: str) -> dict: ...
```

### Rejection logging

Every rejected submission is recorded:

```python
fence.rejections
# [{"tool": "record_citation", "content": "...", "reason": "...", "timestamp": ...}]

fence.save_log("enforcement_log.jsonl")
```

---

## Framework Integration

audit-fence has **zero dependencies**. The decorators produce standard Python functions that compose with any framework's tool system.

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

---

## Examples

| Example | Description |
|---------|-------------|
| [`minimal.py`](examples/minimal.py) | Core pattern in 15 lines |
| [`financial_report.py`](examples/financial_report.py) | Financial report audit with source text verification |
| [`langchain_agent.py`](examples/langchain_agent.py) | Integration with LangGraph / LangChain |

## Scope and Limitations

Honesty about what a tool does and doesn't do matters — especially in compliance contexts.

| | Status | Detail |
|---|---|---|
| **Prevents fabrication** | Solved | The auditor cannot record evidence it never searched for. Submission without a matching search result is programmatically rejected. |
| **Improves attribution accuracy** | Improved | Forcing a targeted search per claim is structurally better than matching against an entire static document — the search result is specific to the claim being verified. |
| **Guarantees correct attribution** | Not yet solved | The agent could still pick the wrong match from valid search results — a passage that is real but corresponds to a different claim. This is the "existence ≠ correspondence" problem. audit-fence reduces it; it does not eliminate it. |
| **Proves causality** | Open research | Tracing a number to its source proves *where* it came from, not *why* it was used or whether the reasoning is sound. Causal verification remains an open problem across the field. |

audit-fence provides a **verifiable enforcement layer** — a necessary foundation that other verification methods (semantic matching, causal reasoning) can build on top of. The enforcement log gives you a complete record of what was searched, what was submitted, and what was rejected, regardless of how sophisticated the verification logic becomes in the future.

## Origin

audit-fence is extracted from [**Firn**](https://github.com/M-HuangX/Firn), a multi-agent financial analysis system with a full 3-phase audit pipeline, 1000+ tests, and deterministic verdict assignment. In Firn, the enforcement mechanism is integrated with a financial-domain workflow — specialist agents, trace directories, verdict merging. audit-fence isolates the core enforcement pattern as a standalone, domain-agnostic library.

## License

[MIT](LICENSE) — use it anywhere, no restrictions.
