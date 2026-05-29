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

Recent [mechanistic analysis of transformer internals](https://doi.org/10.1007/978-3-032-21324-2_35) confirms why this is unreliable: LLM citation decisions rely heavily on shallow heuristics like entity name co-occurrence — the model matches surface patterns rather than genuinely verifying that the cited source supports the claim.

### Generation-time validation — existence, not correspondence

Tools like [instructor](https://github.com/jxnl/instructor) improve on this with structured output validation: every quote the LLM produces must be a verified substring of the source text. If the citation doesn't exist in the original document, the model is asked to retry.

This is better — at least the quote is real. But it has two blind spots:

**Existence ≠ correspondence.** The source text may contain thousands of lines. A substring check proves the quote *exists* somewhere in the document. It cannot prove it's the *right* quote for *this* claim. The document might contain "26%" in five different contexts — a different metric, a different entity, a different time period. The model picks one. The substring check passes. The attribution is wrong. This failure mode is **invisible** to any system that only checks "does this string appear somewhere in the text."

**Writing and citing are separate cognitive tasks.** Asking an LLM to simultaneously compose analysis and attach precise citations degrades both. Independent engineering teams have found that separating claim generation from evidence retrieval — write first, then trace each claim back to its source — [reduces hallucination from ~30% to under 1%](https://medium.com/lets-code-future/how-to-make-llms-cite-their-sources-and-why-rag-isnt-enough-86a9b107feed). The model performs better on focused, single-purpose work: *"find evidence for this specific claim"* produces far more reliable results than *"write a report and cite everything as you go."*

These tools provide a form of citation quality at generation time, but they don't produce a **traceable audit chain** — there's no independent verification step, no record of what was searched, no log of what was rejected.

### The missing piece — who audits the auditor?

The right architecture is clear: let the writer write freely, then send an **independent audit agent** to verify every claim against the raw data after the fact.

But this just moves the problem. The audit agent is itself an LLM. When its context window is packed with the report, source data, and reasoning traces, it can fabricate evidence just as easily: *"I checked line 42 and it says $5.1B."* Did it actually check, or is it confabulating from a 100K-token context?

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

| | Self-declared | Generation-time validated | Post-hoc NLI-verified | **Enforced tracing** |
|---|---|---|---|---|
| **Examples** | ChatGPT, Gemini, Deep Research | instructor, Pydantic validators | [VeriCite](https://arxiv.org/abs/2510.11394) | **audit-fence** |
| **Traceability** | None — citation is an assertion | Partial — quote exists in source | Claim → passage (semantic match) | Claim → search result → source data |
| **Who is constrained** | Nobody | The writer | The auditor (by another model) | The auditor (by code) |
| **Verifier** | None | Schema validator | NLI model (~80% accuracy) | Deterministic substring match |
| **When** | During generation | During generation | After generation | After generation |
| **Prevents fabricated evidence** | No | Partially (existence only) | Mostly (but NLI can misjudge) | Yes (must prove search happened) |
| **Audit trail** | None | Retry silently | NLI scores per statement | Full rejection log (JSONL) |

These approaches are complementary, not competing. Generation-time validation improves output quality. Post-hoc NLI verification catches unsupported statements. audit-fence ensures the tracing process itself is trustworthy — the evidence you record is the evidence you actually found. Different stages, different guarantees.

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
# => ERROR: No search calls recorded. You must call a search tool first to find evidence before submitting.

# Blocked: submit fabricated evidence
search("revenue")
record_citation(claim="Revenue $5.1B", evidence="fabricated text not in results")
# => ERROR: Evidence does not match any recent search result. Call a search tool first, then paste the matching output into the evidence field.
```

Both `@fence.track` and `@fence.enforce` transparently support async functions — no separate API needed.

See [`examples/`](examples/) for complete, runnable scripts including a [financial report audit](examples/financial_report.py) and [LangGraph integration](examples/langchain_agent.py).

---

## Configuration

```python
fence = Fence(
    name="audit_r1",          # Optional identifier (for multi-agent, logging)
    min_evidence_length=20,   # Minimum chars for evidence (default: 20)
    history_window=20,        # How many recent searches to check against (default: 20)
    history_limit=100,        # Max total records kept in memory (default: unlimited)
    context={"ticker": "AAPL", "phase": "audit"},  # Attached to every rejection log entry
    track_all=False,          # When True, wrap() tracks ALL tools (see Soft Enforcement)
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
# [{"tool": "record_citation", "content": "...", "reason": "...", "timestamp": ..., "context": {...}}]

fence.save_log("enforcement_log.jsonl")
```

---

## Framework Integration

audit-fence has **zero dependencies**. It provides two integration paths: `wrap()` for adding enforcement to existing tool lists, and decorators for new projects.

### wrap() — for existing codebases (recommended)

If you already have tools defined, `wrap()` adds enforcement without modifying any function definitions. Pass glob patterns to classify tools by name:

```python
from audit_fence import Fence

fence = Fence()

# Your existing tools — no changes needed
existing_tools = [search_web, get_financials, analyze_data, write_report]

# One call: classify by name pattern, get back enforced tools
protected_tools = fence.wrap(
    existing_tools,
    search=["search_*", "get_*"],     # these get tracked
    submit=["write_*"],               # these get enforced
)

agent = create_react_agent(llm, protected_tools)
```

Tools matching `search` patterns are tracked (results recorded to history). Tools matching `submit` patterns are enforced (evidence validated before execution). Unmatched tools pass through unchanged.

You can also match by function reference instead of name:

```python
protected_tools = fence.wrap(
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
# OpenAI function calling
tools_schema = [describe_function(fn) for fn in fence.tools]
# In your dispatch: call the function, it returns result or "ERROR: ..."

# Any custom framework
for fn in fence.tools:
    register_tool(fn.__name__, fn, fn.__doc__)
```

---

## Multi-Agent Enforcement

A single Fence works for one agent. But production systems often have multiple agents — specialists that search, a core agent that synthesizes, an auditor that verifies. When the auditor cites evidence that a specialist found, which Fence's history should it check against?

The answer: **`fence.link(upstream)`** — one primitive that declares "this fence can cite evidence from that fence's history."

<!-- TODO: multi-agent diagram -->

### Hierarchical — manager cites workers

```python
from audit_fence import Fence

worker_a = Fence(name="worker_a")
worker_b = Fence(name="worker_b")
manager = Fence(name="manager")

manager.link(worker_a, worker_b)

# Workers search independently
tools_a = worker_a.wrap(worker_a_tools, search=["search_*"])
tools_b = worker_b.wrap(worker_b_tools, search=["search_*"])

# Manager's submit tool validates against its own history
# PLUS both workers' histories
manager_tools = manager.wrap(
    [search_summary, write_report],
    search=["search_*"],
    submit=["write_*"],
)
```

When `write_report` runs, enforcement checks the manager's own search history plus all linked upstream histories. If the evidence appeared in any worker's search results, it passes.

### Pipeline — transitive evidence flow

```python
researcher = Fence(name="researcher")
enricher = Fence(name="enricher")
reporter = Fence(name="reporter")

enricher.link(researcher)    # enricher can cite researcher
reporter.link(enricher)      # reporter can cite enricher AND researcher (transitive)
```

Links are transitive. The reporter never directly links to the researcher, but because the enricher does, the reporter sees the full chain. Each stage's enforcement validates against the accumulated history of all stages before it.

### Production — multi-specialist audit (Firn)

A real-world topology from [Firn](https://github.com/M-HuangX/Firn), the financial analysis system audit-fence was extracted from. Four specialist agents search raw data, a core agent writes the report, and parallel audit agents verify claims:

```python
from audit_fence import Fence, FenceGroup

group = FenceGroup()

# Specialist agents (each searches independently)
fund = group.create("fundamental")
tech = group.create("technical")
value = group.create("value")
macro = group.create("macro")

# R1 audit: each auditor is restricted to one specialist
r1_fund = group.create("r1_fundamental")
r1_fund.link(fund)     # can only cite fundamental's searches

r1_tech = group.create("r1_technical")
r1_tech.link(tech)     # can only cite technical's searches

# R2 audit: cross-specialist verification
r2 = group.create("r2_specialist")
r2.link(fund, tech, value, macro)   # can cite all four

# After audit completes — unified view
group.save_log("audit/enforcement_log.jsonl")
print(f"Total rejections: {len(group.all_rejections)}")
```

The key insight: each R1 auditor is **isolated** to its specialist. It cannot accidentally cite evidence from a different specialist's search history. The R2 auditor intentionally sees all four. The topology encodes the audit policy.

### FenceGroup

`FenceGroup` is optional convenience — you can always create and link Fences directly. It provides named lookup, bulk operations, and group-level snapshot/restore:

```python
group = FenceGroup()
fund = group.create("fundamental", min_evidence_length=20)
tech = group.create("technical", min_evidence_length=20)

# Named access
group["fundamental"].rejections

# Bulk operations
group.all_rejections       # sorted by timestamp across all fences
group.all_history          # combined search history
group.save_log("audit.jsonl")
group.reset()
```

---

## Soft Enforcement

Not every agent has explicit submit tools. Some agents search, reason, and produce a final text response. For these, audit-fence provides **soft enforcement**: track all tool calls, then validate the output after the fact.

```python
from audit_fence import Fence

fence = Fence(track_all=True)

# wrap() with track_all and no patterns → every tool is tracked
tools = fence.wrap(existing_tools)

agent = create_react_agent(llm, tools)
result = await agent.ainvoke({"messages": [HumanMessage(content="Analyze AAPL")]})

# Post-hoc: check which quoted passages in the report match search history
report = result["messages"][-1].content
validation = fence.validate_output(report)
```

`validate_output` extracts quoted passages from the text and checks each against search history. It returns a `ValidationResult`:

```python
validation.found       # ["revenue was $5.1B in FY2025", ...]
validation.not_found   # ["fabricated quote not in history", ...]
validation.coverage    # 0.85 (fraction of quotes that matched)
validation.ok          # True if all quotes matched
validation.total       # total number of quoted passages examined
```

This works with multi-agent topologies too — `validate_output` traverses upstream links, so a manager fence validates against its own and all linked workers' histories.

---

## Persistence

Fence state can be serialized for compliance audit trails that survive process restarts.

### Single fence

```python
import json

# Save
state = fence.snapshot()
with open("fence_state.json", "w") as f:
    json.dump(state, f)

# Restore
with open("fence_state.json") as f:
    restored = Fence.restore(json.load(f))
```

### FenceGroup — preserves links

```python
# Save entire topology (fences + link relationships)
state = group.snapshot()
with open("group_state.json", "w") as f:
    json.dump(state, f)

# Restore — all fences and their links are reconstructed
with open("group_state.json") as f:
    restored_group = FenceGroup.restore(json.load(f))
```

The snapshot captures search history, rejections, configuration, and link topology. A compliance officer can load yesterday's audit state and inspect the full evidence trail.

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
