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

## Why Post-Hoc Audit, Not Generation-Time Validation?

Some tools validate citations at generation time — requiring every quote the LLM produces to be a substring of the source text (e.g., [instructor](https://github.com/jxnl/instructor)'s `substring_quote` pattern). This is useful for structured extraction. But it is insufficient for report-level auditing, for three reasons:

**Existence ≠ Correspondence.** A substring check proves a quote *exists* in the source. It cannot prove it's the *right* quote for the claim. When the context window contains thousands of lines, the model may cite a real passage containing "26%" — but from a different metric, a different entity, a different time period. The citation passes validation. The attribution is wrong. This failure mode is invisible to any system that only checks "does this string appear somewhere in the text."

**Analysis and citation should be separate tasks.** Asking an LLM to simultaneously write analysis and attach precise citations degrades both — it is multitasking in a domain where precision matters. Independent engineering teams have found that separating claim generation from evidence retrieval — first write, then trace each claim back to its source — [reduces hallucination rates from ~30% to under 1%](https://medium.com/lets-code-future/how-to-make-llms-cite-their-sources-and-why-rag-isnt-enough-86a9b107feed). The LLM performs better on focused, single-purpose work: "find evidence for *this specific claim*" yields far more reliable citations than "write a report and cite everything as you go."

**Who audits the auditor?** Generation-time validators trust the model: if it produces a valid substring, the citation is accepted. In a post-hoc audit, the *auditor itself* is an LLM that can fabricate evidence just as easily. audit-fence adds a constraint that no generation-time tool provides: it verifies the audit agent's evidence comes from searches it actually performed — not from memory, paraphrase, or confabulation.

| | Generation-time validators | audit-fence |
|---|---|---|
| **When** | During report writing | After the report exists |
| **Who is constrained** | The writer (report-generating LLM) | The auditor (independent verification agent) |
| **What is validated** | Quote ∈ static source text fed to the model | Evidence ∈ agent's real, recorded search history |
| **On failure** | Retry generation (goal: produce output) | Reject + log (goal: determine ground truth) |
| **Prerequisite** | You control the generation process | Report + generation traces + raw data all exist |

These approaches are complementary. You can use structured output validation during generation *and* audit-fence for post-hoc verification. audit-fence doesn't help you write better citations — it helps you **verify** them, with a constraint that even the verifier cannot bypass.

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
