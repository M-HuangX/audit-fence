# audit-fence

**Trace every claim to its source. Identify every hallucination. Before the report reaches your client.**

Every AI agent system hallucinates — regardless of model, framework, or prompt engineering. audit-fence provides automated hallucination detection and source citation for any orchestration pipeline. Model-agnostic, framework-compatible, with a pre-built audit agent that runs out of the box.

<p>
  <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/model-agnostic-brightgreen" alt="Model Agnostic"/>
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"/>
</p>

---

<p align="center">
  <img src="docs/overview.png" alt="audit-fence wraps around your agent orchestration — traces sources, flags hallucinations" width="100%"/>
</p>

## The Problem

Every industry is deploying AI agent systems to generate reports — financial analysis, legal review, medical assessment. These agents cite sources. The question compliance teams ask:

**"Can you prove each citation actually comes from the source data?"**

The answer, today, is usually no. Agents fabricate evidence — they produce citations that look right but don't match anything in the source document. The citations are assertions, not verifiable links. For a blog post, that's fine. For a financial report that a compliance officer must sign off on, it's not.

Recent [mechanistic analysis of transformer internals](https://doi.org/10.1007/978-3-032-21324-2_35) confirms why: LLM citation decisions rely on shallow heuristics like entity name co-occurrence — the model matches surface patterns rather than genuinely verifying the cited source supports the claim.

---

## How It Works

Your agent system generates a report. A separate **audit agent** — any LLM you choose — reviews that report against the underlying source data and, optionally, the reasoning traces from your pipeline. audit-fence **programmatically constrains** every tool this audit agent operates with. By requiring a targeted `search()` before every evidence submission, it forces relevant source material to the **tail of the context window** — where transformer attention is strongest — rather than relying on what the agent "remembers" from the middle of a long context, where hallucination is most likely. This infrastructure-level design maximally compresses the probability of the audit agent itself fabricating evidence.

The approach has been [systematically evaluated](#proven-effective) on real-world financial documents with deterministic ground truth, confirming its precision and reliability as an automated audit system.

The core rule: **you cannot record a citation unless it matches data you actually searched for.** One rule, enforced by code.

```
Without enforcement:
  Audit Agent → "I found X in the data"  →  Record  →  ✓ accepted (unverified)

With audit-fence (happy path):
  Audit Agent → search(query)             →  Result saved to tracked history
  Audit Agent → submit(                   →  Validation Gate
                 claim,                        ✓ search history exists?
                 claim_in_document,            ✓ evidence long enough?
                 evidence,                     ✓ evidence ∈ search history?
                 finding, ...                  ✓ claim in document?
               )                          →  Accepted

With audit-fence (hallucination caught):
  Audit Agent → submit(                   →  Validation Gate
                 claim,                        ✗ no search history!
                 claim_in_document,            ✗ text not in document!
                 evidence="revenue was...",    ✗ fabricated — not from any search!
               )                          →  REJECTED (agent must retry)
```

Minimal integration. Full traceability. Model-agnostic. Four validation checks, pre-built audit agent.

<p align="center">
  <img src="docs/mechanism.png" alt="Runtime enforcement — how audit-fence verifies evidence" width="100%"/>
</p>

**Search** — point audit-fence at your source data directory (where your production system's outputs and trace files are stored) and the audit agent searches them claim by claim using [ripgrep](https://github.com/BurntSushi/ripgrep). Results are automatically tracked for enforcement.

**Evidence submission** — `fence.record_tool()` gives you a fully configured submission tool out of the box: enforcement checks, structured `ClaimRecord` output, and auto-persistence to JSONL.

Before each submission executes, the fence validates:

| Check | What it validates | On failure |
|-------|------------------|------------|
| **Search history** | Recent search history is not empty — the agent must have searched before submitting | Rejected: "No search calls recorded" |
| **Evidence length** | Evidence meets minimum length (`min_evidence_length`, default 20) | Rejected: "Evidence too short" |
| **Evidence match** | Submitted evidence is a verbatim substring of a recent search result — this is the core enforcement that ties each submission to a specific search output | Rejected: "Evidence does not match any recent search result" |
| **Claim in document** | The agent must specify which verbatim text in the report it is auditing — verified as an exact substring of the document provided via `set_document()` | Rejected: "Claim text not found in the source document" |

If any check fails, the submission is **blocked** and the agent is asked to retry. Every rejection is logged for audit review.

### Why this works

Forcing a search before each evidence submission exploits a known property of transformer attention. When an audit agent's context window is packed with report, source data, and reasoning traces, **information in the middle is most prone to hallucination** — the "[lost in the middle](https://arxiv.org/abs/2307.03172)" effect. By requiring a `search()` call before every `submit()`, audit-fence forces the relevant evidence to the **tail of the context window** — where attention is strongest and hallucination is least likely.

The enforcement doesn't just catch fabrication after the fact; it **structurally reduces the conditions under which fabrication occurs**.

---

## Quick Start

```bash
pip install audit-fence langgraph langchain-openai
```

```python
from audit_fence import Fence
from langchain_openai import ChatOpenAI

fence = Fence()
fence.set_document(open("report.md").read())    # the report being audited
fence.set_source("./source_data/")              # where to search (uses ripgrep)
fence.set_output("audit/citations.jsonl")       # auto-persist every record

result = await fence.audit(llm=ChatOpenAI(model="gpt-4o", temperature=0.1))

print(f"{result.summary['total']} claims audited")
print(f"{result.summary.get('found', 0)} found, {result.summary.get('not-found', 0)} not found")
print(f"{len(result.rejections)} enforcement rejections")
```

Works with any LangChain-compatible model:

```bash
pip install langgraph langchain-anthropic   # for Claude
pip install langgraph langchain-openai      # for GPT
pip install langgraph langchain-community   # for DeepSeek, Ollama, etc.
```

```python
from langchain_anthropic import ChatAnthropic
result = await fence.audit(llm=ChatAnthropic(model="claude-sonnet-4-20250514"))
```

See [`examples/`](examples/) for complete, runnable scripts including a [financial report audit](examples/financial_report.py) and [LangGraph integration](examples/langchain_agent.py).

---

## Documentation

| Guide | Description |
|-------|-------------|
| [Integration](docs/guide/integration.md) | Setup methods, tool wrapping, search configuration |
| [Structured Claims](docs/guide/structured-claims.md) | ClaimRecord workflow, enrich hooks |
| [Multi-Agent](docs/guide/multi-agent.md) | FenceGroup, cross-agent linking, topology |
| [Snapshot](docs/guide/snapshot.md) | Production trace capture for audit agents |
| [Soft Enforcement](docs/guide/soft-enforcement.md) | Conditional enforcement, skip patterns |
| [Persistence](docs/guide/persistence.md) | JSON export, state serialization |
| [Examples](docs/guide/examples.md) | End-to-end usage examples |

---

## Proven Effective

Tested on SEC 10-K filings (45K–243K tokens) with deterministic XBRL ground truth — no LLM judges, no human annotation, no evaluation circularity.

<!-- Benchmark results, charts, and methodology link will be added after benchmark completion -->

---

## Designed For

**Financial services** — Agent systems generating investment analysis, risk reports, compliance reviews. Every number traceable to SEC filings, market data feeds, or internal databases.

**Legal** — Automated contract review, regulatory compliance checking. Every statute citation verified against the actual legal text.

**Life sciences** — Clinical trial summaries, drug interaction reports. Every data point linked to the original study or FDA filing.

Any domain where an AI-generated report must withstand regulatory scrutiny.

---

## How It Compares

audit-fence operates in the same paradigm as academic fact verification systems — decompose claims, retrieve evidence, verify against source. The difference is in enforcement mechanism and ground truth.

| | FActScore | RAGAS | FEVER | **audit-fence** |
|---|---|---|---|---|
| **What it does** | Evaluates factual precision | Evaluates RAG faithfulness | Classifies claim veracity | **Detects hallucinations + annotates sources** |
| **Verification** | LLM judge | NLI model | Trained classifier | **Mechanical enforcement** |
| **Source annotation** | No | No | No | **Yes — claim to source data** |
| **Dependencies** | Retriever + LLM | LLM API | Training data | **Zero (core) / LangGraph + any LLM (audit agent)** |
| **Designed for** | Research evaluation | RAG pipeline evaluation | Research benchmark | **Production compliance** |

These approaches are complementary. FActScore and RAGAS evaluate output quality after generation. audit-fence enforces source traceability during generation — different stages, different guarantees.

## Scope and Limitations

Honesty about what a tool does and doesn't do matters — especially in compliance contexts.

| | Status | Detail |
|---|---|---|
| **Prevents fabrication** | Solved | The auditor cannot record evidence it never searched for. Submission without a matching search result is programmatically rejected. |
| **Improves attribution accuracy** | Improved | Forcing a targeted search per claim is structurally better than matching against an entire static document — the search result is specific to the claim being verified. |
| **Guarantees correct attribution** | Not yet solved | The agent could still pick the wrong match from valid search results — a passage that is real but corresponds to a different claim. audit-fence reduces it; it does not eliminate it. |
| **Proves causality** | Open research | Tracing a number to its source proves *where* it came from, not *why* it was used or whether the reasoning is sound. |

audit-fence provides a **verifiable enforcement layer** — a necessary foundation that other verification methods (semantic matching, causal reasoning) can build on top of.

## Origin

audit-fence is extracted from [**Firn**](https://github.com/M-HuangX/Firn), a multi-agent financial analysis system with a full 3-phase audit pipeline, 1000+ tests, and deterministic verdict assignment. In Firn, the enforcement mechanism is integrated with a financial-domain workflow — specialist agents, trace directories, verdict merging. audit-fence isolates the core enforcement pattern as a standalone, domain-agnostic library.

## License

[MIT](LICENSE) — use it anywhere, no restrictions.
