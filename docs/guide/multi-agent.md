[← Back to README](../../README.md)

# Multi-Agent Enforcement

A single Fence works for one agent. Production systems often have multiple agents — specialists that search, a synthesizer that writes, an auditor that verifies. **`fence.link(upstream)`** declares "this fence can cite evidence from that fence's history."

### Hierarchical — manager cites workers

```python
from audit_fence import Fence

worker_a = Fence(name="worker_a")
worker_b = Fence(name="worker_b")
manager = Fence(name="manager")

manager.link(worker_a, worker_b)

# Workers search independently
tools_a = worker_a.wrap_tools(worker_a_tools, search=["search_*"])
tools_b = worker_b.wrap_tools(worker_b_tools, search=["search_*"])

# Manager validates against its own history PLUS both workers'
manager_tools = manager.wrap_tools(
    [search_summary, write_report],
    search=["search_*"],
    submit=["write_*"],
)
```

### Pipeline — transitive evidence flow

```python
researcher = Fence(name="researcher")
enricher = Fence(name="enricher")
reporter = Fence(name="reporter")

enricher.link(researcher)    # enricher can cite researcher
reporter.link(enricher)      # reporter can cite enricher AND researcher (transitive)
```

Links are transitive. Each stage validates against the accumulated history of all preceding stages.

### Production — multi-specialist audit (Firn)

A real-world topology from [Firn](https://github.com/M-HuangX/Firn), the financial analysis system audit-fence was extracted from:

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

# Unified view
group.save_log("audit/enforcement_log.jsonl")
```

Each R1 auditor is **isolated** to its specialist. The R2 auditor intentionally sees all four. The topology encodes the audit policy.

### Path sandboxing

Different agents should be restricted to different data sources. `SandboxedSearch` wraps a search backend with path restrictions:

```python
from audit_fence import SandboxedSearch

search_a = SandboxedSearch(
    backend=grep_backend,
    allowed_dirs=["trace/specialist_outputs/"],
)

search_b = SandboxedSearch(
    backend=grep_backend,
    allowed_dirs=["tools/"],
    allowed_files=["report.md"],
)

search_a("revenue", "trace/specialist_outputs/fund.md")   # allowed
search_a("revenue", "tools/data.json")                     # => ERROR: outside sandbox
```

Path traversal (e.g., `tools/../secret/data.json`) is automatically blocked.

### RipgrepBackend

A ready-to-use search backend wrapping the `rg` CLI via subprocess:

```python
from audit_fence import RipgrepBackend, SandboxedSearch, Fence

fence = Fence()

grep = RipgrepBackend(root="./trace/")
search = fence.wrap_tool(grep, role="search")

# With sandbox
sandboxed = SandboxedSearch(backend=grep, allowed_dirs=["tools/"])
search = fence.wrap_tool(sandboxed, role="search")
```

Requires `rg` (ripgrep) installed on the system — no Python dependencies added.

### FenceGroup

Optional convenience for managing multiple fences:

```python
group = FenceGroup()
fund = group.create("fundamental", min_evidence_length=20)
tech = group.create("technical", min_evidence_length=20)

group["fundamental"].rejections       # named access
group.all_rejections                  # sorted across all fences
group.all_claims                      # all ClaimRecords
group.save_log("audit.jsonl")         # combined log
group.reset()                         # reset all

chain = group.trace_chain(some_claim) # evidence chain traversal
```
