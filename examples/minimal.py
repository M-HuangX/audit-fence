"""Minimal audit-fence example — 15 lines to prevent hallucinated evidence."""

from audit_fence import Fence

fence = Fence()


@fence.track
def search(query: str) -> str:
    """Simulate a search tool."""
    return f"data.json:42: revenue was $5.1B in Q4 2025"


@fence.enforce
def record_citation(claim: str, evidence: str) -> dict:
    """Record a citation. `evidence` must match a recent search result."""
    return {"claim": claim, "evidence": evidence, "status": "recorded"}


# --- Agent workflow ---

# Step 1: Search first — result is automatically tracked
search("revenue Q4")

# Step 2: Submit evidence that matches the search result — accepted
result = record_citation(
    claim="Revenue was $5.1B",
    evidence="revenue was $5.1B in Q4 2025",
)
print(f"Valid:   {result}")
# => {"claim": "Revenue was $5.1B", "evidence": "...", "status": "recorded"}

# Step 3: Try to submit without searching — rejected
fence.reset()
result = record_citation(
    claim="Revenue was $5.1B",
    evidence="revenue was $5.1B in Q4 2025",
)
print(f"No search: {result}")
# => ERROR: No search calls recorded...

# Step 4: Search, then submit fabricated evidence — rejected
search("revenue Q4")
result = record_citation(
    claim="Revenue was $5.1B",
    evidence="revenue was $99.9B in Q4 2025",  # not in search results
)
print(f"Fabricated: {result}")
# => ERROR: Evidence does not match any recent search result...

# Inspect rejections
print(f"\nTotal rejections: {len(fence.rejections)}")
for r in fence.rejections:
    print(f"  [{r['tool']}] {r['reason']}")
