"""Auditing a financial report with source text verification.

Demonstrates:
- Search history enforcement (evidence must match grep results)
- Source text verification (claim must exist in the actual report)
- Rejection logging
"""

from audit_fence import Fence

# The report being audited
REPORT = """
Alphabet (GOOG) reported Q4 2025 revenue of $105.6B, up 12% year-over-year.
Operating margin expanded to 32.1%, driven by Cloud revenue growth of 28%.
The trailing P/E ratio stands at 22.4x, below the 5-year median of 25.1x.
Free cash flow reached $22.6B, supporting continued share buybacks.
"""

fence = Fence()


@fence.track
def grep_trace(pattern: str, path: str = "tools/") -> str:
    """Search raw data files for evidence (simulated)."""
    # In production, this calls ripgrep or your search backend.
    # Here we simulate realistic grep output.
    data = {
        "105.6|105600": 'fundamental_tool_calls.json:42: "totalRevenue": 105600000000  [@ tool_call #2: get_income_statement]',
        "32.1": 'fundamental_tool_calls.json:58: "operatingMargin": 0.321  [@ tool_call #3: get_financial_metrics]',
        "22.4": 'fundamental_tool_calls.json:71: "trailingPE": 22.438  [@ tool_call #1: get_stock_info]',
        "22.6": 'fundamental_tool_calls.json:85: "freeCashFlow": 22600000000  [@ tool_call #4: get_cash_flow]',
    }
    return data.get(pattern, f"No matches for '{pattern}' in {path}")


@fence.enforce(
    evidence_param="grep_evidence",
    claim_param="claim_in_report",
    source_text=REPORT,
)
def record_source_evidence(
    claim_in_report: str,
    grep_evidence: str,
    source_tool: str = "",
) -> dict:
    """Record evidence tracing a report claim to raw data."""
    return {
        "claim": claim_in_report,
        "evidence": grep_evidence,
        "source_tool": source_tool,
        "status": "recorded",
    }


# --- Valid submission: search → match → record ---

grep_trace("105.6|105600")
result = record_source_evidence(
    claim_in_report="revenue of $105.6B",
    grep_evidence='"totalRevenue": 105600000000  [@ tool_call #2: get_income_statement]',
    source_tool="get_income_statement",
)
print(f"1. Valid:     {result}")

# --- Fabricated evidence: search was done, but evidence doesn't match ---

result = record_source_evidence(
    claim_in_report="revenue of $105.6B",
    grep_evidence='"totalRevenue": 999999999999',  # fabricated — not in grep output
    source_tool="get_income_statement",
)
print(f"2. Fabricated: {result}")

# --- False claim: claim text doesn't exist in report ---

grep_trace("32.1")
result = record_source_evidence(
    claim_in_report="Operating margin expanded to 45.0%",  # report says 32.1%
    grep_evidence='"operatingMargin": 0.321  [@ tool_call #3: get_financial_metrics]',
    source_tool="get_financial_metrics",
)
print(f"3. False claim: {result}")

# --- Valid: operating margin ---

result = record_source_evidence(
    claim_in_report="Operating margin expanded to 32.1%",
    grep_evidence='"operatingMargin": 0.321  [@ tool_call #3: get_financial_metrics]',
    source_tool="get_financial_metrics",
)
print(f"4. Valid:     {result}")

# --- Summary ---

print(f"\nTotal rejections: {len(fence.rejections)}")
for r in fence.rejections:
    print(f"  [{r['tool']}] {r['reason'][:80]}")

# Save enforcement log
fence.save_log("/tmp/audit_fence_demo.jsonl")
print(f"\nLog saved to /tmp/audit_fence_demo.jsonl")
