"""Tests for workflow layer: ClaimRecord, SandboxedSearch, prompts, create_record_tool, JSONL."""

import json
import os
import time

from audit_fence import (
    ClaimRecord,
    Fence,
    PROMPTS,
    SandboxedSearch,
    SearchRecord,
    create_record_tool,
)
from audit_fence.workflow import reset_claim_ids


# ============================================================================
# CLAIM RECORD
# ============================================================================


def test_claim_record_creation():
    """ClaimRecord should accept required fields and set defaults."""
    reset_claim_ids()
    record = ClaimRecord(
        claim="Revenue was $5.1B",
        claim_in_document="Revenue of $5.1 billion in FY2025",
        evidence='line 42: "revenue": 5098000000',
    )
    assert record.claim == "Revenue was $5.1B"
    assert record.claim_in_document == "Revenue of $5.1 billion in FY2025"
    assert record.evidence == 'line 42: "revenue": 5098000000'
    assert record.id == 1
    assert record.timestamp > 0


def test_claim_record_to_dict():
    """to_dict() should return a JSON-serializable dict."""
    reset_claim_ids()
    record = ClaimRecord(
        claim="test claim",
        claim_in_document="test in doc",
        evidence="test evidence",
        source_tool="get_stock_info",
        raw_value="5098000000",
        verdict="found",
    )
    d = record.to_dict()
    assert isinstance(d, dict)
    assert d["claim"] == "test claim"
    assert d["source_tool"] == "get_stock_info"
    assert d["raw_value"] == "5098000000"
    assert d["verdict"] == "found"
    # Verify JSON serializable
    json_str = json.dumps(d)
    assert json.loads(json_str) == d


def test_claim_record_defaults():
    """Optional fields should have sensible defaults."""
    reset_claim_ids()
    record = ClaimRecord(
        claim="c", claim_in_document="cid", evidence="e"
    )
    assert record.source_agent == ""
    assert record.source_tool == ""
    assert record.source_index == -1
    assert record.raw_value == ""
    assert record.search_file == ""
    assert record.search_line == -1
    assert record.verdict == ""
    assert record.source_type == "standard"
    assert record.metadata == {}


def test_claim_record_auto_increment_id():
    """ClaimRecord IDs should auto-increment."""
    reset_claim_ids()
    r1 = ClaimRecord(claim="a", claim_in_document="a", evidence="e")
    r2 = ClaimRecord(claim="b", claim_in_document="b", evidence="e")
    r3 = ClaimRecord(claim="c", claim_in_document="c", evidence="e")
    assert r1.id == 1
    assert r2.id == 2
    assert r3.id == 3


def test_claim_record_metadata():
    """metadata dict should be stored and serialized."""
    reset_claim_ids()
    record = ClaimRecord(
        claim="c",
        claim_in_document="cid",
        evidence="e",
        metadata={"custom_field": "value", "score": 0.95},
    )
    assert record.metadata["custom_field"] == "value"
    d = record.to_dict()
    assert d["metadata"]["score"] == 0.95


# ============================================================================
# SET_DOCUMENT + CLAIM_IN_DOCUMENT ENFORCEMENT
# ============================================================================


def test_claim_in_document_pass():
    """claim_in_document text found in document should be accepted."""
    fence = Fence()
    fence.set_document("The company reported revenue of $5.1 billion in FY2025.")

    @fence.track
    def search(query: str) -> str:
        return "line 42: revenue was $5.1B in FY2025 results"

    @fence.enforce
    def record(claim: str, claim_in_document: str, evidence: str) -> dict:
        return {"claim": claim, "status": "ok"}

    search("revenue")
    result = record(
        claim="revenue claim",
        claim_in_document="revenue of $5.1 billion",
        evidence="line 42: revenue was $5.1B in FY2025 results",
    )
    assert isinstance(result, dict)
    assert result["status"] == "ok"


def test_claim_in_document_fail():
    """claim_in_document text NOT in document should be rejected."""
    fence = Fence()
    fence.set_document("The company reported revenue of $5.1 billion in FY2025.")

    @fence.track
    def search(query: str) -> str:
        return "line 42: revenue was $5.1B in FY2025 results"

    @fence.enforce
    def record(claim: str, claim_in_document: str, evidence: str) -> dict:
        return {"claim": claim, "status": "ok"}

    search("revenue")
    result = record(
        claim="revenue claim",
        claim_in_document="this text does not exist in the document at all",
        evidence="line 42: revenue was $5.1B in FY2025 results",
    )
    assert isinstance(result, str)
    assert "ERROR" in result
    assert "not found in the audited document" in result


def test_claim_in_document_markdown_normalized():
    """Bold/italic markdown should be stripped before comparison."""
    fence = Fence()
    fence.set_document("Revenue was **$5.1 billion** in _FY2025_.")

    @fence.track
    def search(query: str) -> str:
        return "line 42: revenue was $5.1B in FY2025 results"

    @fence.enforce
    def record(claim: str, claim_in_document: str, evidence: str) -> dict:
        return {"claim": claim, "status": "ok"}

    search("revenue")
    # The claim text without markdown should match the normalized document
    result = record(
        claim="revenue claim",
        claim_in_document="$5.1 billion in FY2025",
        evidence="line 42: revenue was $5.1B in FY2025 results",
    )
    assert isinstance(result, dict)
    assert result["status"] == "ok"


def test_claim_in_document_callable():
    """Dynamic document via callable should work."""
    doc_content = ["Version 1 of the report with revenue data"]

    fence = Fence()
    fence.set_document(lambda: doc_content[0])

    @fence.track
    def search(query: str) -> str:
        return "line 10: revenue data found in source results"

    @fence.enforce
    def record(claim: str, claim_in_document: str, evidence: str) -> dict:
        return {"claim": claim, "status": "ok"}

    search("revenue")
    # First call: matches
    result = record(
        claim="test",
        claim_in_document="revenue data",
        evidence="line 10: revenue data found in source results",
    )
    assert isinstance(result, dict)

    # Update document
    doc_content[0] = "Version 2 of the report, completely different"

    search("revenue2")
    # Now "revenue data" is no longer in the document
    result = record(
        claim="test",
        claim_in_document="revenue data",
        evidence="line 10: revenue data found in source results",
    )
    assert isinstance(result, str)
    assert "ERROR" in result


def test_claim_in_document_not_set():
    """No document set should skip the check entirely."""
    fence = Fence()
    # Do NOT call set_document

    @fence.track
    def search(query: str) -> str:
        return "line 42: revenue data was found in the output"

    @fence.enforce
    def record(claim: str, claim_in_document: str, evidence: str) -> dict:
        return {"claim": claim, "status": "ok"}

    search("revenue")
    result = record(
        claim="test",
        claim_in_document="anything goes because no document is set",
        evidence="line 42: revenue data was found in the output",
    )
    assert isinstance(result, dict)
    assert result["status"] == "ok"


def test_claim_in_document_with_wrap():
    """claim_in_document enforcement should work via fence.wrap() too."""
    fence = Fence()
    fence.set_document("The P/E ratio is 18.9x based on trailing earnings.")

    def search_fn(query: str) -> str:
        return "line 5: P/E ratio is 18.9x trailing output"

    def record_fn(claim: str, claim_in_document: str, evidence: str) -> dict:
        return {"claim": claim, "status": "ok"}

    wrapped = fence.wrap(
        [search_fn, record_fn],
        search=["search_fn"],
        submit=["record_fn"],
    )
    wrapped_search, wrapped_record = wrapped

    wrapped_search("P/E")

    # Should pass: claim text is in document
    result = wrapped_record(
        claim="PE",
        claim_in_document="P/E ratio is 18.9x",
        evidence="line 5: P/E ratio is 18.9x trailing output",
    )
    assert isinstance(result, dict)
    assert result["status"] == "ok"

    # Should fail: claim text NOT in document
    result = wrapped_record(
        claim="PE",
        claim_in_document="P/E ratio is 25.0x which is very high",
        evidence="line 5: P/E ratio is 18.9x trailing output",
    )
    assert isinstance(result, str)
    assert "ERROR" in result


# ============================================================================
# SANDBOXED SEARCH
# ============================================================================


def test_sandbox_allowed_dir():
    """Search in allowed directory should pass through to backend."""
    results = []

    def backend(pattern: str, path: str, **kw) -> str:
        results.append((pattern, path))
        return f"found {pattern} in {path}"

    search = SandboxedSearch(backend=backend, allowed_dirs=["tools"])
    result = search("revenue", "tools/data.json")
    assert "found revenue" in result
    assert len(results) == 1


def test_sandbox_blocked_dir():
    """Search outside allowed directory should return ERROR."""
    def backend(pattern: str, path: str, **kw) -> str:
        return f"found {pattern}"

    search = SandboxedSearch(backend=backend, allowed_dirs=["tools"])
    result = search("revenue", "trace/specialist_outputs/fund.md")
    assert "ERROR" in result
    assert "outside the allowed search" in result


def test_sandbox_allowed_file():
    """Specific allowed file should pass."""
    def backend(pattern: str, path: str, **kw) -> str:
        return f"found {pattern}"

    search = SandboxedSearch(
        backend=backend,
        allowed_files=["report.md"],
    )
    result = search("revenue", "report.md")
    assert "found revenue" in result
    assert "ERROR" not in result


def test_sandbox_no_restrictions():
    """No dirs/files configured should allow everything."""
    def backend(pattern: str, path: str, **kw) -> str:
        return f"found {pattern}"

    search = SandboxedSearch(backend=backend)
    result = search("revenue", "any/path/at/all.json")
    assert "found revenue" in result


def test_sandbox_path_traversal():
    """Path traversal with '../' should be blocked."""
    def backend(pattern: str, path: str, **kw) -> str:
        return f"found {pattern}"

    search = SandboxedSearch(backend=backend, allowed_dirs=["tools"])
    result = search("revenue", "tools/../secret/data.json")
    assert "ERROR" in result


def test_sandbox_empty_path_with_restrictions():
    """Empty path with restrictions should be blocked."""
    def backend(pattern: str, path: str, **kw) -> str:
        return f"found {pattern}"

    search = SandboxedSearch(backend=backend, allowed_dirs=["tools"])
    result = search("revenue", "")
    assert "ERROR" in result


def test_sandbox_kwargs_forwarded():
    """Extra kwargs should be forwarded to the backend."""
    received = {}

    def backend(pattern: str, path: str, **kw) -> str:
        received.update(kw)
        return "ok"

    search = SandboxedSearch(backend=backend, allowed_dirs=["tools"])
    search("query", "tools/file.json", context=3, case_insensitive=True)
    assert received["context"] == 3
    assert received["case_insensitive"] is True


# ============================================================================
# PROMPTS
# ============================================================================


def test_prompts_dict():
    """All 3 templates should exist in PROMPTS dict."""
    assert "verify_claims" in PROMPTS
    assert "find_evidence" in PROMPTS
    assert "cross_reference" in PROMPTS
    assert len(PROMPTS) == 3


def test_prompt_format():
    """Placeholders should work with .format()."""
    prompt = PROMPTS["verify_claims"].format(
        document="the financial report",
        data_source="raw API call results in tools/",
    )
    assert "the financial report" in prompt
    assert "raw API call results in tools/" in prompt


def test_prompt_contains_rules():
    """Templates should include search-before-record instruction."""
    for name, template in PROMPTS.items():
        # All templates should mention searching before recording
        assert "search" in template.lower(), f"{name} missing search instruction"
        # All templates should mention VERBATIM
        assert "VERBATIM" in template, f"{name} missing VERBATIM rule"


def test_prompt_contains_examples():
    """Templates should include good vs bad examples."""
    for name, template in PROMPTS.items():
        assert "GOOD" in template, f"{name} missing good example"
        assert "BAD" in template, f"{name} missing bad example"


# ============================================================================
# SEARCH RECORD ENHANCEMENTS
# ============================================================================


def test_search_record_tool_name():
    """SearchRecord should accept tool_name field."""
    record = SearchRecord(
        query="revenue",
        result_text="found revenue data",
        tool_name="get_stock_info",
    )
    assert record.tool_name == "get_stock_info"


def test_search_record_file_path():
    """SearchRecord should accept file_path field."""
    record = SearchRecord(
        query="revenue",
        result_text="found revenue data",
        file_path="tools/fundamental_tool_calls.json",
    )
    assert record.file_path == "tools/fundamental_tool_calls.json"


def test_search_record_backward_compat():
    """Old-style SearchRecord creation should still work."""
    record = SearchRecord(
        query="revenue",
        result_text="found revenue data",
        source="fundamental",
    )
    assert record.tool_name == ""
    assert record.file_path == ""


# ============================================================================
# CREATE_RECORD_TOOL
# ============================================================================


def test_factory_basic():
    """create_record_tool should produce a working callable."""
    reset_claim_ids()
    fence = Fence()

    @fence.track
    def search(query: str) -> str:
        return "line 42: revenue was $5.1B in FY2025 results data"

    record = create_record_tool(fence, require_claim_in_document=False)
    search("revenue")

    result = record(
        claim="Revenue claim",
        claim_in_document="revenue was $5.1B",
        evidence="line 42: revenue was $5.1B in FY2025 results data",
    )
    assert isinstance(result, ClaimRecord)
    assert result.claim == "Revenue claim"
    assert len(fence.claims) == 1


def test_factory_enforcement():
    """create_record_tool should enforce search requirement."""
    reset_claim_ids()
    fence = Fence()

    record = create_record_tool(fence, require_claim_in_document=False)

    # No search performed → should fail
    result = record(
        claim="test",
        claim_in_document="test",
        evidence="fabricated evidence that is long enough to pass",
    )
    assert isinstance(result, str)
    assert "ERROR" in result
    assert len(fence.claims) == 0


def test_factory_skip_enforcement_dict():
    """Certain verdicts should bypass search requirement (dict form)."""
    reset_claim_ids()
    fence = Fence()

    record = create_record_tool(
        fence,
        extra_fields=["verdict"],
        skip_enforcement={"verdict": ["not-found", "derived"]},
        require_claim_in_document=False,
    )

    # No search, but verdict="not-found" → should pass
    result = record(
        claim="Missing claim",
        claim_in_document="missing data",
        evidence="No evidence found for this claim in any search result",
        verdict="not-found",
    )
    assert isinstance(result, ClaimRecord)
    assert result.verdict == "not-found"
    assert len(fence.claims) == 1


def test_factory_claim_in_document():
    """create_record_tool should enforce claim_in_document against document."""
    reset_claim_ids()
    fence = Fence()
    fence.set_document("The company reported revenue of $5.1 billion.")

    @fence.track
    def search(query: str) -> str:
        return "line 42: revenue was $5.1B in FY2025 detail output"

    record = create_record_tool(fence, require_claim_in_document=True)
    search("revenue")

    # Pass: claim text is in document
    result = record(
        claim="revenue",
        claim_in_document="revenue of $5.1 billion",
        evidence="line 42: revenue was $5.1B in FY2025 detail output",
    )
    assert isinstance(result, ClaimRecord)

    # Fail: claim text NOT in document
    result = record(
        claim="revenue",
        claim_in_document="this text is NOT in the document at all",
        evidence="line 42: revenue was $5.1B in FY2025 detail output",
    )
    assert isinstance(result, str)
    assert "ERROR" in result


def test_factory_jsonl_output(tmp_path):
    """Records should be written to JSONL file when output is set."""
    reset_claim_ids()
    fence = Fence()
    output_file = str(tmp_path / "claims.jsonl")
    fence.set_output(output_file)

    @fence.track
    def search(query: str) -> str:
        return "line 42: revenue was $5.1B in FY2025 long output"

    record = create_record_tool(fence, require_claim_in_document=False)
    search("revenue")

    record(
        claim="Revenue was $5.1B",
        claim_in_document="revenue",
        evidence="line 42: revenue was $5.1B in FY2025 long output",
    )

    # Verify JSONL output
    with open(output_file) as f:
        lines = f.readlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["claim"] == "Revenue was $5.1B"


def test_factory_extra_fields():
    """Extra fields should be accepted and stored in ClaimRecord."""
    reset_claim_ids()
    fence = Fence()

    @fence.track
    def search(query: str) -> str:
        return "line 42: revenue was $5.1B in FY2025 output text"

    record = create_record_tool(
        fence,
        extra_fields=["source_tool", "raw_value", "verdict"],
        require_claim_in_document=False,
    )
    search("revenue")

    result = record(
        claim="Revenue",
        claim_in_document="revenue",
        evidence="line 42: revenue was $5.1B in FY2025 output text",
        source_tool="get_stock_info",
        raw_value="5098000000",
        verdict="found",
    )
    assert isinstance(result, ClaimRecord)
    assert result.source_tool == "get_stock_info"
    assert result.raw_value == "5098000000"
    assert result.verdict == "found"


# ============================================================================
# JSONL PERSISTENCE
# ============================================================================


def test_set_output_and_save(tmp_path):
    """Claims should be saveable to JSONL via save_claims()."""
    reset_claim_ids()
    fence = Fence()
    output_file = str(tmp_path / "output.jsonl")

    @fence.track
    def search(query: str) -> str:
        return "line 42: data found in the search results output"

    record = create_record_tool(fence, require_claim_in_document=False)
    search("data")

    record(
        claim="Claim 1",
        claim_in_document="data",
        evidence="line 42: data found in the search results output",
    )
    record(
        claim="Claim 2",
        claim_in_document="data",
        evidence="line 42: data found in the search results output",
    )

    fence.save_claims(output_file)

    with open(output_file) as f:
        lines = f.readlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["claim"] == "Claim 1"
    assert json.loads(lines[1])["claim"] == "Claim 2"


def test_claims_property():
    """fence.claims should return list of ClaimRecords."""
    reset_claim_ids()
    fence = Fence()

    @fence.track
    def search(query: str) -> str:
        return "line 42: data found in the search results output"

    record = create_record_tool(fence, require_claim_in_document=False)
    search("data")

    record(
        claim="Claim A",
        claim_in_document="data",
        evidence="line 42: data found in the search results output",
    )
    record(
        claim="Claim B",
        claim_in_document="data",
        evidence="line 42: data found in the search results output",
    )

    claims = fence.claims
    assert len(claims) == 2
    assert claims[0].claim == "Claim A"
    assert claims[1].claim == "Claim B"
    # Should be a copy
    claims.clear()
    assert len(fence.claims) == 2


def test_auto_append_jsonl(tmp_path):
    """Each successful record should auto-append to JSONL."""
    reset_claim_ids()
    fence = Fence()
    output_file = str(tmp_path / "auto.jsonl")
    fence.set_output(output_file)

    @fence.track
    def search(query: str) -> str:
        return "line 42: evidence data found in search results text"

    record = create_record_tool(fence, require_claim_in_document=False)
    search("data")

    # First record
    record(
        claim="First",
        claim_in_document="data",
        evidence="line 42: evidence data found in search results text",
    )

    # Second record
    record(
        claim="Second",
        claim_in_document="data",
        evidence="line 42: evidence data found in search results text",
    )

    # Verify file has 2 lines (auto-appended, not batched)
    with open(output_file) as f:
        lines = f.readlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["claim"] == "First"
    assert json.loads(lines[1])["claim"] == "Second"


def test_save_claims_no_path_raises():
    """save_claims() without path or set_output should raise ValueError."""
    fence = Fence()
    try:
        fence.save_claims()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "No output path" in str(e)


# ============================================================================
# INTEGRATION
# ============================================================================


def test_full_audit_workflow(tmp_path):
    """End-to-end: set_document, track search, enforce record, check claims."""
    reset_claim_ids()
    fence = Fence(name="source_auditor")
    output_file = str(tmp_path / "audit.jsonl")
    fence.set_document(
        "The company reported revenue of $5.1 billion and "
        "net income of $1.2 billion in FY2025."
    )
    fence.set_output(output_file)

    # Sandboxed search
    def grep_backend(pattern: str, path: str, **kw) -> str:
        if "revenue" in pattern.lower():
            return 'line 42: "revenue": 5098000000'
        if "net_income" in pattern.lower() or "income" in pattern.lower():
            return 'line 88: "net_income": 1200000000'
        return "No matches found"

    search = SandboxedSearch(
        backend=grep_backend, allowed_dirs=["tools"]
    )

    @fence.track
    def grep(pattern: str, path: str = "tools/") -> str:
        return search(pattern, path)

    record = create_record_tool(
        fence,
        extra_fields=["source_tool", "raw_value", "verdict"],
        skip_enforcement={"verdict": ["not-found"]},
    )

    # Search and record revenue
    grep("revenue", "tools/fundamental.json")
    result = record(
        claim="Revenue was $5.1B",
        claim_in_document="revenue of $5.1 billion",
        evidence='line 42: "revenue": 5098000000',
        source_tool="get_stock_info",
        raw_value="5098000000",
        verdict="found",
    )
    assert isinstance(result, ClaimRecord)
    assert result.verdict == "found"

    # Search and record net income
    grep("net_income", "tools/fundamental.json")
    result = record(
        claim="Net income was $1.2B",
        claim_in_document="net income of $1.2 billion",
        evidence='line 88: "net_income": 1200000000',
        source_tool="get_stock_info",
        raw_value="1200000000",
        verdict="found",
    )
    assert isinstance(result, ClaimRecord)

    # Verify claims
    assert len(fence.claims) == 2
    assert fence.claims[0].claim == "Revenue was $5.1B"
    assert fence.claims[1].claim == "Net income was $1.2B"

    # Verify JSONL
    with open(output_file) as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == 2
    assert lines[0]["verdict"] == "found"

    # Verify sandboxing blocks outside access
    blocked = grep("revenue", "trace/secrets/")
    assert "ERROR" in blocked


def test_multi_fence_workflow():
    """Two fences with different sandboxes should work independently."""
    reset_claim_ids()

    # Fence A: restricted to specialist outputs
    fence_a = Fence(name="specialist_evidence")
    fence_a.set_document("Revenue grew 15% year-over-year.")

    search_a = SandboxedSearch(
        backend=lambda p, path, **kw: f"specialist: {p} found in {path}",
        allowed_dirs=["trace/specialist_outputs"],
    )

    @fence_a.track
    def grep_a(pattern: str, path: str = "trace/specialist_outputs/") -> str:
        return search_a(pattern, path)

    # Fence B: restricted to tools
    fence_b = Fence(name="source_evidence")
    fence_b.set_document("Revenue grew 15% year-over-year.")

    search_b = SandboxedSearch(
        backend=lambda p, path, **kw: f"source: {p} found in {path}",
        allowed_dirs=["tools"],
    )

    @fence_b.track
    def grep_b(pattern: str, path: str = "tools/") -> str:
        return search_b(pattern, path)

    # A can search specialist outputs
    result_a = grep_a("revenue", "trace/specialist_outputs/fund.md")
    assert "specialist: revenue" in result_a

    # A cannot search tools
    result_a_blocked = grep_a("revenue", "tools/data.json")
    assert "ERROR" in result_a_blocked

    # B can search tools
    result_b = grep_b("revenue", "tools/data.json")
    assert "source: revenue" in result_b

    # B cannot search specialist outputs
    result_b_blocked = grep_b("revenue", "trace/specialist_outputs/fund.md")
    assert "ERROR" in result_b_blocked


def test_factory_no_search_required():
    """require_search=False should skip search enforcement entirely."""
    reset_claim_ids()
    fence = Fence()
    # No searches registered, no document set

    record = create_record_tool(
        fence,
        require_search=False,
        require_claim_in_document=False,
    )

    result = record(
        claim="Test claim",
        claim_in_document="test",
        evidence="any evidence works without search",
    )
    assert isinstance(result, ClaimRecord)
    assert len(fence.claims) == 1


def test_reset_clears_claims():
    """fence.reset() should clear recorded claims."""
    reset_claim_ids()
    fence = Fence()

    @fence.track
    def search(query: str) -> str:
        return "line 42: evidence data in the search results"

    record = create_record_tool(fence, require_claim_in_document=False)
    search("test")
    record(
        claim="test",
        claim_in_document="test",
        evidence="line 42: evidence data in the search results",
    )
    assert len(fence.claims) == 1

    fence.reset()
    assert len(fence.claims) == 0


# ============================================================================
# SKIP_ENFORCEMENT — DICT & CALLABLE FORMS
# ============================================================================


def test_skip_enforcement_by_source_type():
    """Dict form: skip based on source_type (not verdict)."""
    reset_claim_ids()
    fence = Fence()

    record = create_record_tool(
        fence,
        extra_fields=["source_type"],
        skip_enforcement={"source_type": ["kb", "web", "derived"]},
        require_claim_in_document=False,
    )

    # No search, source_type="kb" → should pass (skipped)
    result = record(
        claim="KB claim",
        claim_in_document="kb data",
        evidence="knowledge base entry: stored fact about revenue",
        source_type="kb",
    )
    assert isinstance(result, ClaimRecord)
    assert result.source_type == "kb"

    # No search, source_type="standard" → should fail (not skipped)
    result = record(
        claim="Standard claim",
        claim_in_document="standard data",
        evidence="this should be rejected because no search performed",
        source_type="standard",
    )
    assert isinstance(result, str)
    assert "ERROR" in result


def test_skip_enforcement_callable():
    """Callable form: skip when custom predicate returns True."""
    reset_claim_ids()
    fence = Fence()

    record = create_record_tool(
        fence,
        skip_enforcement=lambda kw: kw.get("confidence", 0) > 0.9,
        require_claim_in_document=False,
    )

    # No search, confidence=0.95 → should pass (skipped)
    # Note: confidence is NOT a ClaimRecord field — it lives only in kwargs
    # and is visible to the skip_enforcement predicate.
    result = record(
        claim="High confidence claim",
        claim_in_document="confident data",
        evidence="pre-verified evidence from trusted external system",
        confidence=0.95,
    )
    assert isinstance(result, ClaimRecord)

    # No search, confidence=0.5 → should fail (not skipped)
    result = record(
        claim="Low confidence claim",
        claim_in_document="uncertain data",
        evidence="unverified evidence that should be rejected here",
        confidence=0.5,
    )
    assert isinstance(result, str)
    assert "ERROR" in result


def test_skip_enforcement_multi_field_dict():
    """Dict with multiple fields: skip if ANY field matches."""
    reset_claim_ids()
    fence = Fence()

    record = create_record_tool(
        fence,
        extra_fields=["verdict", "source_type"],
        skip_enforcement={
            "verdict": ["not-found"],
            "source_type": ["kb", "web"],
        },
        require_claim_in_document=False,
    )

    # No search, verdict="not-found" → skip (first field matches)
    result = record(
        claim="Missing",
        claim_in_document="missing",
        evidence="No evidence found for this claim at all anywhere",
        verdict="not-found",
        source_type="standard",
    )
    assert isinstance(result, ClaimRecord)

    # No search, source_type="kb" → skip (second field matches)
    result = record(
        claim="KB fact",
        claim_in_document="kb fact",
        evidence="knowledge base: stored assertion about the company",
        verdict="found",
        source_type="kb",
    )
    assert isinstance(result, ClaimRecord)

    # No search, verdict="found" + source_type="standard" → fail
    result = record(
        claim="Standard found",
        claim_in_document="standard",
        evidence="should be rejected because no field matches skip map",
        verdict="found",
        source_type="standard",
    )
    assert isinstance(result, str)
    assert "ERROR" in result


def test_skip_enforcement_still_checks_document():
    """Even when search is skipped, claim_in_document should still be checked."""
    reset_claim_ids()
    fence = Fence()
    fence.set_document("Revenue was $5.1B in FY2025.")

    record = create_record_tool(
        fence,
        extra_fields=["verdict"],
        skip_enforcement={"verdict": ["not-found"]},
        require_claim_in_document=True,
    )

    # Search skipped (not-found), but claim_in_document is wrong → fail
    result = record(
        claim="Missing revenue",
        claim_in_document="this text is not in the document at all",
        evidence="No evidence found for this particular missing claim",
        verdict="not-found",
    )
    assert isinstance(result, str)
    assert "ERROR" in result


# ============================================================================
# ENRICH HOOK
# ============================================================================


def test_enrich_basic():
    """enrich callback should modify ClaimRecord before persistence."""
    reset_claim_ids()
    fence = Fence()

    def add_source(record: ClaimRecord) -> ClaimRecord:
        record.source_tool = "auto_resolved_tool"
        record.source_index = 42
        return record

    @fence.track
    def search(query: str) -> str:
        return "line 10: data found in search results output"

    record = create_record_tool(
        fence,
        enrich=add_source,
        require_claim_in_document=False,
    )
    search("test")

    result = record(
        claim="Test claim",
        claim_in_document="data",
        evidence="line 10: data found in search results output",
    )
    assert isinstance(result, ClaimRecord)
    assert result.source_tool == "auto_resolved_tool"
    assert result.source_index == 42
    # Should also be in fence.claims
    assert fence.claims[0].source_tool == "auto_resolved_tool"


def test_enrich_sets_upstream(tmp_path):
    """enrich callback can set upstream_id/upstream_fence for chaining."""
    reset_claim_ids()
    fence = Fence()
    output_file = str(tmp_path / "enriched.jsonl")
    fence.set_output(output_file)

    def link_upstream(record: ClaimRecord) -> ClaimRecord:
        record.upstream_id = 7
        record.upstream_fence = "r1_fundamental"
        return record

    @fence.track
    def search(query: str) -> str:
        return "line 10: evidence found in the output results"

    record = create_record_tool(
        fence,
        enrich=link_upstream,
        require_claim_in_document=False,
    )
    search("test")

    result = record(
        claim="Linked claim",
        claim_in_document="evidence",
        evidence="line 10: evidence found in the output results",
    )
    assert isinstance(result, ClaimRecord)
    assert result.upstream_id == 7
    assert result.upstream_fence == "r1_fundamental"

    # Verify JSONL includes upstream fields
    with open(output_file) as f:
        data = json.loads(f.readline())
    assert data["upstream_id"] == 7
    assert data["upstream_fence"] == "r1_fundamental"


def test_enrich_none_is_noop():
    """enrich=None should not affect record creation."""
    reset_claim_ids()
    fence = Fence()

    @fence.track
    def search(query: str) -> str:
        return "line 10: data found in search results output"

    record = create_record_tool(
        fence, enrich=None, require_claim_in_document=False
    )
    search("test")

    result = record(
        claim="Plain",
        claim_in_document="data",
        evidence="line 10: data found in search results output",
    )
    assert isinstance(result, ClaimRecord)
    assert result.source_tool == ""  # default, not enriched


# ============================================================================
# CLAIM RECORD — UPSTREAM FIELDS
# ============================================================================


def test_claim_record_upstream_defaults():
    """upstream_id and upstream_fence should default to no-link."""
    reset_claim_ids()
    record = ClaimRecord(claim="c", claim_in_document="cid", evidence="e")
    assert record.upstream_id == -1
    assert record.upstream_fence == ""


def test_claim_record_upstream_serialization():
    """Upstream fields should survive to_dict() and JSON roundtrip."""
    reset_claim_ids()
    record = ClaimRecord(
        claim="c",
        claim_in_document="cid",
        evidence="e",
        upstream_id=5,
        upstream_fence="r1_fundamental",
    )
    d = record.to_dict()
    assert d["upstream_id"] == 5
    assert d["upstream_fence"] == "r1_fundamental"
    # JSON roundtrip
    d2 = json.loads(json.dumps(d))
    assert d2["upstream_id"] == 5
    assert d2["upstream_fence"] == "r1_fundamental"
