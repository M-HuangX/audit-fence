"""Tests for audit-fence core enforcement logic."""

import json
from audit_fence import Fence


# -- Rejection: no search history -------------------------------------------

def test_reject_without_search():
    fence = Fence()

    @fence.enforce
    def submit(claim: str, evidence: str) -> dict:
        return {"claim": claim}

    result = submit(claim="test", evidence="some evidence text that is long enough")
    assert isinstance(result, str)
    assert "ERROR" in result
    assert "No search calls recorded" in result
    assert len(fence.rejections) == 1


# -- Rejection: evidence doesn't match search --------------------------------

def test_reject_mismatched_evidence():
    fence = Fence()

    @fence.track
    def search(query: str) -> str:
        return "line 42: revenue was $5.1B in FY2025"

    @fence.enforce
    def submit(claim: str, evidence: str) -> dict:
        return {"claim": claim}

    search("revenue")
    result = submit(claim="test", evidence="fabricated evidence not in any search result")
    assert "ERROR" in result
    assert "does not match" in result
    assert len(fence.rejections) == 1


# -- Rejection: evidence too short -------------------------------------------

def test_reject_short_evidence():
    fence = Fence(min_evidence_length=20)

    @fence.track
    def search(query: str) -> str:
        return "short"

    @fence.enforce
    def submit(claim: str, evidence: str) -> dict:
        return {"claim": claim}

    search("test")
    result = submit(claim="test", evidence="short")
    assert "ERROR" in result
    assert "too short" in result.lower()


# -- Acceptance: matching evidence -------------------------------------------

def test_accept_matching_evidence():
    fence = Fence()

    @fence.track
    def search(query: str) -> str:
        return "line 42: revenue was $5.1B in FY2025"

    @fence.enforce
    def submit(claim: str, evidence: str) -> dict:
        return {"claim": claim, "status": "ok"}

    search("revenue")
    result = submit(claim="test", evidence="revenue was $5.1B in FY2025")
    assert isinstance(result, dict)
    assert result["status"] == "ok"
    assert len(fence.rejections) == 0


# -- Acceptance: grep file:line: format --------------------------------------

def test_accept_grep_format_evidence():
    fence = Fence()

    @fence.track
    def search(query: str) -> str:
        return 'tools/data.json:42: "totalRevenue": 5098000000'

    @fence.enforce
    def submit(claim: str, evidence: str) -> dict:
        return {"claim": claim, "status": "ok"}

    search("revenue")
    result = submit(
        claim="test",
        evidence='tools/data.json:42: "totalRevenue": 5098000000',
    )
    assert isinstance(result, dict)
    assert result["status"] == "ok"


# -- Source text match -------------------------------------------------------

def test_source_text_match_pass():
    report = "The company reported revenue of $5.1B, up 26% year over year."
    fence = Fence()

    @fence.track
    def search(query: str) -> str:
        return "line 1: reported revenue of $5.1B in Q4"

    @fence.enforce(claim_param="claim", source_text=report)
    def submit(claim: str, evidence: str) -> dict:
        return {"claim": claim, "status": "ok"}

    search("revenue")
    result = submit(claim="revenue of $5.1B", evidence="reported revenue of $5.1B in Q4")
    assert isinstance(result, dict)
    assert result["status"] == "ok"


def test_source_text_match_fail():
    report = "The company reported revenue of $5.1B."
    fence = Fence()

    @fence.track
    def search(query: str) -> str:
        return "line 1: reported revenue of $5.1B in Q4"

    @fence.enforce(claim_param="claim", source_text=report)
    def submit(claim: str, evidence: str) -> dict:
        return {"claim": claim}

    search("revenue")
    result = submit(claim="revenue of $10B", evidence="reported revenue of $5.1B in Q4")
    assert "ERROR" in result
    assert "not found in the source" in result


# -- Source text as callable -------------------------------------------------

def test_source_text_callable():
    texts = {"v1": "Revenue was $5.1B in fiscal year 2025."}
    fence = Fence()

    @fence.track
    def search(query: str) -> str:
        return "line 42: Revenue was $5.1B in fiscal year 2025"

    @fence.enforce(
        claim_param="claim",
        source_text=lambda: texts["v1"],
    )
    def submit(claim: str, evidence: str) -> dict:
        return {"claim": claim, "status": "ok"}

    search("revenue")
    result = submit(
        claim="Revenue was $5.1B",
        evidence="Revenue was $5.1B in fiscal year 2025",
    )
    assert isinstance(result, dict)

    # Change the source text dynamically
    texts["v1"] = "Revenue was $3.2B in fiscal year 2025."
    result = submit(
        claim="Revenue was $5.1B",
        evidence="Revenue was $5.1B in fiscal year 2025",
    )
    assert "ERROR" in result


# -- History window ----------------------------------------------------------

def test_history_window():
    fence = Fence(history_window=2)

    @fence.track
    def search(query: str) -> str:
        return f"result for query={query} with extra context padding"

    @fence.enforce
    def submit(claim: str, evidence: str) -> dict:
        return {"claim": claim}

    search("alpha")
    search("beta")
    search("gamma")  # alpha is now outside window of 2

    # gamma (in window) should pass
    result = submit(claim="test", evidence="result for query=gamma with extra context padding")
    assert isinstance(result, dict)

    # alpha (outside window) should fail
    result = submit(claim="test", evidence="result for query=alpha with extra context padding")
    assert "ERROR" in result


# -- Custom evidence_param --------------------------------------------------

def test_custom_evidence_param():
    fence = Fence()

    @fence.track
    def search(query: str) -> str:
        return "found: trailingPE = 18.923 from stock info"

    @fence.enforce(evidence_param="grep_output")
    def submit(claim: str, grep_output: str) -> dict:
        return {"claim": claim, "status": "ok"}

    search("PE")
    result = submit(claim="test", grep_output="trailingPE = 18.923 from stock info")
    assert isinstance(result, dict)
    assert result["status"] == "ok"


# -- Custom min_length ------------------------------------------------------

def test_custom_min_length():
    fence = Fence(min_evidence_length=10)

    @fence.track
    def search(query: str) -> str:
        return "data"

    @fence.enforce(min_length=50)
    def submit(claim: str, evidence: str) -> dict:
        return {"claim": claim}

    search("test")
    result = submit(claim="test", evidence="short evidence")  # 14 chars < 50
    assert "ERROR" in result
    assert "too short" in result.lower()


# -- Positional args --------------------------------------------------------

def test_positional_args():
    fence = Fence()

    @fence.track
    def search(query: str) -> str:
        return "line 99: operating margin 32.1%"

    @fence.enforce
    def submit(claim: str, evidence: str) -> dict:
        return {"claim": claim, "status": "ok"}

    search("margin")
    result = submit("margin claim", "operating margin 32.1%")
    assert isinstance(result, dict)
    assert result["status"] == "ok"


# -- save_log ---------------------------------------------------------------

def test_save_log(tmp_path):
    fence = Fence()

    @fence.enforce
    def submit(claim: str, evidence: str) -> dict:
        return {"claim": claim}

    submit(claim="test", evidence="no search performed, this will be rejected")

    log_path = tmp_path / "enforcement.jsonl"
    fence.save_log(log_path)

    with open(log_path) as f:
        entries = [json.loads(line) for line in f]
    assert len(entries) == 1
    assert entries[0]["tool"] == "submit"
    assert "No search calls" in entries[0]["reason"]


# -- reset -------------------------------------------------------------------

def test_reset():
    fence = Fence()

    @fence.track
    def search(query: str) -> str:
        return f"result: {query}"

    search("test")
    assert len(fence.history) == 1

    fence.reset()
    assert len(fence.history) == 0
    assert len(fence.rejections) == 0


# -- tools property ----------------------------------------------------------

def test_tools_property():
    fence = Fence()

    @fence.track
    def search(query: str) -> str:
        return "result"

    @fence.enforce
    def submit(claim: str, evidence: str) -> dict:
        return {}

    assert len(fence.tools) == 2
    assert fence.tools[0]._fence_role == "search"
    assert fence.tools[1]._fence_role == "submit"


# -- Multiple fences are independent ----------------------------------------

def test_independent_fences():
    fence_a = Fence()
    fence_b = Fence()

    @fence_a.track
    def search_a(q: str) -> str:
        return "result from fence A search"

    @fence_b.enforce
    def submit_b(claim: str, evidence: str) -> dict:
        return {"claim": claim}

    search_a("test")
    # fence_b has no search history — should reject
    result = submit_b(claim="test", evidence="result from fence A search")
    assert "ERROR" in result


# -- Multiline evidence -----------------------------------------------------

def test_multiline_evidence():
    fence = Fence()

    @fence.track
    def search(query: str) -> str:
        return (
            "data.json:10: totalRevenue: 5098000000\n"
            "data.json:11: revenueGrowth: 0.262\n"
            "data.json:12: operatingMargin: 0.321"
        )

    @fence.enforce
    def submit(claim: str, evidence: str) -> dict:
        return {"claim": claim, "status": "ok"}

    search("revenue")
    result = submit(
        claim="test",
        evidence="data.json:10: totalRevenue: 5098000000\ndata.json:11: revenueGrowth: 0.262",
    )
    assert isinstance(result, dict)
    assert result["status"] == "ok"
