"""Tests for audit-fence core enforcement logic."""

import asyncio
import inspect
import json

from audit_fence import Fence, ValidationResult, extract_numbers, normalize_number


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


# ============================================================================
# ASYNC SUPPORT
# ============================================================================


def test_async_track():
    """Async search function should record results just like sync."""
    async def _run():
        fence = Fence()

        @fence.track
        async def search(query: str) -> str:
            return f"async result for {query} with enough padding text"

        result = await search("revenue")
        assert result == "async result for revenue with enough padding text"
        assert len(fence.history) == 1
        assert "revenue" in fence.history[0].result_text

    asyncio.run(_run())


def test_async_enforce_accept():
    """Async enforce decorator should pass through when evidence matches."""
    async def _run():
        fence = Fence()

        @fence.track
        async def search(query: str) -> str:
            return "line 42: revenue was $5.1B in FY2025"

        @fence.enforce
        async def submit(claim: str, evidence: str) -> dict:
            return {"claim": claim, "status": "ok"}

        await search("revenue")
        result = await submit(claim="test", evidence="revenue was $5.1B in FY2025")
        assert isinstance(result, dict)
        assert result["status"] == "ok"
        assert len(fence.rejections) == 0

    asyncio.run(_run())


def test_async_enforce_reject_no_search():
    """Async enforce should reject when no search has been done."""
    async def _run():
        fence = Fence()

        @fence.enforce
        async def submit(claim: str, evidence: str) -> dict:
            return {"claim": claim}

        result = await submit(claim="test", evidence="some evidence text that is long enough")
        assert isinstance(result, str)
        assert "ERROR" in result
        assert "No search calls recorded" in result
        assert len(fence.rejections) == 1

    asyncio.run(_run())


def test_async_enforce_reject_mismatch():
    """Async enforce should reject when evidence doesn't match."""
    async def _run():
        fence = Fence()

        @fence.track
        async def search(query: str) -> str:
            return "line 42: revenue was $5.1B in FY2025"

        @fence.enforce
        async def submit(claim: str, evidence: str) -> dict:
            return {"claim": claim}

        await search("revenue")
        result = await submit(claim="test", evidence="fabricated evidence not in any search result")
        assert "ERROR" in result
        assert "does not match" in result

    asyncio.run(_run())


def test_async_mixed_sync_track_async_enforce():
    """Sync track + async enforce should work together."""
    async def _run():
        fence = Fence()

        @fence.track
        def search(query: str) -> str:
            return "line 42: revenue was $5.1B in FY2025"

        @fence.enforce
        async def submit(claim: str, evidence: str) -> dict:
            return {"claim": claim, "status": "ok"}

        search("revenue")
        result = await submit(claim="test", evidence="revenue was $5.1B in FY2025")
        assert isinstance(result, dict)
        assert result["status"] == "ok"

    asyncio.run(_run())


def test_async_track_sync_enforce():
    """Async track + sync enforce should work together."""
    async def _run():
        fence = Fence()

        @fence.track
        async def search(query: str) -> str:
            return "line 42: revenue was $5.1B in FY2025"

        @fence.enforce
        def submit(claim: str, evidence: str) -> dict:
            return {"claim": claim, "status": "ok"}

        await search("revenue")
        result = submit(claim="test", evidence="revenue was $5.1B in FY2025")
        assert isinstance(result, dict)
        assert result["status"] == "ok"

    asyncio.run(_run())


def test_async_enforce_with_params():
    """Async enforce with custom evidence_param and source_text."""
    async def _run():
        report = "The company reported revenue of $5.1B in Q4."
        fence = Fence()

        @fence.track
        async def search(query: str) -> str:
            return "line 1: reported revenue of $5.1B in Q4"

        @fence.enforce(
            evidence_param="grep_output",
            claim_param="claim",
            source_text=report,
        )
        async def submit(claim: str, grep_output: str) -> dict:
            return {"claim": claim, "status": "ok"}

        await search("revenue")
        result = await submit(
            claim="revenue of $5.1B",
            grep_output="reported revenue of $5.1B in Q4",
        )
        assert isinstance(result, dict)
        assert result["status"] == "ok"

    asyncio.run(_run())


def test_async_track_preserves_coroutine_function():
    """@fence.track on async fn should return a coroutine function."""
    fence = Fence()

    @fence.track
    async def search(query: str) -> str:
        return "result"

    assert inspect.iscoroutinefunction(search)
    assert search._fence_role == "search"


def test_async_enforce_preserves_coroutine_function():
    """@fence.enforce on async fn should return a coroutine function."""
    fence = Fence()

    @fence.enforce
    async def submit(claim: str, evidence: str) -> dict:
        return {}

    assert inspect.iscoroutinefunction(submit)
    assert submit._fence_role == "submit"


def test_sync_track_not_coroutine():
    """@fence.track on sync fn should NOT return a coroutine function."""
    fence = Fence()

    @fence.track
    def search(query: str) -> str:
        return "result"

    assert not inspect.iscoroutinefunction(search)


def test_sync_enforce_not_coroutine():
    """@fence.enforce on sync fn should NOT return a coroutine function."""
    fence = Fence()

    @fence.enforce
    def submit(claim: str, evidence: str) -> dict:
        return {}

    assert not inspect.iscoroutinefunction(submit)


def test_async_tools_property():
    """tools property should include both async search and submit."""
    fence = Fence()

    @fence.track
    async def search(query: str) -> str:
        return "result"

    @fence.enforce
    async def submit(claim: str, evidence: str) -> dict:
        return {}

    assert len(fence.tools) == 2
    assert fence.tools[0]._fence_role == "search"
    assert fence.tools[1]._fence_role == "submit"


# ============================================================================
# ENFORCEMENT CONTEXT / METADATA
# ============================================================================


def test_fence_level_context():
    """Fence-level context should be attached to every rejection."""
    fence = Fence(context={"agent": "fundamental", "session": "abc-123"})

    @fence.enforce
    def submit(claim: str, evidence: str) -> dict:
        return {"claim": claim}

    submit(claim="test", evidence="no search performed, this will be rejected")
    assert len(fence.rejections) == 1
    rej = fence.rejections[0]
    assert "context" in rej
    assert rej["context"]["agent"] == "fundamental"
    assert rej["context"]["session"] == "abc-123"


def test_per_tool_context():
    """Per-tool context should be attached to rejections from that tool."""
    fence = Fence()

    @fence.enforce(context={"tool_type": "specialist_claim"})
    def submit(claim: str, evidence: str) -> dict:
        return {"claim": claim}

    submit(claim="test", evidence="no search performed, this will be rejected")
    assert len(fence.rejections) == 1
    rej = fence.rejections[0]
    assert rej["context"]["tool_type"] == "specialist_claim"


def test_context_merge_per_tool_wins():
    """Per-tool context should override fence-level context on key conflict."""
    fence = Fence(context={"agent": "fundamental", "mode": "audit"})

    @fence.enforce(context={"agent": "technical"})
    def submit(claim: str, evidence: str) -> dict:
        return {"claim": claim}

    submit(claim="test", evidence="no search performed, this will be rejected")
    rej = fence.rejections[0]
    assert rej["context"]["agent"] == "technical"  # per-tool wins
    assert rej["context"]["mode"] == "audit"  # fence-level preserved


def test_no_context_key_when_empty():
    """No 'context' key should appear in rejections when no context is set."""
    fence = Fence()

    @fence.enforce
    def submit(claim: str, evidence: str) -> dict:
        return {"claim": claim}

    submit(claim="test", evidence="no search performed, this will be rejected")
    rej = fence.rejections[0]
    assert "context" not in rej


def test_context_in_saved_log(tmp_path):
    """Context should persist when saving rejection log to JSONL."""
    fence = Fence(context={"agent": "macro"})

    @fence.enforce(context={"round": 1})
    def submit(claim: str, evidence: str) -> dict:
        return {"claim": claim}

    submit(claim="test", evidence="no search performed, this will be rejected")

    log_path = tmp_path / "enforcement.jsonl"
    fence.save_log(log_path)

    with open(log_path) as f:
        entries = [json.loads(line) for line in f]
    assert len(entries) == 1
    assert entries[0]["context"]["agent"] == "macro"
    assert entries[0]["context"]["round"] == 1


def test_async_enforce_with_context():
    """Context should work with async enforce decorators."""
    async def _run():
        fence = Fence(context={"session": "s1"})

        @fence.enforce(context={"tool_type": "source_evidence"})
        async def submit(claim: str, evidence: str) -> dict:
            return {"claim": claim}

        result = await submit(claim="test", evidence="no search, will be rejected here")
        assert "ERROR" in result
        rej = fence.rejections[0]
        assert rej["context"]["session"] == "s1"
        assert rej["context"]["tool_type"] == "source_evidence"

    asyncio.run(_run())


# ============================================================================
# NUMBER FORMAT MATCHING
# ============================================================================


class TestNormalizeNumber:
    """Tests for the normalize_number() helper."""

    def test_plain_integer(self):
        assert normalize_number("5098000000") == 5098000000.0

    def test_plain_float(self):
        assert normalize_number("18.923") == 18.923

    def test_comma_separated(self):
        assert normalize_number("5,098,000,000") == 5098000000.0

    def test_suffix_b(self):
        result = normalize_number("5.1B")
        assert result is not None
        assert abs(result - 5.1e9) < 1e3

    def test_suffix_m(self):
        result = normalize_number("320M")
        assert result is not None
        assert abs(result - 320e6) < 1e3

    def test_suffix_k(self):
        result = normalize_number("1.5K")
        assert result is not None
        assert abs(result - 1500.0) < 0.1

    def test_suffix_t(self):
        result = normalize_number("2.3T")
        assert result is not None
        assert abs(result - 2.3e12) < 1e6

    def test_percentage(self):
        result = normalize_number("26.2%")
        assert result is not None
        assert abs(result - 0.262) < 1e-6

    def test_dollar_sign(self):
        result = normalize_number("$5.1B")
        assert result is not None
        assert abs(result - 5.1e9) < 1e3

    def test_empty_string(self):
        assert normalize_number("") is None

    def test_not_a_number(self):
        assert normalize_number("hello") is None

    def test_negative(self):
        result = normalize_number("-3.2%")
        assert result is not None
        assert abs(result - (-0.032)) < 1e-6

    def test_lowercase_suffix(self):
        result = normalize_number("5.1b")
        assert result is not None
        assert abs(result - 5.1e9) < 1e3


class TestExtractNumbers:
    """Tests for the extract_numbers() helper."""

    def test_revenue_string(self):
        nums = extract_numbers("Revenue $5.1B, up 26.2% YoY")
        # Should find 5.1B and 26.2 (the % is outside the match in this regex)
        assert len(nums) >= 2
        # Check that 5.1B was found
        assert any(abs(n - 5.1e9) < 1e3 for n in nums)

    def test_json_number(self):
        nums = extract_numbers('"totalRevenue": 5098000000')
        assert any(abs(n - 5098000000) < 1 for n in nums)

    def test_no_numbers(self):
        assert extract_numbers("no numbers here") == []

    def test_mixed(self):
        nums = extract_numbers("PE ratio 18.9, market cap $320M, growth 5.2%")
        assert len(nums) >= 3


class TestNumberMatchInFence:
    """Tests for number-format fallback matching in _verify_search_match."""

    def test_abbreviation_matches_raw(self):
        """'5.1B' in evidence should match '5098000000' in search results."""
        fence = Fence()

        @fence.track
        def search(query: str) -> str:
            return '"totalRevenue": 5098000000, reported quarterly'

        @fence.enforce
        def submit(claim: str, evidence: str) -> dict:
            return {"claim": claim, "status": "ok"}

        search("revenue")
        # Evidence uses abbreviated form; search has raw number
        result = submit(
            claim="test",
            evidence="totalRevenue was approximately $5.1B reported quarterly",
        )
        assert isinstance(result, dict)
        assert result["status"] == "ok"

    def test_raw_matches_abbreviation(self):
        """Raw number in evidence should match abbreviated form in search."""
        fence = Fence()

        @fence.track
        def search(query: str) -> str:
            return "Revenue $5.1B for the fiscal quarter"

        @fence.enforce
        def submit(claim: str, evidence: str) -> dict:
            return {"claim": claim, "status": "ok"}

        search("revenue")
        result = submit(
            claim="test",
            evidence="Revenue was 5098000000 for the fiscal quarter",
        )
        assert isinstance(result, dict)
        assert result["status"] == "ok"

    def test_no_false_positive_on_unrelated_numbers(self):
        """Numbers that don't match should still be rejected."""
        fence = Fence()

        @fence.track
        def search(query: str) -> str:
            return '"totalRevenue": 5098000000 for fiscal period'

        @fence.enforce
        def submit(claim: str, evidence: str) -> dict:
            return {"claim": claim, "status": "ok"}

        search("revenue")
        # Different number, but some word overlap
        result = submit(
            claim="test",
            evidence="totalRevenue was $999B for fiscal period today",
        )
        # The numbers don't match (999B != 5098000000)
        assert "ERROR" in result

    def test_no_false_positive_numbers_only(self):
        """Pure number match without text overlap should NOT match."""
        fence = Fence()

        @fence.track
        def search(query: str) -> str:
            return "alpha beta: 5098000000"

        @fence.enforce
        def submit(claim: str, evidence: str) -> dict:
            return {"claim": claim, "status": "ok"}

        search("revenue")
        # Same number but completely different text context
        result = submit(
            claim="test",
            evidence="gamma delta: $5.1B something entirely different",
        )
        assert "ERROR" in result

    def test_percentage_match(self):
        """Percentage values should be matchable across formats."""
        fence = Fence()

        @fence.track
        def search(query: str) -> str:
            return '"revenueGrowth": 0.262, reported quarterly growth'

        @fence.enforce
        def submit(claim: str, evidence: str) -> dict:
            return {"claim": claim, "status": "ok"}

        search("growth")
        result = submit(
            claim="test",
            evidence="revenueGrowth was 26.2% reported quarterly growth",
        )
        assert isinstance(result, dict)
        assert result["status"] == "ok"

    def test_exact_match_still_preferred(self):
        """Exact substring match should still work (not broken by number fallback)."""
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


# ============================================================================
# fence.wrap() API
# ============================================================================


def test_wrap_with_glob_patterns():
    """wrap() should match tools by glob patterns on __name__."""
    fence = Fence()

    def search_web(query: str) -> str:
        return f"results for {query} with enough padding text here"

    def fetch_data(url: str) -> str:
        return f"data from {url} with enough padding text here"

    def record_finding(claim: str, evidence: str) -> dict:
        return {"claim": claim, "status": "ok"}

    def unrelated_tool() -> str:
        return "hello"

    protected = fence.wrap(
        [search_web, fetch_data, record_finding, unrelated_tool],
        search=["search_*", "fetch_*"],
        submit=["record_*"],
    )

    assert len(protected) == 4

    # search tools should track
    protected[0]("revenue")
    protected[1]("http://example.com")
    assert len(fence.history) == 2

    # submit tool should enforce
    result = protected[2](
        claim="test",
        evidence="results for revenue with enough padding text here",
    )
    assert isinstance(result, dict)
    assert result["status"] == "ok"

    # unrelated tool passes through unchanged
    assert protected[3] is unrelated_tool
    assert protected[3]() == "hello"


def test_wrap_with_function_refs():
    """wrap() should accept direct function references."""
    fence = Fence()

    def my_search(query: str) -> str:
        return f"found {query} in the database with extra text"

    def my_submit(claim: str, evidence: str) -> dict:
        return {"claim": claim, "status": "ok"}

    protected = fence.wrap(
        [my_search, my_submit],
        search=[my_search],
        submit=[my_submit],
    )

    protected[0]("test query")
    assert len(fence.history) == 1

    result = protected[1](
        claim="test",
        evidence="found test query in the database with extra text",
    )
    assert isinstance(result, dict)
    assert result["status"] == "ok"


def test_wrap_mixed_patterns_and_refs():
    """wrap() should handle mixed glob patterns and function references."""
    fence = Fence()

    def search_web(query: str) -> str:
        return f"web results for {query} with extra padding"

    def grep_files(pattern: str) -> str:
        return f"grep output for {pattern} with extra padding"

    def record_citation(claim: str, evidence: str) -> dict:
        return {"claim": claim, "status": "ok"}

    def write_report(claim: str, evidence: str) -> dict:
        return {"claim": claim, "status": "ok"}

    protected = fence.wrap(
        [search_web, grep_files, record_citation, write_report],
        search=["search_*", grep_files],  # one glob, one ref
        submit=[record_citation, "write_*"],  # one ref, one glob
    )

    # Both search tools should track
    protected[0]("test")
    protected[1]("pattern")
    assert len(fence.history) == 2

    # Both submit tools should enforce
    result = protected[2](
        claim="test",
        evidence="web results for test with extra padding",
    )
    assert isinstance(result, dict)

    result = protected[3](
        claim="test",
        evidence="grep output for pattern with extra padding",
    )
    assert isinstance(result, dict)


def test_wrap_preserves_metadata():
    """wrap() should preserve __name__ and __doc__ on wrapped functions."""
    fence = Fence()

    def search_evidence(query: str) -> str:
        """Search for evidence in the database."""
        return f"result for {query} with extra padding text"

    def record_citation(claim: str, evidence: str) -> dict:
        """Record a verified citation."""
        return {"claim": claim}

    protected = fence.wrap(
        [search_evidence, record_citation],
        search=["search_*"],
        submit=["record_*"],
    )

    assert protected[0].__name__ == "search_evidence"
    assert protected[0].__doc__ == "Search for evidence in the database."
    assert protected[1].__name__ == "record_citation"
    assert protected[1].__doc__ == "Record a verified citation."


def test_wrap_no_double_wrap():
    """wrap() should not re-wrap functions that already have _fence_role."""
    fence = Fence()

    @fence.track
    def already_tracked(query: str) -> str:
        return f"result for {query} with extra text"

    protected = fence.wrap(
        [already_tracked],
        search=["already_*"],
    )

    # Should pass through the already-wrapped function unchanged
    assert protected[0] is already_tracked
    assert protected[0]._fence_role == "search"

    # Calling it should still track (via original decorator)
    protected[0]("test")
    assert len(fence.history) == 1


def test_wrap_passthrough():
    """wrap() should pass through tools that don't match any pattern."""
    fence = Fence()

    def helper_tool() -> str:
        return "helper"

    protected = fence.wrap(
        [helper_tool],
        search=["search_*"],
        submit=["record_*"],
    )

    assert protected[0] is helper_tool


def test_wrap_async_functions():
    """wrap() should correctly wrap async tools."""
    async def _run():
        fence = Fence()

        async def search_async(query: str) -> str:
            return f"async result for {query} with extra padding"

        async def submit_async(claim: str, evidence: str) -> dict:
            return {"claim": claim, "status": "ok"}

        protected = fence.wrap(
            [search_async, submit_async],
            search=["search_*"],
            submit=["submit_*"],
        )

        # Verify they're still coroutine functions
        assert inspect.iscoroutinefunction(protected[0])
        assert inspect.iscoroutinefunction(protected[1])

        # Search should track
        await protected[0]("revenue")
        assert len(fence.history) == 1

        # Submit should enforce
        result = await protected[1](
            claim="test",
            evidence="async result for revenue with extra padding",
        )
        assert isinstance(result, dict)
        assert result["status"] == "ok"

        # Submit without search should fail (reset first)
        fence.reset()
        result = await protected[1](
            claim="test",
            evidence="fabricated evidence that is long enough to pass length check",
        )
        assert "ERROR" in result

    asyncio.run(_run())


def test_wrap_one_search():
    """wrap_one() should wrap a single function as search."""
    fence = Fence()

    def my_search(query: str) -> str:
        return f"found {query} in the knowledge base"

    wrapped = fence.wrap_one(my_search, role="search")

    assert wrapped.__name__ == "my_search"
    assert wrapped._fence_role == "search"

    wrapped("test query")
    assert len(fence.history) == 1
    assert "found test query" in fence.history[0].result_text


def test_wrap_one_submit():
    """wrap_one() should wrap a single function as submit."""
    fence = Fence()

    def my_search(query: str) -> str:
        return f"search result for {query} with padding text"

    def my_submit(claim: str, evidence: str) -> dict:
        return {"claim": claim, "status": "ok"}

    search_fn = fence.wrap_one(my_search, role="search")
    submit_fn = fence.wrap_one(my_submit, role="submit")

    assert submit_fn.__name__ == "my_submit"
    assert submit_fn._fence_role == "submit"

    # Without search, submit should fail
    result = submit_fn(
        claim="test",
        evidence="fabricated evidence that is definitely long enough",
    )
    assert "ERROR" in result

    # After search, matching evidence should pass
    search_fn("test query")
    result = submit_fn(
        claim="test",
        evidence="search result for test query with padding text",
    )
    assert isinstance(result, dict)
    assert result["status"] == "ok"


def test_wrap_one_invalid_role():
    """wrap_one() should raise ValueError for invalid role."""
    fence = Fence()

    def my_fn() -> str:
        return "hello"

    try:
        fence.wrap_one(my_fn, role="invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "invalid" in str(e)


def test_wrap_empty_list():
    """wrap() should handle an empty tool list."""
    fence = Fence()
    result = fence.wrap([], search=["search_*"], submit=["record_*"])
    assert result == []


def test_wrap_order_preserved():
    """wrap() output order should match input order."""
    fence = Fence()

    def alpha() -> str:
        return "a"

    def beta() -> str:
        return "b"

    def gamma() -> str:
        return "c"

    def delta() -> str:
        return "d"

    protected = fence.wrap(
        [alpha, beta, gamma, delta],
        search=["alpha", "gamma"],
    )

    assert protected[0].__name__ == "alpha"
    assert protected[1] is beta  # passthrough
    assert protected[2].__name__ == "gamma"
    assert protected[3] is delta  # passthrough


def test_wrap_evidence_param_override():
    """wrap() evidence_param kwarg should control which param is validated."""
    fence = Fence()

    def my_search(query: str) -> str:
        return f"result for {query} with enough padding text"

    def my_submit(claim: str, grep_output: str) -> dict:
        return {"claim": claim, "status": "ok"}

    protected = fence.wrap(
        [my_search, my_submit],
        search=["my_search"],
        submit=["my_submit"],
        evidence_param="grep_output",
    )

    protected[0]("test")
    result = protected[1](
        claim="test",
        grep_output="result for test with enough padding text",
    )
    assert isinstance(result, dict)
    assert result["status"] == "ok"


# ============================================================================
# track_all MODE
# ============================================================================


def test_track_all_mode():
    """track_all=True should track ALL tools when no patterns are given."""
    fence = Fence(track_all=True)

    def tool_a(x: str) -> str:
        return f"output from a: {x}"

    def tool_b(x: str) -> str:
        return f"output from b: {x}"

    protected = fence.wrap([tool_a, tool_b])

    protected[0]("hello")
    protected[1]("world")

    assert len(fence.history) == 2
    assert "output from a: hello" in fence.history[0].result_text
    assert "output from b: world" in fence.history[1].result_text


def test_track_all_with_explicit_patterns():
    """track_all=True should defer to explicit patterns when provided."""
    fence = Fence(track_all=True)

    def search_fn(q: str) -> str:
        return f"search result: {q}"

    def submit_fn(claim: str, evidence: str) -> dict:
        return {"claim": claim, "status": "ok"}

    def other_fn() -> str:
        return "other"

    # When explicit patterns are given, track_all doesn't force everything
    protected = fence.wrap(
        [search_fn, submit_fn, other_fn],
        search=["search_*"],
        submit=["submit_*"],
    )

    # other_fn should pass through (not tracked) because explicit patterns given
    assert protected[2] is other_fn

    protected[0]("test")
    assert len(fence.history) == 1


# ============================================================================
# validate_output()
# ============================================================================


def test_validate_output_found():
    """validate_output should find quoted passages that match search history."""
    fence = Fence()

    @fence.track
    def search(query: str) -> str:
        return "The company reported revenue of $5.1B in fiscal year 2025."

    search("revenue")

    result = fence.validate_output(
        'The report states "The company reported revenue of $5.1B in fiscal year 2025." as a key finding.'
    )

    assert isinstance(result, ValidationResult)
    assert len(result.found) == 1
    assert len(result.not_found) == 0
    assert result.coverage == 1.0
    assert result.ok is True


def test_validate_output_not_found():
    """validate_output should flag quoted passages that don't match."""
    fence = Fence()

    @fence.track
    def search(query: str) -> str:
        return "The company reported revenue of $5.1B."

    search("revenue")

    result = fence.validate_output(
        'The report claims "Operating margins expanded to 42% this quarter" which is notable.'
    )

    assert len(result.found) == 0
    assert len(result.not_found) == 1
    assert result.coverage == 0.0
    assert result.ok is False


def test_validate_output_partial():
    """validate_output should handle a mix of found and not-found passages."""
    fence = Fence()

    @fence.track
    def search(query: str) -> str:
        return "Revenue was $5.1B in FY2025. Operating margin was 32.1%."

    search("financials")

    result = fence.validate_output(
        'Two findings: "Revenue was $5.1B in FY2025" is supported, '
        'but "Net income doubled year over year" has no source.'
    )

    assert len(result.found) == 1
    assert len(result.not_found) == 1
    assert result.total == 2
    assert result.coverage == 0.5
    assert result.ok is False


def test_validate_output_empty_text():
    """validate_output with no quotes should return ok=True with zero total."""
    fence = Fence()
    result = fence.validate_output("No quoted passages here at all.")
    assert result.total == 0
    assert result.ok is True
    assert result.coverage == 1.0


def test_validate_output_short_quotes_skipped():
    """Quoted passages shorter than 10 chars should be skipped."""
    fence = Fence()

    @fence.track
    def search(query: str) -> str:
        return "some data"

    search("test")

    result = fence.validate_output('He said "yes" and "no" but not much else.')
    # "yes" and "no" are too short to validate
    assert result.total == 0
    assert result.ok is True
