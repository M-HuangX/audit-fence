"""Tests for multi-agent enforcement: link(), FenceGroup, snapshot/restore, etc."""

import json
import time

from audit_fence import ClaimRecord, Fence, FenceGroup, SearchRecord, create_record_tool
from audit_fence.workflow import reset_claim_ids


# ============================================================================
# NAMED FENCES
# ============================================================================


def test_fence_name():
    """Fence(name='x') should expose name via property."""
    fence = Fence(name="agent_a")
    assert fence.name == "agent_a"


def test_unnamed_fence():
    """Fence() without name should have name=None and source=''."""
    fence = Fence()
    assert fence.name is None

    @fence.track
    def search(query: str) -> str:
        return f"result for {query} with padding text"

    search("test")
    assert fence.history[0].source == ""


def test_search_record_source():
    """Tracked records should get source populated from fence name."""
    fence = Fence(name="fundamental")

    @fence.track
    def search(query: str) -> str:
        return f"result for {query} with enough padding"

    search("AAPL revenue")
    assert len(fence.history) == 1
    assert fence.history[0].source == "fundamental"


# ============================================================================
# MULTI-FENCE LINKING
# ============================================================================


def test_link_basic():
    """B.link(A): evidence from A should be visible to B's enforce."""
    fence_a = Fence(name="worker")
    fence_b = Fence(name="manager")
    fence_b.link(fence_a)

    @fence_a.track
    def search_a(query: str) -> str:
        return f"worker found {query} in the database with context"

    @fence_b.enforce
    def submit_b(claim: str, evidence: str) -> dict:
        return {"claim": claim, "status": "ok"}

    search_a("revenue data")
    # B has no own history, but can cite A's via link
    result = submit_b(
        claim="test",
        evidence="worker found revenue data in the database with context",
    )
    assert isinstance(result, dict)
    assert result["status"] == "ok"


def test_link_transitive():
    """C.link(B), B.link(A): C should see A's history transitively."""
    a = Fence(name="a")
    b = Fence(name="b")
    c = Fence(name="c")
    b.link(a)
    c.link(b)

    @a.track
    def search_a(query: str) -> str:
        return f"a found {query} with detailed analysis results"

    @c.enforce
    def submit_c(claim: str, evidence: str) -> dict:
        return {"claim": claim, "status": "ok"}

    search_a("deep data")
    # C can see A's history through B
    result = submit_c(
        claim="test",
        evidence="a found deep data with detailed analysis results",
    )
    assert isinstance(result, dict)
    assert result["status"] == "ok"


def test_link_cycle_safe():
    """A.link(B), B.link(A): no infinite loop in _collect_history."""
    a = Fence(name="a")
    b = Fence(name="b")
    a.link(b)
    b.link(a)

    @a.track
    def search_a(query: str) -> str:
        return f"a found {query} with substantial context"

    @b.track
    def search_b(query: str) -> str:
        return f"b found {query} with substantial context"

    search_a("test query")
    search_b("other query")

    # Both should be able to collect without infinite loop
    history_a = a._collect_history()
    history_b = b._collect_history()
    assert len(history_a) == 2
    assert len(history_b) == 2


def test_link_chaining():
    """B.link(A).link(C) should return B (chaining)."""
    a = Fence(name="a")
    b = Fence(name="b")
    c = Fence(name="c")

    result = b.link(a).link(c)
    assert result is b
    # b should have both a and c as upstream
    assert len(b._upstream) == 2


def test_link_multiple_args():
    """B.link(A, C) should add both in one call."""
    a = Fence(name="a")
    b = Fence(name="b")
    c = Fence(name="c")

    b.link(a, c)
    assert len(b._upstream) == 2
    assert a in b._upstream
    assert c in b._upstream


def test_link_duplicate_ignored():
    """B.link(A), B.link(A): only one edge should exist."""
    a = Fence(name="a")
    b = Fence(name="b")

    b.link(a)
    b.link(a)
    assert len(b._upstream) == 1


def test_link_type_error():
    """B.link('not a fence') should raise TypeError."""
    b = Fence(name="b")

    try:
        b.link("not a fence")
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "Fence instances" in str(e)
        assert "str" in str(e)


def test_link_no_upstream_backward_compat():
    """Single fence with no links should work exactly as before."""
    fence = Fence()

    @fence.track
    def search(query: str) -> str:
        return f"result for {query} with enough context padding"

    @fence.enforce
    def submit(claim: str, evidence: str) -> dict:
        return {"claim": claim, "status": "ok"}

    search("revenue")
    result = submit(
        claim="test",
        evidence="result for revenue with enough context padding",
    )
    assert isinstance(result, dict)
    assert result["status"] == "ok"


def test_link_history_window_per_fence():
    """Each fence in DAG should contribute its own history_window records."""
    a = Fence(name="a", history_window=2)
    b = Fence(name="b", history_window=20)
    b.link(a)

    @a.track
    def search_a(query: str) -> str:
        return f"a result for {query} with padding text"

    # Add 5 records to A — but window is 2, so only last 2 contribute
    for i in range(5):
        search_a(f"query_{i}")

    assert len(a.history) == 5  # all 5 stored
    collected = b._collect_history()
    # A contributes 2 (window), B contributes 0 (empty)
    assert len(collected) == 2
    # The last 2 are query_3 and query_4
    assert "query_3" in collected[0].query or "query_3" in collected[0].result_text
    assert "query_4" in collected[1].query or "query_4" in collected[1].result_text


# ============================================================================
# FENCE GROUP
# ============================================================================


def test_group_create():
    """group.create() should create named fences."""
    group = FenceGroup()
    fund = group.create("fundamental")
    tech = group.create("technical")

    assert fund.name == "fundamental"
    assert tech.name == "technical"
    assert len(group.fences) == 2


def test_group_access():
    """group['name'] should return the fence."""
    group = FenceGroup()
    fund = group.create("fundamental")

    assert group["fundamental"] is fund


def test_group_get():
    """group.get() should return fence or default."""
    group = FenceGroup()
    group.create("a")

    assert group.get("a") is not None
    assert group.get("nonexistent") is None
    assert group.get("nonexistent", "default") == "default"


def test_group_contains():
    """'name' in group should work."""
    group = FenceGroup()
    group.create("alpha")

    assert "alpha" in group
    assert "beta" not in group


def test_group_duplicate_name_error():
    """Creating a fence with a duplicate name should raise ValueError."""
    group = FenceGroup()
    group.create("alpha")

    try:
        group.create("alpha")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "alpha" in str(e)


def test_group_all_rejections():
    """all_rejections should return all rejections sorted by timestamp."""
    group = FenceGroup()
    fence_a = group.create("a")
    fence_b = group.create("b")

    @fence_a.enforce
    def submit_a(claim: str, evidence: str) -> dict:
        return {"claim": claim}

    @fence_b.enforce
    def submit_b(claim: str, evidence: str) -> dict:
        return {"claim": claim}

    submit_a(claim="test", evidence="no search performed, will be rejected")
    time.sleep(0.01)  # ensure different timestamps
    submit_b(claim="test", evidence="no search performed, will be rejected")

    all_rej = group.all_rejections
    assert len(all_rej) == 2
    # Should be sorted by timestamp
    assert all_rej[0]["timestamp"] <= all_rej[1]["timestamp"]


def test_group_all_history():
    """all_history should return all records sorted by timestamp."""
    group = FenceGroup()
    fence_a = group.create("a")
    fence_b = group.create("b")

    @fence_a.track
    def search_a(query: str) -> str:
        return f"a result for {query}"

    @fence_b.track
    def search_b(query: str) -> str:
        return f"b result for {query}"

    search_a("first")
    time.sleep(0.01)
    search_b("second")

    all_hist = group.all_history
    assert len(all_hist) == 2
    assert all_hist[0].timestamp <= all_hist[1].timestamp


def test_group_reset():
    """group.reset() should clear all fences."""
    group = FenceGroup()
    fence_a = group.create("a")
    fence_b = group.create("b")

    @fence_a.track
    def search_a(query: str) -> str:
        return f"result for {query}"

    @fence_b.track
    def search_b(query: str) -> str:
        return f"result for {query}"

    search_a("test")
    search_b("test")
    assert len(fence_a.history) == 1
    assert len(fence_b.history) == 1

    group.reset()
    assert len(fence_a.history) == 0
    assert len(fence_b.history) == 0


def test_group_save_log(tmp_path):
    """group.save_log() should save all fences' rejections."""
    group = FenceGroup()
    fence_a = group.create("a")
    fence_b = group.create("b")

    @fence_a.enforce
    def submit_a(claim: str, evidence: str) -> dict:
        return {"claim": claim}

    @fence_b.enforce
    def submit_b(claim: str, evidence: str) -> dict:
        return {"claim": claim}

    submit_a(claim="test", evidence="no search, rejected by fence a here")
    submit_b(claim="test", evidence="no search, rejected by fence b here")

    log_path = tmp_path / "all_rejections.jsonl"
    group.save_log(log_path)

    with open(log_path) as f:
        entries = [json.loads(line) for line in f]
    assert len(entries) == 2


# ============================================================================
# SERIALIZATION
# ============================================================================


def test_fence_snapshot_restore_roundtrip():
    """Data should survive serialize/deserialize roundtrip."""
    fence = Fence(name="test_fence", min_evidence_length=30, history_window=10)

    @fence.track
    def search(query: str) -> str:
        return f"result for {query} with enough padding text"

    search("AAPL revenue")
    search("MSFT earnings")

    # Take snapshot
    data = fence.snapshot()

    # Verify JSON-serializable
    json_str = json.dumps(data)
    data_back = json.loads(json_str)

    # Restore
    restored = Fence.restore(data_back)
    assert restored.name == "test_fence"
    assert restored._min_evidence_length == 30
    assert restored._history_window == 10
    assert len(restored.history) == 2
    assert restored.history[0].source == "test_fence"
    assert "AAPL" in restored.history[0].query


def test_restore_with_rejections():
    """Rejections should survive snapshot/restore."""
    fence = Fence(name="rejector")

    @fence.enforce
    def submit(claim: str, evidence: str) -> dict:
        return {"claim": claim}

    submit(claim="test", evidence="no search performed, this will be rejected")
    assert len(fence.rejections) == 1

    data = fence.snapshot()
    restored = Fence.restore(data)
    assert len(restored.rejections) == 1
    assert restored.rejections[0]["tool"] == "submit"


def test_group_snapshot_restore_with_links():
    """FenceGroup snapshot should preserve links across serialize/deserialize."""
    group = FenceGroup()
    worker = group.create("worker")
    manager = group.create("manager")
    manager.link(worker)

    @worker.track
    def search_w(query: str) -> str:
        return f"worker found {query} with detailed evidence text"

    search_w("important data")

    # Snapshot
    data = group.snapshot()
    json_str = json.dumps(data)
    data_back = json.loads(json_str)

    # Restore
    restored = FenceGroup.restore(data_back)
    assert "worker" in restored
    assert "manager" in restored

    # Check links are restored
    assert len(restored["manager"]._upstream) == 1
    assert restored["manager"]._upstream[0] is restored["worker"]

    # Check history is restored
    assert len(restored["worker"].history) == 1

    # Manager should be able to see worker's history
    collected = restored["manager"]._collect_history()
    assert len(collected) == 1
    assert "important data" in collected[0].result_text


def test_snapshot_config_preserved():
    """Snapshot should preserve all config including history_limit and track_all."""
    fence = Fence(
        name="configured",
        min_evidence_length=15,
        history_window=5,
        history_limit=100,
        context={"agent": "test"},
        track_all=True,
    )

    data = fence.snapshot()
    restored = Fence.restore(data)

    assert restored._min_evidence_length == 15
    assert restored._history_window == 5
    assert restored._history_limit == 100
    assert restored._context == {"agent": "test"}
    assert restored._track_all is True


# ============================================================================
# INJECT / DROP_LAST
# ============================================================================


def test_inject_record():
    """inject() should add a record visible to enforce."""
    fence = Fence(name="manual")

    record = SearchRecord(
        query="manual entry",
        result_text="Human-approved evidence: revenue was $5.1B in FY2025",
        source="human",
    )
    fence.inject(record)

    assert len(fence.history) == 1
    assert fence.history[0].source == "human"

    @fence.enforce
    def submit(claim: str, evidence: str) -> dict:
        return {"claim": claim, "status": "ok"}

    result = submit(
        claim="test",
        evidence="Human-approved evidence: revenue was $5.1B in FY2025",
    )
    assert isinstance(result, dict)
    assert result["status"] == "ok"


def test_drop_last():
    """drop_last(n) should remove last n history entries."""
    fence = Fence()

    @fence.track
    def search(query: str) -> str:
        return f"result for {query}"

    search("alpha")
    search("beta")
    search("gamma")
    assert len(fence.history) == 3

    fence.drop_last(2)
    assert len(fence.history) == 1
    assert "alpha" in fence.history[0].result_text


def test_drop_last_more_than_available():
    """drop_last(n) with n > len(history) should clear safely."""
    fence = Fence()

    @fence.track
    def search(query: str) -> str:
        return f"result for {query}"

    search("alpha")
    assert len(fence.history) == 1

    fence.drop_last(100)
    assert len(fence.history) == 0


def test_drop_last_default():
    """drop_last() with no args should remove exactly 1 entry."""
    fence = Fence()

    @fence.track
    def search(query: str) -> str:
        return f"result for {query}"

    search("alpha")
    search("beta")
    assert len(fence.history) == 2

    fence.drop_last()
    assert len(fence.history) == 1
    assert "alpha" in fence.history[0].result_text


# ============================================================================
# HISTORY LIMIT
# ============================================================================


def test_history_limit_trims():
    """history_limit should trim oldest records when exceeded."""
    fence = Fence(history_limit=3)

    @fence.track
    def search(query: str) -> str:
        return f"result for {query}"

    for i in range(5):
        search(f"query_{i}")

    # Only last 3 should remain
    assert len(fence.history) == 3
    assert "query_2" in fence.history[0].result_text
    assert "query_3" in fence.history[1].result_text
    assert "query_4" in fence.history[2].result_text


def test_history_limit_none():
    """Default history_limit=None should keep all records."""
    fence = Fence()

    @fence.track
    def search(query: str) -> str:
        return f"result for {query}"

    for i in range(100):
        search(f"query_{i}")

    assert len(fence.history) == 100


def test_history_limit_with_inject():
    """history_limit should also apply when using inject()."""
    fence = Fence(history_limit=2)

    for i in range(3):
        fence.inject(SearchRecord(
            query=f"injected_{i}",
            result_text=f"injected result {i}",
        ))

    assert len(fence.history) == 2
    assert "injected result 1" == fence.history[0].result_text
    assert "injected result 2" == fence.history[1].result_text


def test_history_limit_with_wrap():
    """history_limit should trim records added via wrap()-tracked tools."""
    fence = Fence(history_limit=2)

    def my_search(query: str) -> str:
        return f"wrapped result for {query} with padding"

    protected = fence.wrap([my_search], search=["*"])

    protected[0]("alpha")
    protected[0]("beta")
    protected[0]("gamma")

    assert len(fence.history) == 2
    assert "beta" in fence.history[0].result_text
    assert "gamma" in fence.history[1].result_text


# ============================================================================
# INTEGRATION: wrap() + link()
# ============================================================================


def test_wrap_with_linked_fences():
    """wrap() on linked fences: evidence should flow across links."""
    worker = Fence(name="worker")
    manager = Fence(name="manager")
    manager.link(worker)

    def search_tool(query: str) -> str:
        return f"database result for {query} with full context"

    def submit_tool(claim: str, evidence: str) -> dict:
        return {"claim": claim, "status": "ok"}

    worker_tools = worker.wrap([search_tool], search=["*"])
    manager_tools = manager.wrap([submit_tool], submit=["*"])

    # Worker searches
    worker_tools[0]("revenue analysis")

    # Manager submits — should pass because manager.link(worker)
    result = manager_tools[0](
        claim="test",
        evidence="database result for revenue analysis with full context",
    )
    assert isinstance(result, dict)
    assert result["status"] == "ok"


def test_validate_output_with_linked_fences():
    """validate_output should see upstream history via links."""
    worker = Fence(name="worker")
    manager = Fence(name="manager")
    manager.link(worker)

    @worker.track
    def search_w(query: str) -> str:
        return "The company reported revenue of $5.1B in fiscal year 2025."

    search_w("revenue")

    # Manager's validate_output should see worker's history
    result = manager.validate_output(
        'The report states "The company reported revenue of $5.1B in fiscal year 2025." as a key finding.'
    )

    assert result.ok is True
    assert len(result.found) == 1


def test_wrap_submit_no_own_history_but_upstream():
    """Submit via wrap should pass when only upstream has history (not self)."""
    upstream = Fence(name="source")
    downstream = Fence(name="consumer")
    downstream.link(upstream)

    def search_fn(query: str) -> str:
        return f"found {query} evidence in raw data source"

    def submit_fn(claim: str, evidence: str) -> dict:
        return {"claim": claim, "status": "ok"}

    upstream_tools = upstream.wrap([search_fn], search=["*"])
    downstream_tools = downstream.wrap([submit_fn], submit=["*"])

    # Only upstream has history
    upstream_tools[0]("important fact")
    assert len(upstream.history) == 1
    assert len(downstream.history) == 0

    # Downstream submit should still work via link
    result = downstream_tools[0](
        claim="test",
        evidence="found important fact evidence in raw data source",
    )
    assert isinstance(result, dict)
    assert result["status"] == "ok"


# ============================================================================
# MULTI-FENCE EDGE CASES
# ============================================================================


def test_link_reset_upstream_clears_evidence():
    """Resetting an upstream fence should remove its evidence from downstream."""
    worker = Fence(name="worker")
    manager = Fence(name="manager")
    manager.link(worker)

    @worker.track
    def search_w(query: str) -> str:
        return f"worker found {query} with detailed context"

    @manager.enforce
    def submit_m(claim: str, evidence: str) -> dict:
        return {"claim": claim, "status": "ok"}

    search_w("revenue data")

    # Works before reset
    result = submit_m(
        claim="test",
        evidence="worker found revenue data with detailed context",
    )
    assert isinstance(result, dict)

    # Reset worker
    worker.reset()

    # Now manager should fail
    result = submit_m(
        claim="test",
        evidence="worker found revenue data with detailed context",
    )
    assert "ERROR" in result


def test_link_diamond_topology():
    """Diamond: D.link(B, C), B.link(A), C.link(A) — A visited once."""
    a = Fence(name="a")
    b = Fence(name="b")
    c = Fence(name="c")
    d = Fence(name="d")

    b.link(a)
    c.link(a)
    d.link(b, c)

    @a.track
    def search_a(query: str) -> str:
        return f"a found {query} with context"

    search_a("data")

    # D collects from D, B, C, A — but A only once
    collected = d._collect_history()
    assert len(collected) == 1  # only A's one record, visited once

    @d.enforce
    def submit_d(claim: str, evidence: str) -> dict:
        return {"claim": claim, "status": "ok"}

    result = submit_d(claim="test", evidence="a found data with context")
    assert isinstance(result, dict)
    assert result["status"] == "ok"


def test_source_field_in_collected_history():
    """Source field should identify which fence produced each record."""
    fund = Fence(name="fundamental")
    tech = Fence(name="technical")
    core = Fence(name="core")
    core.link(fund, tech)

    @fund.track
    def search_fund(query: str) -> str:
        return f"fundamental: {query} analysis"

    @tech.track
    def search_tech(query: str) -> str:
        return f"technical: {query} indicators"

    search_fund("AAPL")
    search_tech("AAPL")

    collected = core._collect_history()
    sources = {r.source for r in collected}
    assert "fundamental" in sources
    assert "technical" in sources


def test_enforce_decorator_with_linked_fence():
    """@fence.enforce should work with linked fences (not just wrap)."""
    worker = Fence(name="worker")
    manager = Fence(name="manager")
    manager.link(worker)

    @worker.track
    def search_w(query: str) -> str:
        return f"worker discovered {query} in dataset analysis"

    @manager.enforce
    def submit_m(claim: str, evidence: str) -> dict:
        return {"claim": claim, "status": "ok"}

    search_w("key insight")

    result = submit_m(
        claim="test",
        evidence="worker discovered key insight in dataset analysis",
    )
    assert isinstance(result, dict)
    assert result["status"] == "ok"
    assert len(manager.rejections) == 0


# ============================================================================
# ALL_CLAIMS
# ============================================================================


def test_group_all_claims():
    """all_claims should return claims from all fences sorted by timestamp."""
    reset_claim_ids()
    group = FenceGroup()
    fence_a = group.create("a")
    fence_b = group.create("b")

    @fence_a.track
    def search_a(query: str) -> str:
        return "a found data in the search results with context"

    @fence_b.track
    def search_b(query: str) -> str:
        return "b found data in the search results with context"

    rec_a = create_record_tool(fence_a, name="rec_a", require_claim_in_document=False)
    rec_b = create_record_tool(fence_b, name="rec_b", require_claim_in_document=False)

    search_a("query")
    rec_a(claim="Claim A", claim_in_document="a", evidence="a found data in the search results with context")

    search_b("query")
    rec_b(claim="Claim B", claim_in_document="b", evidence="b found data in the search results with context")

    all_c = group.all_claims
    assert len(all_c) == 2
    assert all_c[0].timestamp <= all_c[1].timestamp


def test_group_all_claims_empty():
    """all_claims on empty group should return []."""
    group = FenceGroup()
    group.create("empty")
    assert group.all_claims == []


# ============================================================================
# TRACE_CHAIN — EVIDENCE CHAIN TRAVERSAL
# ============================================================================


def _build_two_level_group():
    """Helper: build a group with R1 and R2a fences and linked claims."""
    reset_claim_ids()
    group = FenceGroup()
    r1 = group.create("r1")
    r2a = group.create("r2a")

    @r1.track
    def search_r1(query: str) -> str:
        return "tools/fundamental.json:18: pe_ratio: 18.923 in output"

    @r2a.track
    def search_r2a(query: str) -> str:
        return "specialist_outputs/fund.md:45: P/E ratio of 18.9x"

    # R1 records a claim
    rec_r1 = create_record_tool(
        r1, name="record_r1", require_claim_in_document=False,
        extra_fields=["verdict"],
    )
    search_r1("pe_ratio")
    r1_claim = rec_r1(
        claim="P/E ratio of 18.9x",
        claim_in_document="pe_ratio",
        evidence="tools/fundamental.json:18: pe_ratio: 18.923 in output",
        verdict="found",
    )

    # R2a records a claim that links back to R1
    def link_to_r1(record: ClaimRecord) -> ClaimRecord:
        record.upstream_id = r1_claim.id
        record.upstream_fence = "r1"
        return record

    rec_r2a = create_record_tool(
        r2a, name="record_r2a", require_claim_in_document=False,
        enrich=link_to_r1,
    )
    search_r2a("P/E")
    r2a_claim = rec_r2a(
        claim="Report states P/E of 18.9x",
        claim_in_document="P/E",
        evidence="specialist_outputs/fund.md:45: P/E ratio of 18.9x",
    )

    return group, r1_claim, r2a_claim


def test_trace_chain_two_levels():
    """trace_chain should follow one upstream link."""
    group, r1_claim, r2a_claim = _build_two_level_group()

    chain = group.trace_chain(r2a_claim)
    assert len(chain) == 2
    assert chain[0] is r2a_claim
    assert chain[1].id == r1_claim.id
    assert chain[1].claim == "P/E ratio of 18.9x"


def test_trace_chain_root():
    """trace_chain on a root claim (no upstream) should return [claim]."""
    group, r1_claim, _ = _build_two_level_group()

    chain = group.trace_chain(r1_claim)
    assert len(chain) == 1
    assert chain[0] is r1_claim


def test_trace_chain_three_levels():
    """trace_chain should traverse multi-hop chains."""
    reset_claim_ids()
    group = FenceGroup()
    source = group.create("source")
    mid = group.create("mid")
    final = group.create("final")

    @source.track
    def search_s(q: str) -> str:
        return "raw tool output data found in the source"

    @mid.track
    def search_m(q: str) -> str:
        return "intermediate analysis output data with context"

    @final.track
    def search_f(q: str) -> str:
        return "report text with final claim and evidence data"

    # Source claim
    rec_s = create_record_tool(source, name="rec_s", require_claim_in_document=False)
    search_s("data")
    source_claim = rec_s(
        claim="Source fact",
        claim_in_document="data",
        evidence="raw tool output data found in the source",
    )

    # Mid claim → links to source
    rec_m = create_record_tool(
        mid, name="rec_m", require_claim_in_document=False,
        enrich=lambda r: _set_upstream(r, source_claim.id, "source"),
    )
    search_m("analysis")
    mid_claim = rec_m(
        claim="Intermediate finding",
        claim_in_document="analysis",
        evidence="intermediate analysis output data with context",
    )

    # Final claim → links to mid
    rec_f = create_record_tool(
        final, name="rec_f", require_claim_in_document=False,
        enrich=lambda r: _set_upstream(r, mid_claim.id, "mid"),
    )
    search_f("report")
    final_claim = rec_f(
        claim="Report conclusion",
        claim_in_document="report",
        evidence="report text with final claim and evidence data",
    )

    chain = group.trace_chain(final_claim)
    assert len(chain) == 3
    assert chain[0].claim == "Report conclusion"
    assert chain[1].claim == "Intermediate finding"
    assert chain[2].claim == "Source fact"


def _set_upstream(record: ClaimRecord, uid: int, fence_name: str) -> ClaimRecord:
    """Test helper: set upstream link on a ClaimRecord."""
    record.upstream_id = uid
    record.upstream_fence = fence_name
    return record


def test_trace_chain_missing_fence():
    """trace_chain should stop when upstream fence doesn't exist."""
    reset_claim_ids()
    group = FenceGroup()
    f = group.create("only")

    claim = ClaimRecord(
        claim="orphan",
        claim_in_document="o",
        evidence="e",
        upstream_id=99,
        upstream_fence="nonexistent",
    )
    f._claims.append(claim)

    chain = group.trace_chain(claim)
    assert len(chain) == 1  # stops at orphan


def test_trace_chain_missing_claim_id():
    """trace_chain should stop when upstream claim ID doesn't exist."""
    reset_claim_ids()
    group = FenceGroup()
    f = group.create("only")

    @f.track
    def search(q: str) -> str:
        return "some search results with evidence data here"

    rec = create_record_tool(f, name="rec", require_claim_in_document=False)
    search("test")
    claim = rec(
        claim="lonely",
        claim_in_document="search",
        evidence="some search results with evidence data here",
    )
    # Manually set upstream to non-existent ID
    claim.upstream_id = 999
    claim.upstream_fence = "only"

    chain = group.trace_chain(claim)
    assert len(chain) == 1  # stops because ID 999 doesn't exist


def test_trace_chain_cycle_safe():
    """trace_chain should not loop if claims form a cycle."""
    reset_claim_ids()
    group = FenceGroup()
    f = group.create("cycle")

    c1 = ClaimRecord(claim="A", claim_in_document="a", evidence="e")
    c2 = ClaimRecord(claim="B", claim_in_document="b", evidence="e")
    c1.upstream_id = c2.id
    c1.upstream_fence = "cycle"
    c2.upstream_id = c1.id
    c2.upstream_fence = "cycle"

    f._claims.extend([c1, c2])

    chain = group.trace_chain(c1)
    # Should visit c1 → c2 then stop (c1 already visited)
    assert len(chain) == 2
    assert chain[0].claim == "A"
    assert chain[1].claim == "B"
