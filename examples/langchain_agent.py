"""Using audit-fence with LangGraph / LangChain agents.

audit-fence decorators compose with LangChain's @tool decorator.
The enforcement is transparent to the agent framework — tools just
return "ERROR: ..." when validation fails, and the ReAct agent retries.

Requirements:
    pip install audit-fence langchain-core langgraph
"""

import json
import subprocess

from audit_fence import Fence

# Uncomment when langchain is installed:
# from langchain_core.tools import tool

fence = Fence()


# Stack decorators: @tool (outer) wraps @fence.track (inner)
# @tool
@fence.track
def search_evidence(query: str, path: str = "traces/") -> str:
    """Search trace files for evidence. Must be called before record_citation."""
    try:
        result = subprocess.run(
            ["rg", "-n", "-i", "--max-count", "20", query, path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() or f"No matches for '{query}' in {path}"
    except FileNotFoundError:
        return f"ripgrep (rg) not found — install it or use a different search backend"


# @tool
@fence.enforce
def record_citation(claim: str, evidence: str, source: str = "") -> str:
    """Record a verified citation. `evidence` must match a recent search_evidence result."""
    return json.dumps({
        "claim": claim,
        "evidence": evidence,
        "source": source,
        "status": "recorded",
    })


# To use with LangGraph:
#
#   from langgraph.prebuilt import create_react_agent
#   from langchain_openai import ChatOpenAI
#
#   llm = ChatOpenAI(model="gpt-4o")
#   agent = create_react_agent(llm, [search_evidence, record_citation])
#
#   result = await agent.ainvoke({
#       "messages": [HumanMessage(content="Audit this report: ...")]
#   })
#
# The agent will:
# 1. Call search_evidence("revenue") → gets results, tracked by fence
# 2. Call record_citation(claim="...", evidence="<paste from search>")
#    → fence validates evidence matches search history
#    → if valid: returns JSON with status "recorded"
#    → if invalid: returns "ERROR: ..." → agent sees error, retries with correct evidence
