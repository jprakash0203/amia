"""
AMIA's shared state — the central data structure that flows through
every node in the graph. Every agent reads from and writes to this.

Think of this as the "order ticket" in a restaurant kitchen.
Each station (agent) reads the ticket, does its work, and updates it.
"""

# ── Imports explained ─────────────────────────────────────────────
from typing import (
    TypedDict,      # Creates a dict with named, typed keys
    List,           # A list of items: List[str] = ["a", "b"]
    Optional,       # Value can be the declared type OR None
    Annotated,      # Adds metadata to a type (we add the reducer)
)
import operator     # operator.add is the "append" merge strategy


class AMIAState(TypedDict):
    """
    Every field here is a piece of information that agents either
    need (to do their work) or produce (as output).

    Fields with Annotated[List[X], operator.add]:
        These are list fields where parallel agents APPEND their results
        instead of overwriting each other.

    Fields with Optional[X]:
        These start as None and are filled in later by specific agents.
    """

    # ── INPUT ──────────────────────────────────────────────────────
    # Set once at the beginning; never changed after that.
    query: str
    # The original research question from the user.
    # Example: "What are the latest advances in quantum computing?"

    session_id: str
    # Unique identifier for this research session.
    # Used for conversation continuity and LangSmith tracing.

    # ── PLANNING ───────────────────────────────────────────────────
    plan: List[str]
    # The supervisor breaks the query into sub-tasks.
    # Example: ["search web for recent papers", "check vector DB for background"]

    next_agent: Optional[str]
    # Set by the supervisor to tell the graph which agent to run next.
    # Example: "web_search", "vector_retrieval", "reasoning", "FINISH"

    # ── AGENT OUTPUTS ──────────────────────────────────────────────
    # Each agent writes its results to its designated field.
    # Annotated[List[dict], operator.add] means:
    #   "This is a list of dicts. When agents run in parallel,
    #    APPEND new items to the list instead of replacing it."

    web_results: Annotated[List[dict], operator.add]
    # Results from the web search agent.
    # Each dict: {"title": "...", "url": "...", "content": "...", "score": 0.9}

    vector_results: Annotated[List[dict], operator.add]
    # Results from ChromaDB vector search.
    # Each dict: {"document": "...", "source": "...", "relevance_score": 0.85}

    reasoning_trace: Annotated[List[str], operator.add]
    # Step-by-step reasoning from the ReAct agent.
    # Each string: "Thought: I need to verify... Action: search_web..."

    summaries: Annotated[List[str], operator.add]
    # Condensed summaries from the summarizer agent.

    # ── MEMORY ─────────────────────────────────────────────────────
    working_memory: List[dict]
    # In-session context that grows as agents work.
    # Trimmed to last N items to prevent unbounded growth.

    recalled_memories: Annotated[List[dict], operator.add]
    # Memories from past sessions, recalled from ChromaDB.

    # ── QUALITY CONTROL ────────────────────────────────────────────
    critique_score: Optional[float]
    # Score from 0.0 to 1.0 assigned by the critique agent.
    # Below 0.7 → retry loop. Above 0.7 → proceed to synthesis.

    critique_feedback: Optional[str]
    # Specific feedback explaining the score.
    # Example: "Missing recent data on quantum error correction."

    retry_count: int
    # How many times we've retried due to low critique scores.
    # Max 2 retries → then proceed regardless to avoid infinite loops.

    # ── OUTPUT ─────────────────────────────────────────────────────
    final_response: Optional[str]
    # The synthesized, formatted, cited final answer.

    citations: Annotated[List[dict], operator.add]
    # Sources used in the final response.
    # Each dict: {"type": "web", "title": "...", "url": "..."}

    # ── SAFETY ─────────────────────────────────────────────────────
    guardrail_flags: Annotated[List[str], operator.add]
    # Any safety/compliance issues detected.
    # Example: ["GDPR:email_detected", "HIPAA:ssn_detected"]

    injection_detected: bool
    # True if prompt injection was detected in the input.

    # ── OBSERVABILITY ──────────────────────────────────────────────
    agent_trace: Annotated[List[dict], operator.add]
    # Log of every agent action — displayed in the React frontend.
    # Each dict: {"agent": "...", "action": "...", "timestamp": "...", "status": "..."}

    token_usage: Annotated[List[dict], operator.add]
    # Cost tracking per agent call.
    # Each dict: {"agent": "...", "model": "...", "input_tokens": N, "cost_usd": 0.003}

    error: Optional[str]
    # If any agent fails, the error message is stored here.


def create_initial_state(query: str, session_id: str = "default") -> AMIAState:
    """
    Creates a fresh state for a new research query.

    Called ONCE at the start of each research request.
    All list fields start empty; agents will populate them.
    All Optional fields start as None.
    """
    return AMIAState(
        query=query,
        session_id=session_id,
        plan=[],
        next_agent=None,
        web_results=[],
        vector_results=[],
        reasoning_trace=[],
        summaries=[],
        working_memory=[],
        recalled_memories=[],
        critique_score=None,
        critique_feedback=None,
        retry_count=0,
        final_response=None,
        citations=[],
        guardrail_flags=[],
        injection_detected=False,
        agent_trace=[],
        token_usage=[],
        error=None,
    )


# ── Test this file directly ────────────────────────────────────────
if __name__ == "__main__":
    state = create_initial_state("What is RAG?", "test-001")
    print(f"State created with {len(state)} fields")
    print(f"Query: {state['query']}")
    print(f"Session: {state['session_id']}")
    print(f"Web results: {state['web_results']}")  # → []
    print(f"Final response: {state['final_response']}")  # → None
    print("✅ State definition working")