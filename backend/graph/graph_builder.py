from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from backend.graph.state import AMIAState

# Import all agent node functions
from backend.agents.supervisor import supervisor_node
from backend.agents.web_search_agent import web_search_agent_node
from backend.agents.vector_retrieval_agent import vector_retrieval_agent_node
from backend.agents.reasoning_agent import reasoning_agent_node
from backend.agents.summarizer_agent import summarizer_agent_node
from backend.agents.critique_agent import critique_agent_node
from backend.agents.synthesizer_agent import synthesizer_agent_node

# Import routing functions
from backend.graph.edges import route_from_supervisor, route_after_critique


def build_amia_graph():
    graph = StateGraph(AMIAState)

    # ── Add all nodes ───────────────────────────────────────────────
    graph.add_node("supervisor",               supervisor_node)
    graph.add_node("web_search_agent",         web_search_agent_node)
    graph.add_node("vector_retrieval_agent",   vector_retrieval_agent_node)
    graph.add_node("reasoning_agent",          reasoning_agent_node)
    graph.add_node("summarizer_agent",         summarizer_agent_node)
    graph.add_node("critique_agent",           critique_agent_node)
    graph.add_node("synthesizer_agent",        synthesizer_agent_node)

    # ── Entry point ─────────────────────────────────────────────────
    graph.set_entry_point("supervisor")

    # ── Supervisor routes to any agent ──────────────────────────────
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "web_search_agent":         "web_search_agent",
            "vector_retrieval_agent":   "vector_retrieval_agent",
            "reasoning_agent":          "reasoning_agent",
            "summarizer_agent":         "summarizer_agent",
            "critique_agent":           "critique_agent",
            "synthesizer_agent":        "synthesizer_agent",
            "END":                      END,
        },
    )

    # ── Worker agents return to supervisor after each run ───────────
    graph.add_edge("web_search_agent",       "supervisor")
    graph.add_edge("vector_retrieval_agent", "supervisor")
    graph.add_edge("reasoning_agent",        "supervisor")
    graph.add_edge("summarizer_agent",       "supervisor")

    # ── After critique: retry or synthesize ─────────────────────────
    graph.add_conditional_edges(
        "critique_agent",
        route_after_critique,
        {
            "supervisor":        "supervisor",
            "synthesizer_agent": "synthesizer_agent",
        },
    )

    # ── Synthesizer ends the graph ───────────────────────────────────
    graph.add_edge("synthesizer_agent", END)

    # ── Compile with memory checkpointing ───────────────────────────
    # MemorySaver enables conversation continuity across turns
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


if __name__ == "__main__":
    from backend.graph.state import create_initial_state

    print("Building AMIA graph...")
    app = build_amia_graph()
    print("✅ Graph compiled successfully")

    # Test with a simple query
    state = create_initial_state("What is LangGraph?", "test-001")
    config = {"configurable": {"thread_id": "test-001"}}

    print("\nRunning test query: 'What is LangGraph?'")
    for step in app.stream(state, config=config):
        node = list(step.keys())[0]
        updates = step[node]
        for trace in updates.get("agent_trace", []):
            print(f"  → {trace.get('agent', node)}: {trace.get('action', '')[:80]}")

    print("\n✅ Graph ran successfully!")