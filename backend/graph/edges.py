from backend.graph.state import AMIAState


def route_from_supervisor(state: AMIAState) -> str:
    """After supervisor runs, route to the chosen agent."""
    mapping = {
        "web_search":       "web_search_agent",
        "vector_retrieval": "vector_retrieval_agent",
        "reasoning":        "reasoning_agent",
        "summarize":        "summarizer_agent",
        "critique":         "critique_agent",
        "synthesize":       "synthesizer_agent",
        "FINISH":           "END",
    }
    next_agent = state.get("next_agent", "synthesize")
    return mapping.get(next_agent, "synthesizer_agent")


def route_after_critique(state: AMIAState) -> str:
    """After critique, either retry (back to supervisor) or synthesize."""
    score = state.get("critique_score", 1.0)
    retries = state.get("retry_count", 0)

    if score < 0.7 and retries < 2:
        return "supervisor"
    return "synthesizer_agent"