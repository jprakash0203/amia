# backend/memory/short_term.py
# In-session working memory — lives in AMIAState.working_memory
from typing import List


def add_to_working_memory(state: dict, agent: str, content: str,
                          max_items: int = 10) -> List[dict]:
    """
    Adds a memory item to the working memory list.
    Trims to max_items to prevent unbounded growth.
    This prevents the LLM context from growing too large over a long session.
    """
    import datetime
    memory = state.get("working_memory", [])
    memory.append({
        "agent": agent,
        "content": content,
        "timestamp": datetime.datetime.utcnow().isoformat()
    })
    # Keep only the most recent max_items entries
    return memory[-max_items:]


def format_working_memory(state: dict) -> str:
    """Formats working memory for injection into an LLM prompt."""
    memory = state.get("working_memory", [])
    if not memory:
        return "No working memory yet."
    lines = [f"[{m['agent']}] {m['content']}" for m in memory]
    return "\n".join(lines)