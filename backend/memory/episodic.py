import datetime
from typing import List
from langchain_core.documents import Document
from backend.vector_store.chroma_store import get_chroma_store


def store_episode(
    query: str,
    agent_trace: List[dict],
    final_response: str,
    critique_score: float = 1.0,
) -> str:

    if critique_score < 0.7:
            print(f"  Skipping episodic memory storage (score {critique_score:.2f} < 0.7)")
            return ""

    store = get_chroma_store()

    # Summarise the agent trace into a brief description
    agents_used = list({t.get("agent", "") for t in agent_trace})
    trace_summary = f"Agents used: {', '.join(agents_used)}"

    doc = Document(
        page_content=(
            f"Query: {query}\n"
            f"Approach: {trace_summary}\n"
            f"Response preview: {final_response[:300]}"
        ),
        metadata={
            "original_query": query,
            "critique_score": critique_score,
            "agents_used": str(agents_used),
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "outcome": "success",
        },
    )

    ids = store.memory.add_documents([doc])
    print(f"  Stored episodic memory (score {critique_score:.2f})")
    return ids[0]


def recall_similar_episodes(query: str, k: int = 3) -> List[Document]:
    """
    Finds past research sessions similar to the current query.

    The reasoning agent calls this to check:
    'Have I researched something like this before?
     What approach worked? What sources were useful?'
    """
    return get_chroma_store().recall_similar_queries(query, k=k)