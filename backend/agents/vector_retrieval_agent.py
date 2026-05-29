import datetime
from backend.llm_config import get_fast_llm
from backend.graph.state import AMIAState
from backend.vector_store.chroma_store import get_chroma_store

class VectorRetrievalAgent:

    def __init__(self):
        self.llm = get_fast_llm()
        self.store = get_chroma_store()

    def run(self, state: AMIAState) -> dict:
        query = state["query"]
        vector_results = []

        # Search documents collection
        doc_results = self.store.search_with_scores(query, k=5)
        for document, score in doc_results:
            # Only include results with score below 1.0 threshold
            # ChromaDB L2 distance: lower = more similar
            # Score < 1.5 is generally relevant
            vector_results.append({
                "document": document.page_content,
                "source": document.metadata.get("source_type", "unknown"),
                "file_name": document.metadata.get("file_name", ""),
                "url": document.metadata.get("url", ""),
                "relevance_score": round(float(score), 3),
                "collection": "documents",
            })

        # Search episodic memory — "Have I answered something similar before?"
        memory_results = self.store.recall_similar_queries(query, k=2)
        for memory in memory_results:
            vector_results.append({
                "document": memory.page_content,
                "source": "episodic_memory",
                "timestamp": memory.metadata.get("timestamp", ""),
                "outcome": memory.metadata.get("outcome", ""),
                "relevance_score": 0.0,
                "collection": "memory",
            })

        return {
            "vector_results": vector_results,
            "agent_trace": [{
                "agent": "vector_retrieval_agent",
                "action": f"ChromaDB search: {len(vector_results)} results",
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "status": "complete",
            }],
        }


def vector_retrieval_agent_node(state: AMIAState) -> dict:
    return VectorRetrievalAgent().run(state)        