# backend/memory/long_term.py
# Cross-session persistent memory stored in ChromaDB
from langchain_core.documents import Document
from backend.vector_store.chroma_store import get_chroma_store
import datetime


def remember(user_id: str, fact: str, category: str = "general"):
    """
    Stores a fact about a user or topic for future sessions.
    Example: remember("user-1", "Prefers concise summaries", "preference")
    """
    store = get_chroma_store()
    doc = Document(
        page_content=fact,
        metadata={
            "user_id": user_id,
            "category": category,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "type": "long_term_memory"
        }
    )
    return store.memory_store.add_documents([doc])


def recall(user_id: str, context: str, k: int = 3) -> list:
    """
    Retrieves memories relevant to the current context for a user.
    """
    store = get_chroma_store()
    results = store.memory_store.similarity_search(
        context, k=k,
        filter={"user_id": user_id}
    )
    return [r.page_content for r in results]