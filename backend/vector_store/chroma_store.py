import os
import json
import datetime
from typing import List, Dict, Any, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from backend.llm_config import get_embedding_model

COLL_DOCUMENTS = "amia_documents"
COLL_MEMORY = "amia_call_memory"
COLL_TOOLS = "amia_tools"

class ChromaVectorStore:

    def __init__(self, use_local_embeddings: bool = True):
        self.persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chromaDB")
        self.embedding_model = get_embedding_model(use_local=use_local_embeddings)
        os.makedirs(self.persist_dir, exist_ok=True)
        self.docs = self._create_collection(COLL_DOCUMENTS)
        self.memory = self._create_collection(COLL_MEMORY)
        self.tools = self._create_collection(COLL_TOOLS)
        print(f"✅ ChromaDB initialised at {self.persist_dir}")
        
    
    def _create_collection(self, name: str) -> Chroma:
        return Chroma(
            collection_name = name,
            embedding_function = self.embedding_model,
            persist_directory=self.persist_dir,
        )

        # ══════════════════════════════════════════════════════════════
    # USE CASE 1: Document RAG
    # ══════════════════════════════════════════════════════════════

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Stores document chunks in the documents collection.

        Args:
            documents: List of LangChain Document objects.
                       Each has .page_content (text) and .metadata (dict).

        Returns:
            List of IDs assigned by ChromaDB (one per document).

        Behind the scenes:
        1. Each document's text is converted to an embedding vector
        2. The vector + text + metadata are stored in ChromaDB
        3. An ID is assigned for future reference
        """
        if not documents:
            return []
        ids = self.docs.add_documents(documents)
        print(f"  Added {len(documents)} chunks to documents collection")
        return ids

    def search_documents(self, query: str, k: int = 5) -> List[Document]:
        """
        Searches documents by semantic similarity.

        Behind the scenes:
        1. Your query string is converted to an embedding vector
        2. ChromaDB finds the k vectors closest to your query vector
        3. Returns the corresponding Document objects

        Args:
            query: Natural language search query.
            k: Number of results to return.
        """
        return self.docs.similarity_search(query, k=k)

    def search_with_scores(self, query: str, k: int = 5) -> List[tuple]:
        """
        Like search_documents but also returns similarity scores.
        Returns: List of (Document, score) tuples.
        Score interpretation: lower = more similar (L2 distance).
        """
        return self.docs.similarity_search_with_score(query, k=k)

    # ══════════════════════════════════════════════════════════════
    # USE CASE 2: Episodic Memory
    # ══════════════════════════════════════════════════════════════

    def store_memory(self, query: str, reasoning: str, outcome: str) -> str:
        """
        Stores a past reasoning session as episodic memory.

        Next time a similar question is asked, the reasoning agent can
        recall: "I answered something like this before. Here's what worked."
        """
        doc = Document(
            page_content=f"Query: {query}\nReasoning: {reasoning[:800]}",
            metadata={
                "original_query": query,
                "outcome": outcome,
                "timestamp": datetime.datetime.utcnow().isoformat(),
            },
        )
        ids = self.memory.add_documents([doc])
        return ids[0]

    def recall_memories(self, query: str, k: int = 3) -> List[Document]:
        """Finds past sessions semantically similar to the current query."""
        return self.memory.similarity_search(query, k=k)

    # ══════════════════════════════════════════════════════════════
    # USE CASE 3: Tool Registry (MCP)
    # ══════════════════════════════════════════════════════════════

    def register_tool(self, name: str, description: str, schema: dict) -> str:
        """
        Registers a tool description for semantic discovery.

        Instead of hardcoding "use Tavily for web search", agents
        query the registry: "I need to search the web" and get back
        matching tools by meaning — even if the exact words differ.
        """
        doc = Document(
            page_content=f"{name}: {description}",
            metadata={
                "tool_name": name,
                "schema": json.dumps(schema),
            },
        )
        ids = self.tools.add_documents([doc])
        return ids[0]

    def find_tools(self, task_description: str, k: int = 3) -> List[Document]:
        """Semantic tool discovery — finds tools matching a task description."""
        return self.tools.similarity_search(task_description, k=k)

    def stats(self) -> dict:
        """Returns document counts for each collection."""
        return {
            "documents": self.docs._collection.count(),
            "memories": self.memory._collection.count(),
            "tools": self.tools._collection.count(),
        }


# ── Singleton pattern ──────────────────────────────────────────────
# Create the store ONCE and share it across all agents.
_store: Optional[ChromaVectorStore] = None


def get_chroma_store() -> ChromaVectorStore:
    """Returns the shared ChromaDB instance."""
    global _store
    if _store is None:
        _store = ChromaVectorStore(use_local_embeddings=True)
    return _store
   