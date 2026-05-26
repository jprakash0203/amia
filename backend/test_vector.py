from langchain_core.documents import Document
from backend.vector_store.chroma_store import get_chroma_store

def test_document_rag():
    print("Test 1: Document RAG")
    print("-" * 40)

    store = get_chroma_store()

    # Create sample documents
    docs = [
        Document(page_content="LangGraph is a framework for building multi-agent AI systems.",
                 metadata={"source": "langchain-docs"}),
        Document(page_content="RAG combines retrieval with generation for accurate AI responses.",
                 metadata={"source": "ai-textbook"}),
        Document(page_content="ChromaDB is a vector database for storing embeddings locally.",
                 metadata={"source": "chroma-docs"}),
    ]

    # Add to ChromaDB
    ids = store.add_documents(docs)
    print(f"  Stored {len(ids)} documents")

    # Search by meaning — not exact keywords
    results = store.search_documents("How do multi-agent systems work?", k=2)
    print(f"  Found {len(results)} results for 'multi-agent systems':")
    for r in results:
        print(f"    → {r.page_content[:80]}...")

    print(f"  Stats: {store.stats()}")
    print("✅ Document RAG working\n")


def test_episodic_memory():
    """Test storing and recalling memories."""
    print("Test 2: Episodic Memory")
    print("-" * 40)

    store = get_chroma_store()

    # Store a past reasoning session
    store.store_memory(
        query="What is quantum computing?",
        reasoning="I searched the web and found 3 sources explaining qubits...",
        outcome="success"
    )

    # Recall similar past sessions
    memories = store.recall_memories("Tell me about quantum technologies")
    print(f"  Recalled {len(memories)} memories")
    for m in memories:
        print(f"    → {m.page_content[:80]}...")
    print("✅ Episodic memory working\n")


def test_tool_registry():
    """Test registering and discovering tools."""
    print("Test 3: Tool Registry (MCP)")
    print("-" * 40)

    store = get_chroma_store()

    # Register tools
    store.register_tool(
        "tavily_web_search",
        "Search the internet for current news, events, and recent information.",
        {"type": "object", "properties": {"query": {"type": "string"}}}
    )
    store.register_tool(
        "python_repl",
        "Execute Python code for calculations and data analysis.",
        {"type": "object", "properties": {"code": {"type": "string"}}}
    )

    # Discover tools by task description
    tools = store.find_tools("I need to find recent news about AI", k=2)
    print(f"  Found {len(tools)} tools for 'find recent news':")
    for t in tools:
        print(f"    → {t.metadata.get('tool_name')}: {t.page_content[:60]}...")
    print("✅ Tool registry working\n")


if __name__ == "__main__":
    test_document_rag()
    test_episodic_memory()
    test_tool_registry()
    print("🎉 All vector store tests passed!")