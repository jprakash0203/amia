import os
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults


@tool
def search_web(query: str) -> str:
    """
    Search the live internet for current information.

    Use when you need recent events, news, current data, or to verify facts.
    Be specific in your query for better results.

    Args:
        query: A specific search query. Example: "LangGraph 0.2 release notes 2024"
    """
    # The @tool decorator transforms this function into a LangChain Tool.
    # The LLM reads the docstring above to decide WHEN to call this tool.
    # IMPORTANT: Write clear, descriptive docstrings — the LLM literally
    # reads them as instructions for when and how to use the tool.

    search = TavilySearchResults(
        max_results=5,
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
    )
    results = search.invoke(query)

    if not results:
        return "No results found."

    # Format results as readable text for the LLM
    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(
            f"[{i}] {r.get('title', 'N/A')}\n"
            f"URL: {r.get('url', '')}\n"
            f"{r.get('content', '')[:400]}"
        )
    return "\n\n".join(formatted)