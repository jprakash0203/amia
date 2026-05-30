import json
import datetime
from langchain_core.messages import SystemMessage, HumanMessage

from backend.llm_config import get_llm
from backend.graph.state import AMIAState
from backend.tools.search_tools import search_web

SYSTEM_PROMPT = """You are AMIA's Web Search Agent. Find current, accurate information.
When searching:
1. Generate 2-3 specific targeted queries (not one vague query)
2. Extract the most relevant information from each result
3. Note publication dates when visible
4. Flag conflicting information across sources

Return ONLY a JSON array of search queries. Example:
["LangGraph 0.2 release notes 2024", "LangGraph vs CrewAI comparison", "LangGraph state management"]"""


class WebSearchAgent:

    def __init__(self):
        self.llm = get_llm(temperature=0.1)

    def run(self, state: AMIAState) -> dict:
        query = state["query"]

        # Step 1: Generate targeted sub-queries
        response = self.llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Research topic: {query}"),
        ])

        try:
            clean = response.content.strip().replace("```json", "").replace("```", "")
            search_queries = json.loads(clean)
            if not isinstance(search_queries, list):
                search_queries = [query]
        except Exception:
            search_queries = [query]

        # Step 2: Execute each query
        all_results = []
        seen_urls = set()

        for sq in search_queries[:3]:   # Max 3 to control cost
            try:
                result_text = search_web.invoke({"query": sq})
                # Parse the formatted result text into structured dicts
                for block in result_text.split("\n\n"):
                    lines = block.strip().split("\n")
                    if len(lines) >= 2:
                        url_line = next((l for l in lines if l.startswith("URL:")), "")
                        url = url_line.replace("URL:", "").strip()
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            all_results.append({
                                "title": lines[0].lstrip("[0123456789] "),
                                "url": url,
                                "content": "\n".join(lines[2:])[:500],
                                "search_query": sq,
                            })
            except Exception as e:
                print(f"  Search failed for '{sq}': {e}")

        return {
            "web_results": all_results,
            "agent_trace": [{
                "agent": "web_search_agent",
                "action": f"Searched {len(search_queries)} queries, found {len(all_results)} results",
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "status": "complete",
            }],
        }


def web_search_agent_node(state: AMIAState) -> dict:
    return WebSearchAgent().run(state)
