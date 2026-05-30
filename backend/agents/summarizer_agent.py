
import datetime
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from backend.llm_config import get_llm
from backend.graph.state import AMIAState

SYSTEM_PROMPT = """You are AMIA's Summarizer Agent.

Your job is to condense scattered research into clear, concise summaries.

Rules:
1. Remove duplicate information across sources
2. Preserve specific facts, numbers, and citations
3. Group related information together
4. Note which source each key fact came from
5. Flag contradictions between sources

Output a structured summary with clear sections."""


class SummarizerAgent:

    def __init__(self):
        self.llm = get_llm(temperature=0.1)

    def run(self, state: AMIAState) -> dict:
        query = state["query"]

        # Gather all raw content
        content_parts = []

        for r in state.get("web_results", [])[:5]:
            content_parts.append(
                f"[WEB - {r.get('url', 'unknown')}]\n{r.get('content', '')[:400]}"
            )

        for r in state.get("vector_results", [])[:5]:
            content_parts.append(
                f"[KB - {r.get('source', 'unknown')}]\n{r.get('document', '')[:400]}"
            )

        for step in state.get("reasoning_trace", [])[-3:]:
            content_parts.append(f"[REASONING]\n{step[:300]}")

        if not content_parts:
            return {
                "summaries": ["No content to summarize yet."],
                "agent_trace": [{
                    "agent": "summarizer_agent",
                    "action": "No content available",
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                    "status": "skipped",
                }],
            }

        all_content = "\n\n---\n\n".join(content_parts)

        response = self.llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Research query: {query}\n\n"
                f"Raw content to summarize:\n{all_content}\n\n"
                f"Produce a concise, well-structured summary."
            )),
        ])

        return {
            "summaries": [response.content],
            "agent_trace": [{
                "agent": "summarizer_agent",
                "action": f"Summarized {len(content_parts)} content blocks",
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "status": "complete",
            }],
        }


def summarizer_agent_node(state: AMIAState) -> dict:
    return SummarizerAgent().run(state)