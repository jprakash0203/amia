import datetime
from langchain_core.messages import SystemMessage, HumanMessage

from backend.llm_config import get_llm
from backend.graph.state import AMIAState
from backend.memory.episodic import store_episode

SYSTEM_PROMPT = """You are AMIA's Response Synthesizer.

Combine all gathered information into a comprehensive, well-cited response.

Requirements:
1. ACCURACY: Only state what the sources support. Never invent facts.
2. CITATIONS: Every specific claim must reference its source as [Source: URL or description].
3. STRUCTURE: Use clear headers and logical flow.
4. COMPLETENESS: Address all aspects of the original query.
5. CLARITY: Write for an intelligent non-expert audience.

Response format:
## Summary
[2-3 sentence overview]

## [Topic Sections]
[Detailed findings with inline citations]

## Key Takeaways
[3-5 bullet points]

## Sources
[Numbered list of all cited sources]"""


class SynthesizerAgent:

    def __init__(self):
        self.llm = get_llm(temperature=0.2)

    def run(self, state: AMIAState) -> dict:
        context = self._build_context(state)

        response = self.llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Research Query: {state['query']}\n\n"
                f"All gathered information:\n{context}\n\n"
                f"Critique feedback to address:\n"
                f"{state.get('critique_feedback', 'None')}\n\n"
                f"Synthesize into a comprehensive cited response."
            )),
        ])

        citations = self._extract_citations(state)

        # Store this session as episodic memory for future recall
        store_episode(
            query=state["query"],
            agent_trace=state.get("agent_trace", []),
            final_response=response.content,
            critique_score=state.get("critique_score", 1.0),
        )

        return {
            "final_response": response.content,
            "citations": citations,
            "next_agent": "FINISH",
            "agent_trace": [{
                "agent": "synthesizer_agent",
                "action": f"Synthesized response ({len(response.content)} chars)",
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "status": "complete",
            }],
        }

    def _build_context(self, state: AMIAState) -> str:
        sections = []

        web = state.get("web_results", [])
        if web:
            sections.append("=== WEB SEARCH RESULTS ===")
            for i, r in enumerate(web, 1):
                sections.append(
                    f"[{i}] {r.get('title', 'N/A')}\n"
                    f"    URL: {r.get('url', '')}\n"
                    f"    {r.get('content', '')[:400]}"
                )

        vec = state.get("vector_results", [])
        if vec:
            sections.append("\n=== KNOWLEDGE BASE RESULTS ===")
            for i, r in enumerate(vec, 1):
                sections.append(
                    f"[KB{i}] Source: {r.get('source', 'unknown')}\n"
                    f"       {r.get('document', '')[:400]}"
                )

        reasoning = state.get("reasoning_trace", [])
        if reasoning:
            sections.append("\n=== REASONING ANALYSIS ===")
            sections.extend(reasoning[-3:])

        summaries = state.get("summaries", [])
        if summaries:
            sections.append("\n=== SUMMARIES ===")
            sections.extend(summaries)

        return "\n\n".join(sections) if sections else "No information gathered."

    def _extract_citations(self, state: AMIAState) -> list:
        citations = []
        for r in state.get("web_results", []):
            if r.get("url"):
                citations.append({
                    "type": "web",
                    "title": r.get("title", "Web Source"),
                    "url": r.get("url", ""),
                })
        for r in state.get("vector_results", []):
            if r.get("collection") == "documents":
                citations.append({
                    "type": "document",
                    "title": r.get("file_name", "Internal Document"),
                    "url": r.get("url", ""),
                })
        return citations


def synthesizer_agent_node(state: AMIAState) -> dict:
    return SynthesizerAgent().run(state)
