import datetime
from typing import List

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from backend.llm_config import get_llm
from backend.graph.state import AMIAState


class CritiqueResult(BaseModel):
    """Structured output from the critique agent."""
    score: float = Field(description="Quality score 0.0–1.0")
    feedback: str = Field(description="What is missing or incorrect")
    is_sufficient: bool = Field(description="True if ready to synthesize")
    missing_aspects: List[str] = Field(description="Gaps in coverage; use an empty list if none")


SYSTEM_PROMPT = """You are AMIA's Critique Agent — a quality assurance expert.

Evaluate whether the gathered information is sufficient to answer the query accurately.

Score criteria:
  0.9–1.0: Excellent — comprehensive, accurate, well-sourced
  0.7–0.89: Good — mostly complete, minor gaps, ready to synthesize
  0.5–0.69: Adequate — significant gaps, more research recommended
  Below 0.5: Poor — insufficient or unreliable

Evaluate:
1. Relevance: Does the info directly answer the query?
2. Completeness: Are all aspects covered?
3. Recency: Is the info current enough?
4. Source diversity: Multiple credible sources?
5. Consistency: Do sources agree?"""


class CritiqueAgent:

    def __init__(self):
        self.llm = get_llm(temperature=0.0)
        # with_structured_output forces the LLM to return a CritiqueResult object
        # More reliable than asking for JSON and parsing manually
        self.structured_llm = self.llm.with_structured_output(CritiqueResult)

    def run(self, state: AMIAState) -> dict:
        # Build status summary
        web_count = len(state.get("web_results", []))
        vec_count = len(state.get("vector_results", []))
        trace_count = len(state.get("reasoning_trace", []))

        summary_parts = [
            f"Web results: {web_count}",
            f"Vector DB results: {vec_count}",
            f"Reasoning steps: {trace_count}",
        ]

        sample_content = []
        for r in state.get("web_results", [])[:2]:
            sample_content.append(f"Web: {r.get('content', '')[:200]}")
        for r in state.get("vector_results", [])[:2]:
            sample_content.append(f"KB: {r.get('document', '')[:200]}")

        result = self.structured_llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Query: {state['query']}\n\n"
                f"Information gathered:\n" + "\n".join(summary_parts) + "\n\n"
                f"Sample content:\n" + "\n".join(sample_content) + "\n\n"
                f"Evaluate the quality and sufficiency."
            )),
        ])

        # Increment retry counter if score is too low
        new_retry = state.get("retry_count", 0)
        if result.score < 0.7:
            new_retry += 1

        return {
            "critique_score": result.score,
            "critique_feedback": result.feedback,
            "retry_count": new_retry,
            "agent_trace": [{
                "agent": "critique_agent",
                "action": f"Score: {result.score:.2f} — {'PASS' if result.score >= 0.7 else 'RETRY'}",
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "status": "complete",
            }],
        }


def critique_agent_node(state: AMIAState) -> dict:
    return CritiqueAgent().run(state)


def route_after_critique(state: AMIAState) -> str:
    """
    Conditional edge after critique.
    If score < 0.7 AND retries remain → back to supervisor for more research.
    Otherwise → proceed to synthesizer.
    """
    score = state.get("critique_score", 1.0)
    retries = state.get("retry_count", 0)

    if score < 0.7 and retries < 2:
        return "supervisor"
    return "synthesizer_agent"