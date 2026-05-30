import datetime
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.messages import SystemMessage, HumanMessage

from backend.llm_config import get_llm
from backend.graph.state import AMIAState


class SupervisorDecision(BaseModel):
    """What the supervisor decides at each step."""
    next_agent: Literal[
        "web_search",
        "vector_retrieval",
        "reasoning",
        "summarize",
        "critique",
        "synthesize",
        "FINISH",
    ] = Field(description="Which agent should run next")
    reasoning: str = Field(description="Why this agent was chosen")


SYSTEM_PROMPT = """You are AMIA's Supervisor. Orchestrate specialist agents to research queries.

Available agents:
- web_search:       Fetch live internet information. Use for recent events or current data.
- vector_retrieval: Search internal knowledge base. Use FIRST for background knowledge.
- reasoning:        Multi-step analysis and calculations. Use after gathering raw data.
- summarize:        Condense scattered results. Use when you have lots of raw content.
- critique:         Evaluate quality. Use before synthesizing to check completeness.
- synthesize:       Build final response. Use LAST when information is sufficient.
- FINISH:           Research complete. Use after synthesize.

Decision rules:
1. Always start with vector_retrieval (check existing knowledge first)
2. Use web_search if current data is needed or vector results are insufficient
3. Use reasoning for analytical questions requiring calculation or logic
4. Always critique before synthesizing
5. Never synthesize prematurely — better to over-research than under-research
6. If retry_count >= 2, proceed to synthesize regardless"""


class SupervisorAgent:

    def __init__(self):
        # temperature=0.0 = deterministic routing decisions
        self.llm = get_llm(temperature=0.0)
        self.structured_llm = self.llm.with_structured_output(SupervisorDecision)

    def decide(self, state: AMIAState) -> dict:
        status = self._build_status(state)

        decision = self.structured_llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Query: {state['query']}\n\n"
                f"Current status:\n{status}\n\n"
                f"Retry count: {state.get('retry_count', 0)}/2\n\n"
                f"What should AMIA do next?"
            )),
        ])

        return {
            "next_agent": decision.next_agent,
            "agent_trace": [{
                "agent": "supervisor",
                "action": f"Routing to: {decision.next_agent} — {decision.reasoning[:100]}",
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "status": "routing",
            }],
        }

    def _build_status(self, state: AMIAState) -> str:
        lines = [
            f"Web results:      {len(state.get('web_results', []))}",
            f"Vector results:   {len(state.get('vector_results', []))}",
            f"Reasoning steps:  {len(state.get('reasoning_trace', []))}",
            f"Summaries:        {len(state.get('summaries', []))}",
            f"Critique score:   {state.get('critique_score', 'Not run yet')}",
            f"Final response:   {'Ready' if state.get('final_response') else 'Not yet'}",
        ]
        return "\n".join(lines)


def supervisor_node(state: AMIAState) -> dict:
    return SupervisorAgent().decide(state)
