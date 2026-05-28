"""
The Reasoning Agent — performs multi-step ReAct reasoning with tool use.

ReAct = Reasoning + Acting
The agent thinks, optionally calls a tool, observes the result,
thinks again, and repeats until it has a confident answer.
"""

import json
import datetime
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from backend.llm_config import get_llm
from backend.graph.state import AMIAState
from backend.tools.search_tools import search_web
from backend.tools.code_tools import run_python

SYSTEM_PROMPT = """You are AMIA's Reasoning Agent. Perform step-by-step analysis.

Available tools:
- search_web: Search the internet for current information
- run_python: Execute Python code for calculations

Process:
1. Think about what you know and what you need
2. Use a tool if you need external data or calculation
3. Observe the tool result
4. Repeat until confident
5. Provide your final conclusion

Be methodical. Show your reasoning. Cite evidence for each claim."""


class ReasoningAgent:
    """
    Executes the ReAct loop:
    Think → Act (optional tool call) → Observe → Repeat → Final Answer
    """

    def __init__(self):
        self.llm = get_llm(temperature=0.1)
        self.tools = [search_web, run_python]

        # bind_tools() tells the LLM about available tools.
        # The LLM can then request tool calls in its response.
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Lookup dict: tool name → tool function
        self.tool_map = {t.name: t for t in self.tools}

    def run(self, state: AMIAState) -> dict:
        """
        Called by LangGraph when this node is activated.
        Returns dict of state fields to update.
        """
        query = state["query"]

        # Build context from what other agents have already found
        context_parts = []
        for r in state.get("web_results", [])[:3]:
            context_parts.append(f"Web: {r.get('content', '')[:200]}")
        for r in state.get("vector_results", [])[:3]:
            context_parts.append(f"KB: {r.get('document', '')[:200]}")
        context = "\n".join(context_parts) or "No prior context."

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Query: {query}\n\n"
                f"Context gathered so far:\n{context}\n\n"
                f"Reason step-by-step. Use tools if needed."
            )),
        ]

        reasoning_steps = []
        max_iterations = 5  # Safety limit — prevents infinite tool-calling loops

        for i in range(max_iterations):
            # Call the LLM — it may respond with text OR a tool request
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)  # Add to history for multi-turn

            if response.tool_calls:
                # LLM wants to use a tool
                for tc in response.tool_calls:
                    tool_name = tc["name"]
                    tool_args = tc["args"]

                    reasoning_steps.append(
                        f"[Step {i+1}] Tool: {tool_name}({json.dumps(tool_args)[:100]})"
                    )

                    # Execute the tool
                    result = self.tool_map[tool_name].invoke(tool_args)
                    reasoning_steps.append(f"[Step {i+1}] Result: {str(result)[:300]}")

                    # Feed result back to LLM as a ToolMessage
                    messages.append(ToolMessage(
                        content=str(result),
                        tool_call_id=tc["id"],
                    ))
            else:
                # LLM gave a text response — it's done reasoning
                reasoning_steps.append(f"[Final] {response.content[:500]}")
                break

        return {
            "reasoning_trace": reasoning_steps,
            "agent_trace": [{
                "agent": "reasoning_agent",
                "action": f"ReAct: {len(reasoning_steps)} steps",
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "status": "complete",
            }],
        }


def reasoning_agent_node(state: AMIAState) -> dict:
    """LangGraph node function wrapper."""
    return ReasoningAgent().run(state)