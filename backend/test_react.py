# backend/test_react.py
"""Test the reasoning agent. Run: python -m backend.test_react"""
from backend.graph.state import create_initial_state
from backend.agents.reasoning_agent import reasoning_agent_node

if __name__ == "__main__":
    state = create_initial_state("What is 2^10 and what day is it today?")
    result = reasoning_agent_node(state)
    print("Reasoning trace:")
    for step in result["reasoning_trace"]:
        print(f"  {step[:120]}")
    print("✅ ReAct agent working")