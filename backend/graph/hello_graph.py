from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class HelloSate(TypedDict):
    message:str
    step: int


def greet_node(state: HelloSate) -> dict:
    print(f" greet_node: recieved Message = '{state['message']}'")    
    return {
        "message": "hello from AIMA",
        "step": state["step"] + 1
    }


def enhance_node(state:HelloSate) -> dict:
    print(f" enhance_node: recieved Message = '{state['message']}'")    
    return {
        "message": state["message"] + " — enhanced",
        "step": state["step"] + 1
    }

def decide_node(state:HelloSate) -> dict:
    print(f" decide_node: recieved Message = '{state['message']}'")    
    return {}


def should_continue(state:HelloSate) -> str:
    if state['step'] < 4 :
      return "continue"
    else:
      return "stop"

def build_hello_graph():
    graph = StateGraph(HelloSate)
    graph.add_node("greet", greet_node)
    graph.add_node("enhance", enhance_node)
    graph.add_node("decide", decide_node)
    graph.set_entry_point("greet")
    graph.add_edge("greet", "enhance")
    graph.add_edge("enhance", "decide")
    graph.add_conditional_edges(
        "decide",
        should_continue,
        {
            "continue": "enhance",
            "stop": END,
        },
    )

    compiled = graph.compile()

    return compiled


if __name__ == "__main__":
    app = build_hello_graph()
    final_state = app.invoke({"message": "", "step": 0})
    print(final_state)
    print(f"Final message: {final_state['message']}")
    print(f"Total steps:   {final_state['steps']}")