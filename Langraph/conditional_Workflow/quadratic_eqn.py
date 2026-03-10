from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal


class QuadState(TypedDict):
    a: int
    b: int
    c: int

    eqn: str
    dis: float
    res: str


def show_equation(state: QuadState):
    eqn = f"{state['a']}x^2 + {state['b']}x + {state['c']}"
    return {"eqn": eqn}


def calculate_dis(state: QuadState):
    dis = (state["b"] ** 2) - 4 * (state["a"] * state["c"])
    return {"dis": dis}


def real_roots(state: QuadState):
    root1 = (-state["b"] + state["dis"] ** 0.5) / (2 * state["a"])
    root2 = (-state["b"] - state["dis"] ** 0.5) / (2 * state["a"])
    result = f"The roots are {root1} and {root2}"
    return {"res": result}


def repeated_roots(state: QuadState):
    root = -state["b"] / (2 * state["a"])
    result = f"Only repeating root is {root}"
    return {"res": result}


def no_real_roots(state: QuadState):
    return {"res": "No real roots"}


def check_condition(state: QuadState) -> Literal[
    "real_roots", "repeated_roots", "no_real_roots"
]:
    if state["dis"] > 0:
        return "real_roots"
    elif state["dis"] == 0:
        return "repeated_roots"
    else:
        return "no_real_roots"


graph = StateGraph(QuadState)

graph.add_node("show_equation", show_equation)
graph.add_node("calculate_dis", calculate_dis)
graph.add_node("real_roots", real_roots)
graph.add_node("repeated_roots", repeated_roots)
graph.add_node("no_real_roots", no_real_roots)

graph.add_edge(START, "show_equation")
graph.add_edge("show_equation", "calculate_dis")

graph.add_conditional_edges("calculate_dis", check_condition)

graph.add_edge("real_roots", END)
graph.add_edge("repeated_roots", END)
graph.add_edge("no_real_roots", END)

workflow = graph.compile()

initialState = {
    "a": 4,
    "b": -5,
    "c": -4
}

print(workflow.invoke(initialState))