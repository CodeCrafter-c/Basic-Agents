from langgraph.graph import StateGraph,START,END
from typing import TypedDict

class BatsmanState(TypedDict):
    runs:int
    balls:int
    fours:int
    sixes:int

    sr:float
    boundary_percentage:float
    bpb:int
    summary:str

def calculate_sr(state:BatsmanState)->BatsmanState :
    runs=state["runs"]
    balls=state["balls"]
    sr=(runs/balls)*100
    return {"sr":sr}

def calculate_boundary_percentage(state:BatsmanState)->BatsmanState :
    runs=state["runs"]
    fours=state["fours"]*4
    sixes=state["sixes"]*6
    
    boundary_percentage= ((fours+sixes)/runs)*100
    return {"boundary_percentage":boundary_percentage}

def calculate_bpb(state:BatsmanState)->BatsmanState :
    balls=state["balls"]
    fours=state["fours"]
    sixes=state["sixes"]
    bpb=round(balls/(fours+sixes))  
    return {"bpb":bpb}

def summary(state:BatsmanState)->BatsmanState:
    summary=f"""
    Strike Rate = {state["sr"]}\n
    Balls Per Boundary = {state["bpb"]}\n
    Boundary Percentage= {state["boundary_percentage"]}
    """
    return {"summary":summary}

graph=StateGraph(BatsmanState)

# nodes
graph.add_node("calculate_sr",calculate_sr)
graph.add_node("calculate_boundary_percentage",calculate_boundary_percentage)
graph.add_node("calculate_bpb",calculate_bpb)
graph.add_node("summary",summary)

#edges
graph.add_edge(START,"calculate_sr")
graph.add_edge(START,"calculate_boundary_percentage")
graph.add_edge(START,"calculate_bpb")

graph.add_edge("calculate_sr","summary")
graph.add_edge("calculate_boundary_percentage","summary")
graph.add_edge("calculate_bpb","summary")

graph.add_edge("summary",END)


#compile
workflow=graph.compile()


#INITAL STATE
initial_state={
    "runs":100,
    "balls":50,
    "fours":6,
    "sixes":4
}

final_state=workflow.invoke(initial_state)
print(final_state)