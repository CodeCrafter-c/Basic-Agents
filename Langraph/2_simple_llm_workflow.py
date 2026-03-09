from langgraph.graph import StateGraph,START,END
from typing import TypedDict
from langchain_ollama import ChatOllama 

model=ChatOllama(model="qwen2.5:7b")

def llm_qa(state : LLMState)-> LLMState:
    ques=state["question"]
    prompt=f'Answer the following question {ques}'
    # ask llm
    res=model.invoke(prompt)
    
    state["answer"]=res.content
    return state

class LLMState(TypedDict):
    question:str
    answer :str
    
graph=StateGraph(LLMState)


graph.add_node("llm_qa",llm_qa)


graph.add_edge(START,"llm_qa")
graph.add_edge("llm_qa",END)

workflow=graph.compile()

initial_state={"question":"distance between sun and the earth is ?"}
final_state=workflow.invoke(initial_state)
print(final_state)