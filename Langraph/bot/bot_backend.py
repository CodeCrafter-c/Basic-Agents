from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph.message import add_messages 
from langgraph.checkpoint.memory import InMemorySaver

llm=ChatOllama(model="qwen2.5:7b")

class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]
    
def chat_node(state:ChatState)->ChatState:
    messages=state["messages"]
    
    res=llm.invoke(messages)
    return{'messages':[res]}
    
check_pointer=InMemorySaver()
graph=StateGraph(ChatState)

#Node
graph.add_node("chat_node",chat_node)


#edges
graph.add_edge(START,"chat_node")
graph.add_edge("chat_node",END)


chatBot=graph.compile(checkpointer=check_pointer)

