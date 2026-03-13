from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph.message import add_messages 
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field
import sqlite3



llm=ChatOllama(model="qwen2.5:7b")
title_model=ChatOllama(model= "qwen2.5:7b")

class Title(BaseModel):
    title:str =Field(description="title of the chat" ,max_length=50)
    
structured_title_model=title_model.with_structured_output(Title)


class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]
    title:str|None


def title_generator(state: ChatState):

    if state.get("title"):  
        return {}   # already exists

    first_message = state["messages"][0].content

    prompt = f"""
    Generate a short title (max 6 words) for this conversation.

    Example 1
    User message:How do I learn LangGraph step by step?
    Generated title:Learning LangGraph Step by Step

    Example 2
    User message:Why is my Python recursion function causing a stack overflow error?
    Generated title:Fixing Python Recursion Stack Overflow


    Message:
    {first_message}
    """

    res = structured_title_model.invoke(prompt)

    return {"title": res.title}


    
def chat_node(state:ChatState)->ChatState:
    messages=state["messages"]
    
    res=llm.invoke(messages)
    return{'messages':[res]}

conn=sqlite3.connect("chatbot.db",check_same_thread=False)    
check_pointer=SqliteSaver(conn)

graph=StateGraph(ChatState)

#Node
graph.add_node("title_generator", title_generator)
graph.add_node("chat_node",chat_node)


#edges
graph.add_edge(START,"title_generator")
graph.add_edge("title_generator","chat_node")
graph.add_edge("chat_node",END)


chatBot=graph.compile(checkpointer=check_pointer)


def retrieve_all_threads():
    threads = {}
    thread_ids = set()

    print("Scanning checkpoints...")

    for checkpoint in check_pointer.list(None):
        thread_ids.add(checkpoint.config["configurable"]["thread_id"])

    print("Unique threads:", thread_ids)

    for thread_id in thread_ids:
        state = chatBot.get_state(
            {"configurable": {"thread_id": thread_id}}
        )

        title = state.values.get("title") or "New Chat"
        threads[thread_id] = title

    # print("Threads:", threads)
    return threads
