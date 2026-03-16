from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Dict, Optional, Any
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition

from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
import sqlite3
import requests
import os
import tempfile

# Simple global that chat_node sets before the LLM runs.
# ToolNode executes in the same process/thread so this is reliably visible.
_active_thread_id: Optional[str] = None

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings



stock_price_api = os.getenv("STOCK_PRICE_API")
llm = ChatOllama(model="qwen2.5:7b")
title_model = ChatOllama(model="qwen2.5:7b")

# FIX 1: embeddings must be an OllamaEmbeddings object, not a raw string.
# os.getenv() returns a plain string like "nomic-embed-text", which cannot
# be passed to FAISS.from_documents(). Wrapping it in OllamaEmbeddings() fixes that.
embeddings = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL"))


# PDF store per thread
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.
    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
        print(f"[ingest_pdf] stored retriever for thread_id={thread_id} | known threads={list(_THREAD_RETRIEVERS.keys())}")

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


# Tools
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={stock_price_api}"
    r = requests.get(url)
    return r.json()




@tool
def rag_tool(query: str) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Use this whenever the user asks about the uploaded document.
    """
    thread_id = _active_thread_id
    print(f"[rag_tool] called | thread_id={thread_id} | known threads={list(_THREAD_RETRIEVERS.keys())}")
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


tools = [search_tool, calculator, get_stock_price, rag_tool]
llm_with_tools = llm.bind_tools(tools)


class Title(BaseModel):
    title: str = Field(description="title of the chat", max_length=50)


structured_title_model = title_model.with_structured_output(Title)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    title: str | None


def title_generator(state: ChatState):
    if state.get("title"):
        return {}  # already exists

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


def chat_node(state: ChatState, config: RunnableConfig) -> ChatState:
    global _active_thread_id
    messages = state["messages"]
    thread_id = config.get("configurable", {}).get("thread_id")
    _active_thread_id = thread_id
    print(f"[chat_node] thread_id={thread_id} | retrievers available={list(_THREAD_RETRIEVERS.keys())}")
    res = llm_with_tools.invoke(messages)
    return {"messages": [res]}


conn = sqlite3.connect("chatbot.db", check_same_thread=False)
check_pointer = SqliteSaver(conn)

graph = StateGraph(ChatState)
tool_node = ToolNode(tools)

# Nodes
graph.add_node("title_generator", title_generator)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

# Edges
graph.add_edge(START, "title_generator")
graph.add_edge("title_generator", "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")
graph.add_edge("chat_node", END)

chatBot = graph.compile(checkpointer=check_pointer)


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

    return threads


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})