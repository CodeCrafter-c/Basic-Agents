import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from bot_backend import chatBot, retrieve_all_threads, ingest_pdf, thread_document_metadata
import uuid
from langgraph.types import Command

# ---------- utility functions ----------

def generate_thread_id():
    return str(uuid.uuid4())


def new_chat():
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []


def add_chats(thread_id, title):
    if thread_id not in st.session_state["previous_chats"]:
        st.session_state["previous_chats"][thread_id] = title


def show_chats():
    # Show chats newest-first
    chats = list(st.session_state["previous_chats"].items())[::-1]
    for thread_id, title in chats:
        if st.sidebar.button(title, key=thread_id):
            
            st.session_state["thread_id"] = thread_id
            load_conversations(thread_id)
            st.rerun()  


def load_conversations(thread_id):
    """
    Load messages from LangGraph state into session_state["message_history"].
    Returns nothing — it directly sets session state.
    """
    state = chatBot.get_state(config={"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", [])
    history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and msg.content:
            history.append({"role": "assistant", "content": msg.content})
    st.session_state["message_history"] = history


# ---------- session state initialisation ----------
if "pending_interrupt" not in st.session_state:
    st.session_state["pending_interrupt"] = None

if "previous_chats" not in st.session_state:
    st.session_state["previous_chats"] = retrieve_all_threads()

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["previous_chats"]

config = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_run",
    }

# ---------- sidebar ----------

st.sidebar.title("let's chat")

if st.sidebar.button("new chat"):
    new_chat()

if st.session_state["previous_chats"]:
    show_chats()

# PDF status & uploader
if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)

# ---------- main chat area ----------

st.title("Chatbot")

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("type here")

state = chatBot.get_state(config)
if state.next:
    st.session_state["pending_interrupt"] = state.tasks[0].interrupts[0].value
else:
    st.session_state["pending_interrupt"] = None
if st.session_state.get("pending_interrupt"):
    st.warning(st.session_state["pending_interrupt"])
    col1, col2 = st.columns(2)
    if col1.button("✅ Confirm", key="hitl_confirm"):
        chatBot.invoke(Command(resume="yes"), config=config)
        st.session_state["pending_interrupt"] = None
        load_conversations(st.session_state["thread_id"])
        st.rerun()
    if col2.button("❌ Cancel", key="hitl_cancel"):
        chatBot.invoke(Command(resume="no"), config=config)
        st.session_state["pending_interrupt"] = None
        load_conversations(st.session_state["thread_id"])
        st.rerun()
if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    

    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatBot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            ):
                if metadata.get("langgraph_node") == "title_generator":
                    continue

                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(f"🔧 Using `{tool_name}` …", expanded=True)
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …", state="running", expanded=True
                        )

                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(label="✅ Tool finished", state="complete", expanded=False)

        # fetch fresh state FIRST, then check
        state = chatBot.get_state(config)
        if state.next:
            st.rerun()

        title = state.values.get("title")   # reuse same fresh state, no need to fetch again
        add_chats(st.session_state["thread_id"], title)

        ai_text = ai_message if isinstance(ai_message, str) else "".join(ai_message or [])
        st.session_state["message_history"].append({"role": "assistant", "content": ai_text})


    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )