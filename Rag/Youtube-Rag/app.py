import streamlit as st
from langchain_ollama import ChatOllama

from ingestion.transcript_loader import load_transcript
from ingestion.chunking import split_documents
from ingestion.vector_store import add_to_db
from retrieval.rag_service import answer_query
import re

def extract_video_id(url: str):
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})"
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None

st.set_page_config(
    page_title="Multilingual Video RAG",
    layout="wide"
)

st.title("🎥 Multilingual Video RAG (Qdrant + Ollama)")
st.markdown("Ask questions about a YouTube video transcript.")


if "indexed" not in st.session_state:
    st.session_state.indexed = False

if "video_id" not in st.session_state:
    st.session_state.video_id = None

if "transcript_langcode" not in st.session_state:
    st.session_state.transcript_langcode = None


@st.cache_resource
def get_llm():
    return ChatOllama(model="qwen2.5:7b")

llm = get_llm()


st.subheader("Step 1: Load Video Transcript")

video_url = st.text_input("Paste YouTube Video Link")

if st.button("Load Transcript"):
    video_input=extract_video_id(video_url)

    if not video_input.strip():
        st.warning("Please enter a valid video ID.")
    else:
        st.success(f"Video ID detected: {video_input}")
        with st.spinner("Loading and indexing transcript..."):

            docs, lang = load_transcript(video_input)
            split_docs = split_documents(docs)
            add_to_db(split_docs)

            st.session_state.indexed = True
            st.session_state.video_id = video_input
            st.session_state.transcript_langcode = lang

        st.success("Transcript indexed successfully!")


if st.session_state.indexed:

    st.subheader("Step 2: Ask a Question")

    user_query = st.text_input("Enter your question")

    if st.button("Ask"):

        if not user_query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating answer..."):

                response = answer_query(
                    video_id=st.session_state.video_id,
                    llm=llm,
                    user_query=user_query,
                    transcript_langcode=st.session_state.transcript_langcode
                )

            st.markdown("### Answer:")
            st.write(response)

else:
    st.info("Load a transcript first to start asking questions.")