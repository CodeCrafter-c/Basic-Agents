# from ingestion.transcript_loader import load_transcript
# from ingestion.chunking import split_documents
# from ingestion.vector_store import add_to_db, get_retriever
from retrieval.rag_service import answer_query 
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os

load_dotenv()
# video
video_id = "etnLX7m2MiA"

#load transcript
# docs, lang = load_transcript(video_id)

#chunking
# split_docs = split_documents(docs)

#store
# add_to_db(split_docs)

# print("Stored successfully")

#model
# llm=ChatOllama(
#     model=os.getenv("LLM")
# )

# #query
# user_query = "what are built in tools?"

# #get answer
# lang="hi"
# response = answer_query(
#     video_id,
#     llm,
#     user_query,
#     lang,
# )

# print(response)