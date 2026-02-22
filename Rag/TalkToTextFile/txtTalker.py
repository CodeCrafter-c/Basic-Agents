from langchain_text_splitters import RecursiveCharacterTextSplitter
# from  pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
context=""

llm = ChatOllama(
    model="gemma3:4b",
    temperature=0
)

chat_history=[]

system_prompt="""
You are a helpful ai assistant , who is very good in ansering the user query based on the given 
context only.
Beacuse you are being used in a rag application where you will recieve information from the text file user has uploaded.
If the answer is not in the context, respond with:
"I don't know based on the provided document."
so the context is :- 
"""





# file=Path(__file__).resolve().parent /"TalkToTextFile"/"testing.txt"
file='testing.txt'
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)
chunks = []
buffer = ""
# chunking
with open('testing.txt', "r") as f:
    for line in f:
        buffer += line
        if len(buffer) >= 1000:
            chunks.extend(text_splitter.split_text(buffer))
            buffer = ""

if buffer:
    chunks.extend(text_splitter.split_text(buffer))

# embeddings
embeddings=OllamaEmbeddings(
    model="nomic-embed-text"
)

vectorstore = QdrantVectorStore.from_texts(
    texts=chunks,
    embedding=embeddings,
    url="https://4ec64686-fe66-47eb-bf2c-1ec7ae32c0e8.europe-west3-0.gcp.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.bpwUipxhOeUTKiVM89AXvWe4nfyGGy7Uq3Xr44I_it8",
    collection_name="text-file-data",
    force_recreate=True

)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
while True:
    query = input("\nAsk: ")

    if query.lower() == "exit":
        print("thank you")
        break

    # Retrieve based on USER query
    docs = retriever.invoke(query)
    retrieved_context = "\n\n".join([doc.page_content for doc in docs])

    messages = [
        SystemMessage(
            content=system_prompt + "\n\nContext:\n" + retrieved_context
        )
    ]

    # Add previous conversation
    messages.extend(chat_history)

    # Add new question
    messages.append(HumanMessage(content=query))

    # Invoke model
    response = llm.invoke(messages)

    print("\nAnswer:", response.content)

    # Save conversation properly
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response.content))