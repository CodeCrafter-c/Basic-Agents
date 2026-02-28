from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

embedding_model = os.getenv('EMBEDDING_MODEL')
api = os.getenv('QDRANT_API')
url = os.getenv('QDRANT_URL')

embeddings = OllamaEmbeddings(model=embedding_model)

client = QdrantClient(
    url=url,
    api_key=api
)

try:
    client.create_payload_index(
        collection_name="transcript",
        field_name="metadata.video_id",
        field_schema=models.PayloadSchemaType.KEYWORD
    )
except Exception:
    pass
def delete_video_chunks(video_id):
    client.delete(
        collection_name="transcript",
        points_selector=models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.video_id",
                    match=models.MatchValue(value=video_id)
                )
            ]
        )
    )

    print(f"Deleted chunks for video_id: {video_id}")



vector_store = QdrantVectorStore(
    client=client,
    collection_name="transcript",
    embedding=embeddings,
    vector_name="dense"
)

def add_to_db(split_docs):
    vector_store.add_documents(split_docs)
        
    print("Total points:", client.count("transcript").count)


def get_retriever(video_id):
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":3,
            "filter":models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.video_id",
                        match=models.MatchValue(value=video_id)
                    )
                ]
            )
            }
    )
