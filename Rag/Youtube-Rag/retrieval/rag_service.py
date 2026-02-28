from utils.language_utils import get_langCode , translate_query
from .prompts import Rag_Prompt 
# from .retrieve import retrieve_docs
from ingestion.vector_store import get_retriever


def answer_query(video_id,llm,user_query,transcript_langcode):
    # detects user lang
    user_lang=get_langCode(user_query);

    #check if needs to translate query
    if(user_lang!=transcript_langcode):
        retrieval_query=translate_query(llm,user_query,transcript_langcode)
    else:
        retrieval_query=user_query
        
    #retrieve docs
    retriever=get_retriever(video_id)
    docs=retriever.invoke(retrieval_query)
    
    #context
    context="\n\n".join([doc.page_content for doc in docs])\
        
    #final answer
    chain=Rag_Prompt|llm
    
    response=chain.invoke({
        "context":context,
        "question":user_query
    })
    return response.content
    