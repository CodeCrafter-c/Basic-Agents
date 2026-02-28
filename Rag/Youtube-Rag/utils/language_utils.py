from langchain_core.messages import HumanMessage , SystemMessage
from langdetect import detect, DetectorFactory

DetectorFactory.seed=0

def get_langCode(text:str)-> str:
    """
    Detect language of given text.
    Returns ISO 639-1 code like:
    'en' for English
    'hi' for Hindi
    """
    try:
        return detect(text)
    except:
        return "unknown"

# def detect_language(llm, query):
#     messages = [
#         SystemMessage(
#             content="""
# You are a language detection system.

# Identify the language of the user query.
# Respond with only the language name.
# Do not explain.

# Examples:
# Query: what is langchain?
# Answer: English

# Query: इस वीडियो का मुख्य विषय क्या है?
# Answer: Hindi
# """
#         ),
#         HumanMessage(content=query)
#     ]

#     response = llm.invoke(messages)
#     return response.content.strip().lower()



def translate_query(llm, query, target_lang):
    messages = [
        SystemMessage(
            content=f"""
You are a query translation system.

Translate the user query into {target_lang}.
Return only the translated text.
Do not explain.

Examples:

Example 1:
Query: इस वीडियो का मुख्य विषय क्या है?
Target Language: English
Answer: What is the main topic of this video?

Example 2:
Query: What is JWT?
Target Language: Hindi
Answer: JWT क्या है?
"""
        ),
        HumanMessage(content=query)
    ]

    response = llm.invoke(messages)
    return response.content.strip()