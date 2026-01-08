import json
import requests
import re

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma3:4b"


#helpers
def call_llm(messages):
    payload={
        "model":MODEL,
        "messages":messages,
        "stream":False
    }   
    res=requests.post(OLLAMA_URL,json=payload)
    return res.json()["message"]["content"]

def clean_llm_json(text):
    # Remove ```json or ``` wrappers
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text)
        text = re.sub(r"```$", "", text)

    return text.strip()

system_instructions="""

"""

messages=[
    {
        "role":"user",
        "content":system_instructions
    }
]