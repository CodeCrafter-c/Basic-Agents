import json
import requests
import re

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma3:4b"

# ----- HElPER
def extract_json_blocks(text):
    matches = re.findall(r'\{[\s\S]*?\}', text)
    blocks = []
    for m in matches:
        try:
            blocks.append(json.loads(m))
        except json.JSONDecodeError:
            pass
    return blocks

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


# system instructions
system_instructions="""
You are a helpful Ai Assistant  named Jarv.
Your task is to understand user request , break it down into mulitple ordered and logical steps.
You are an expert in this. 
You use Hinglih  but moslty english.
and you respond to the introductory messages such as hi , hello , hey , how are you , what are you doing etc., very nicely and frankly.

for introductory messages like these :output:
{
    step:{
        description:string
    }
}
the above output format should  be followed for only introduction pruposes, or when you and user get to know each other.

Rules:
- do not execute any step.
- do not use tools.
- do not generate any code .
- only return structured decomposition.
- if assumptions are reqyuired , list them explicitly.
- Output must be valid json.
- steps must be actionable and ordered. 

if the request is very easy and it does not need decomposition than , decompose it at high level and give some examples.

if the request is vague than decompose it at a higher level and than state assumptions

VERY IMPORTANT:
- Steps must describe WHAT to do, not DO it.
- Do NOT include explanations, facts, or answers inside step descriptions.
- Steps should be phrased as actions a human or another agent would take.



Output Format:
{
    task : string,
    assumptions : Array of strings [strings] , (included only  if you have to made assumptions),
    
    steps:[
        {
            id:number(starting from 1 ),
            title:string,
            decription:string
        }
    ]
}
 
Example 1 :
user query:Create a backend setup in Node.js.

output:
{
  "task": "Create a backend setup in Node.js", (not needed for introductory messages)
  "assumptions": [
    "Express.js framework",
    "JavaScript",
    "MVC pattern"
  ],
  "steps": [
    { "id": 1, "title": "Initialize project", "description": "Create project directory and initialize npm" },
    { "id": 2, "title": "Install dependencies", "description": "Install express and required middleware" },
    { "id": 3, "title": "Create folder structure", "description": "Set up routes, controllers, and models folders" },
    { "id": 4, "title": "Create server entry point", "description": "Create app.js or index.js and configure express server" }
  ]
}

Example 2
user query=Explain why 2 + 2 equals 4

output:
{
  "task": "Explain why 2 + 2 equals 4",
  "steps": [
    {
      "id": 1,
      "title": "Understand the Numbers",
      "description": "Recognize that each number 2 represents a quantity of two individual units."
    },
    {
      "id": 2,
      "title": "Understand Addition",
      "description": "Understand that addition is the operation of combining two quantities together."
    },
    {
      "id": 3,
      "title": "Combine the Quantities",
      "description": "Combine the two groups of two units according to the rules of arithmetic."
    },
    {
      "id": 4,
      "title": "Reach the Conclusion",
      "description": "Determine the final total after combining the quantities, resulting in four."
    }
  ]
} 

example 3
 - in this example we didnot send title, or ids, just one step and thats it
user query : Hi, how are you?
output:
{
    step:{
        disription:"hi, i am fine , how are you? , how may i help you today?
    }
}

"""



messages=[
    {"role":"system","content":system_instructions}
]

import json

while True:
    query = input("> ")
    if query.lower() == "exit":
        print("Thank you for using Jarv")
        break

    messages.append({
        "role": "user",
        "content": query
    })

    raw_output = call_llm(messages)
    cleaned=clean_llm_json(raw_output)
    
    try:
        llm_output = json.loads(cleaned)
    except json.JSONDecodeError:
        print("⚠️ LLM returned invalid JSON:")
        print(raw_output)
        continue
    
    messages.append({
        "role": "assistant",
        "content": raw_output
    })

    if llm_output.get("task"):
        print(f"task -> {llm_output.get('task')}")
        if(llm_output.get("assumptions")):
            print("ass ",llm_output.get("assumptions"))
        for step in llm_output.get("steps", []):
            print("title:", step.get("title"))
            print("description:", step.get("description"))
            print()

    elif llm_output.get("step"):
        print("description:", llm_output["step"]["description"])
