import requests
import json
import re

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma3:4b"


# ---------------- TOOLS ---------------- 

def add(*args):
    return sum(args)

def subtraction(a, b):
    return a - b

def product(*args):
    result = 1
    for num in args:
        result *= num
    return result

def divide(numerator, denominator):
    if denominator == 0:
        return "Error: Division by zero"
    return numerator / denominator


AVAILABLE_TOOLS = {
    "add": {"fn": add, "arity": "n"},
    "subtraction": {"fn": subtraction, "arity": 2},
    "product": {"fn": product, "arity": "n"},
    "divide": {"fn": divide, "arity": 2}
}


# ---------------- HELPERS ---------------- 

def extract_json_blocks(text):
    matches = re.findall(r'\{[\s\S]*?\}', text)
    blocks = []
    for m in matches:
        try:
            blocks.append(json.loads(m))
        except json.JSONDecodeError:
            pass
    return blocks


def parse_tool_input(input_str):
    # print(input_str)
    return [float(x.strip()) for x in input_str.split(",")]


def call_llm(messages):
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False
    }
    res = requests.post(OLLAMA_URL, json=payload)
    return res.json()["message"]["content"]


def execute_tool(step):
    tool_name = step.get("function")
    raw_input = step.get("input")

    if tool_name not in AVAILABLE_TOOLS:
        return "Error: Tool not available"

    tool = AVAILABLE_TOOLS[tool_name]
    parsed_input = parse_tool_input(raw_input)

    if tool["arity"] != "n" and len(parsed_input) != tool["arity"]:
        return f"Error: {tool_name} requires {tool['arity']} arguments"

    return tool["fn"](*parsed_input)


system_instructions=""""
you are an smart ai calculator assistant.
You help user to solve basic mathematical problems.
You are proficient at basic arithmetic operations.
For the given query ,  you understand and than plan and based on planning choose from the available tools , which tool need to be used to solve the user query and then perfrom the execution step to solve the query using the aviable tool, and than observe the result that the tool returned and than return it to the user.

you shall perform nomral introductory conversation if user insists and for normal conversation return direct step = output and if user ask something out of the scope , say something like  , sorry i am here to help you with basic arithmetic operations only .

so if the only step that you will be following at a given time are:
plan
execution
observe
output

- Use tools ONLY if computation is required.
- If the query can be answered using previous observations or reasoning, 
  directly return step="output".
- NEVER fabricate tool results.

Output Format:{
    "step":"string",
    "content":"string",
    "function":"the name of the function if the step is execution",
    "input":"the parameter of the function"
}

Available_tools:
- add= takes n number as input and return their sum.
- subtraction= takes 2 number as input and return their differnce.
- product= takes n number as input and return their product.
- divide= take numerator and denominator as input and return the Quotient in floating point.

Examples:

-User Query: what is the sum of 2+9+8+8+4+5+5
output:{"step":"plan","content":"The user wants to know the sum of 2+9+8+8+4+5+5"},
output:{"step":"plan":"content":"From the available tools i shall call add"},
output:{"step":"execution","function":"add","input":"2,9,8,8,4,5,5"}
output:{"step":"observe","content":"41"},
output:{"step":"output","content":"the result of addition of the given number is 41"}

-User Query: find the 10^2;
output:{"step":"plan","content":"The user wants to know 10^2 which means user want to calculate 10 power 2 which mean  10 * 10"},
output:{"step":"plan":"content":"From the available tools i shall call prodcut"},
output:{"step":"execution","function":"product","input":"10,10"}
output:{"step":"observe","content":"100"},
output:{"step":"output","content":"the result of 10 power 2 is  100"}

"""


messages = [
    {"role": "system", "content": system_instructions}
]

while True:
    user_query = input("> ")

    if user_query.lower() == "exit":
        print("👋 Goodbye")
        break

    messages.append({"role": "user", "content": user_query})

    llm_output = call_llm(messages)
    messages.append({"role": "assistant", "content": llm_output})

    steps = extract_json_blocks(llm_output)

    if not steps:
        print("⚠️ No valid steps found")
        continue

    for step in steps:
        step_type = step.get("step")

        if step_type == "plan":
            print(f"🤔 Plan: {step.get('content')}")

        elif step_type == "execution":
            result = execute_tool(step)

            observe_msg = {
                "step": "observe",
                "content": str(result)
            }

            messages.append({
                "role": "assistant",
                "content": json.dumps(observe_msg)
            })

        elif step_type == "output":
            print(f"✅ Result: {step.get('content')}")
            break
