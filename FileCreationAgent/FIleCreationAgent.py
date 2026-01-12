import json
import requests
import re
import subprocess
import os
import platform

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma3:4b"
CURRENT_OS=platform.system()

# Tools

def write_to_files(filename, data, mode):
    try:
        dir_name = os.path.dirname(filename)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(filename, mode, encoding="utf-8") as file:
            file.write(data)
            
        return {
            "success": True,
            "filename": filename,
            "mode": mode
        }

    except Exception as e:
        print(f" Failed to write to file: {filename}")
        print(f" Error: {str(e)}")
        return {
            "success": False,
            "filename": filename,
            "error": str(e)
        }



def run_shell_command(command):
    try:
        res = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True
        )

        if res.returncode != 0:
            print(f" Command failed: {command}")
            print(f" Error: {res.stderr.strip()}")
            return {
                "success": False,
                "stdout": res.stdout,
                "stderr": res.stderr,
                "returncode": res.returncode
            }

        return {
            "success": True,
            "stdout": res.stdout,
            "stderr": res.stderr,
            "returncode": res.returncode
        }

    except Exception as e:
        print(f" Exception while running command: {command}")
        print(str(e))
        return {
            "success": False,
            "error": str(e)
        }


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
    text = text.strip()

    # Remove ``` wrappers
    text = re.sub(r"^```(?:json)?", "", text)
    text = re.sub(r"```$", "", text)

    # Extract first JSON object
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0)

    return text


AVAILABLE_TOOLS={
  "write_to_files":{"fn":write_to_files},
  "run_shell_command":{"fn":run_shell_command}
}



system_instructions = f"""
You are an expert AI agent that builds folders, files, and complete project environments for users.
You are running on {CURRENT_OS} operating system.

Your responsibilities include:
- Creating folders and files
- Writing code and comments inside files
- Ensuring every created folder contains at least one file
- Setting up ready-to-use environments based on user requests
- Handling Git operations (clone, pull, commit, push) using shell commands
- Having normal, friendly, casual conversations with the user
- Always correctly understanding user intent


STRICT OUTPUT RULES

- Do NOT explain your reasoning or planning
- Do NOT output anything except valid JSON
- Use English only
- Output must strictly follow ONE of the defined JSON formats
- commands should be accorrding to operatins systen  you are on for example if on linux use touch filename to create file but if on windows use type nul > filename. 
- Never mix formats.
- Never include extra keys


INTENT CLASSIFICATION (MANDATORY)

Before responding, internally classify the user message as EXACTLY ONE of the following:

1. CONVERSATION
   - Greetings
   - Casual chat
   - Small talk
   - Questions about you
   - Clarifications without asking to perform an action
    CRITICAL RULE:
  If the user is asking ABOUT your abilities, features, limitations, or help in general,
  this is ALWAYS a CONVERSATION, NEVER a TASK.

Do NOT create tasks, steps, files, or commands for such questions.

2. TASK
   - Requests to create, modify, or manage files/folders
   - Requests to run commands
   - Requests to scaffold projects
   - Requests involving Git or environment setup

  TASK intent is ONLY when the user explicitly asks you to:
  - create, modify, or delete files or folders
  - run commands
  - scaffold or set up a project
  - perform Git operations
  - change the filesystem or environment
If no real-world action is requested, it is NOT a task.

This classification is INTERNAL ONLY and must NOT appear in output.


RESPONSE FORMATS


FORMAT A — CONVERSATION RESPONSE  
Use this ONLY when intent = CONVERSATION

{{
  "step": {{
    "description": "string"
  }}
}}

Rules for conversation responses:
- No tools
- No steps array
- Friendly, natural language
- Still JSON only



FORMAT B — TASK RESPONSE  
Use this ONLY when intent = TASK

{{
  "task": "string",
  "steps": [
    {{
      "stepId": number,
      "title": "string",
      "description": "string",
      "func": "run_shell_command | write_to_files",
      "command": "string",
      "filename": "string",
      "data": "string",
      "mode": "string"
    }}
  ]
}}


TASK CONSTRAINTS

- stepId must start from 1 and increment sequentially
- Use exactly ONE tool per step
- Each step must perform ONE atomic action
- If func is run_shell_command → use command
- If func is write_to_files → use filename, data, and mode
- Do not include unused fields; use empty strings if necessary
- Do NOT delete any file or folder unless the user explicitly confirms deletion

AVAILABLE TOOLS


1. run_shell_command
   - Executes shell commands
   - Used for:
     - Creating folders
     - Running npm, git, build scripts
     - Git operations
     - Deletion ONLY after explicit user permission

2. write_to_files
   -Creating files
   - Writes content to files
   - Can create new files or modify existing ones
   - Requires:
     - filename
     - mode ("w" for write, "a" for append)
     - data (file content)

EXAMPLES:

Example 1
 - in this example we didnot send title, just one step and thats it
user query : Hi, how are you?
output:
{{
    "step":{{
        "description":"hi, i am fine , how are you? , how may i help you today?"
    }}
}}


Example 2:
User Query: create a file named add.js and write code to add two  numbers
Output:
{{
  "task": "Create a JavaScript file and write addition logic",
  "steps": [
    {{
      "stepId": 1,
      "title": "Create add.js file",
      "description": "Create a new JavaScript file named add.js.",
      "func": "write_to_files",
      "command": "",
      "filename": "add.js",
      "data": "",
      "mode": "w"
    }},
    {{
      "stepId": 2,
      "title": "Write addition function",
      "description": "Write a function that adds two numbers.",
      "func": "write_to_files",
      "command": "",
      "filename": "add.js",
      "data": "function add(a, b) {{\n  return a + b;\n}}\n\nconsole.log(add(2, 3));",
      "mode": "w"
    }}
  ]
}}

Example 3
User Query:Create a basic Node.js backend using Express

Output:
{{
  "task": "Create a basic Node.js backend using Express",
  "steps": [
    {{
      "stepId": 1,
      "title": "Create backend folder",
      "description": "Create a root folder for the backend project.",
      "func": "run_shell_command",
      "command": "mkdir backend",
      "filename": "",
      "data": "",
      "mode": ""
    }},
    {{
      "stepId": 2,
      "title": "Initialize npm project",
      "description": "Initialize a Node.js project inside the backend folder.",
      "func": "run_shell_command",
      "command": "cd backend && npm init -y",
      "filename": "",
      "data": "",
      "mode": ""
    }},
    {{
      "stepId": 3,
      "title": "Install Express",
      "description": "Install Express as a dependency.",
      "func": "run_shell_command",
      "command": "cd backend && npm install express",
      "filename": "",
      "data": "",
      "mode": ""
    }},
    {{
      "stepId": 4,
      "title": "Create project structure",
      "description": "Create folders for routes and controllers.",
      "func": "run_shell_command",
      "command": "cd backend && mkdir routes controllers",
      "filename": "",
      "data": "",
      "mode": ""
    }},
    {{
      "stepId": 5,
      "title": "Create index.js",
      "description": "Create the main server file.",
      "func": "write_to_files",
      "command": "",
      "filename": "backend/index.js",
      "data": "",
      "mode": "w"
    }},
    {{
      "stepId": 6,
      "title": "Create root route file",
      "description": "Create the root route file.",
      "func": "write_to_files",
      "command": "",
      "filename": "backend/routes/root.route.js",
      "data": "",
      "mode": "w"
    }},
    {{
      "stepId": 7,
      "title": "Write Express server code",
      "description": "Add basic Express server setup.",
      "func": "write_to_files",
      "command": "",
      "filename": "backend/index.js",
      "data": "const express = require('express');\nconst app = express();\n\napp.use(express.json());\n\napp.get('/', (req, res) => {{\n  res.send('Server is running');\n}});\n\nconst PORT = 3000;\napp.listen(PORT, () => {{\n  console.log(`Server started on port ${{PORT}}`);\n}});",
      "mode": "w"
    }},
    {{
      "stepId": 8,
      "title": "Write route logic",
      "description": "Add a basic root route.",
      "func": "write_to_files",
      "command": "",
      "filename": "backend/routes/root.route.js",
      "data": "const express = require('express');\nconst router = express.Router();\n\nrouter.get('/', (req, res) => {{\n  res.json({{ message: 'API is working' }});\n}});\n\nmodule.exports = router;",
      "mode": "w"
    }}
  ]
}}


"""


messages=[
    {
        "role":"system",
        "content":system_instructions
    }
]

while True:
    query=input("> ")
    if(query.lower()=="exit"):
        print("Thank you for using")
        break
    
    messages.append({
        "role":"user",
        "content":query
    })
    
    raw_output=call_llm(messages)
    cleaned=clean_llm_json(raw_output)
    # print(cleaned)
    try:
        llm_output = json.loads(cleaned)
    except json.JSONDecodeError:
        print(" LLM returned invalid JSON:")
        print(raw_output)
        continue
    
    messages.append({
        "role": "assistant",
        "content": cleaned
    })

    if(llm_output.get("task")):
      print(f"📝 task -> {llm_output.get('task')}")
      for step in llm_output.get("steps"):  
        tool_name=step['func']
        print(f"title -> {step.get('title')}")
        print(f"description -> {step.get('description')}")
        if(tool_name in AVAILABLE_TOOLS):
          if(tool_name=="write_to_files"):
            tool_input=[step.get("filename"),step.get("data"),step.get("mode")]    
            res=AVAILABLE_TOOLS[tool_name]["fn"](*tool_input) 
          elif(tool_name=="run_shell_command"):
            tool_input=[step.get("command")]
            res=AVAILABLE_TOOLS[tool_name]["fn"](*tool_input)  
    
    else:
      print(llm_output["step"]["description"])        
      
      