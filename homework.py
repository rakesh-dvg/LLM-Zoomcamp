import os
import json
import minsearch
from dotenv import load_dotenv
from openai import OpenAI
from gitsource import GithubRepositoryDataReader, chunk_documents
from toyaikit.llm import OpenAIClient
from toyaikit.chat.runners import OpenAIResponsesRunner

load_dotenv()

# =====================================================================
# 0. SETUP GROQ CLIENT & COMPATIBILITY LAYER
# =====================================================================
openai_client = OpenAI(
   api_key=os.getenv("GROQ_API_KEY"),
   base_url="https://api.groq.com/openai/v1"
)

class GroqCompatibleClient(OpenAIClient):
    def send_request(self, chat_messages, tools=None, output_format=None):
        raw_messages = [msg if isinstance(msg, dict) else msg.model_dump() for msg in chat_messages]
        processed_messages = []
        allowed_keys = {"role", "content", "name", "tool_calls", "tool_call_id"}
        
        for msg in raw_messages:
            if msg.get("role") == "developer":
                msg["role"] = "system"
            cleaned_msg = {k: v for k, v in msg.items() if k in allowed_keys and v is not None}
            processed_messages.append(cleaned_msg)

        args = {"model": self.model, "messages": processed_messages}
        
        if tools and hasattr(tools, 'tools') and tools.tools:
            tool_list = []
            for tool_name, tool_obj in tools.tools.items():
                raw_schema = tool_obj if isinstance(tool_obj, dict) else getattr(tool_obj, "json_schema", tool_obj)
                if "function" in raw_schema and "parameters" in raw_schema["function"]:
                    params = raw_schema["function"]["parameters"]
                    desc = raw_schema["function"].get("description", "Search tool")
                elif "parameters" in raw_schema:
                    params = raw_schema["parameters"]
                    desc = raw_schema.get("description", "Search tool")
                else:
                    params = raw_schema
                    desc = "Search tool"

                if isinstance(params, dict):
                    params.pop("additionalProperties", None)

                tool_list.append({
                    "type": "function",
                    "function": {"name": tool_name, "description": desc, "parameters": params}
                })
            args["tools"] = tool_list
            args["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**args)
        if hasattr(response, "usage") and response.usage:
            response.usage.input_tokens = getattr(response.usage, "prompt_tokens", 0)
            response.usage.output_tokens = getattr(response.usage, "completion_tokens", 0)
            
        return response

# =====================================================================
# Q1. DOWNLOAD THE LESSON DATASET
# =====================================================================
print("--- Running Q1 ---")
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
files = reader.read()

documents = []
for file in files:
    documents.append(file.parse())

print(f"Q1 Answer (Total lesson pages): {len(documents)}\n")

# =====================================================================
# Q2. MINSEARCH INDEXING
# =====================================================================
print("--- Running Q2 ---")
index = minsearch.Index(text_fields=["content"], keyword_fields=["filename"])
index.fit(documents)

query = "How does the agentic loop keep calling the model until it stops?"
search_results = index.search(query, num_results=1)
print(f"Q2 Answer (First result filename): {search_results[0]['filename']}\n")

# =====================================================================
# Q3. RAG PROMPT TOKEN COUNTING
# =====================================================================
print("--- Running Q3 ---")
context_entries = [res['content'] for res in index.search(query, num_results=3)]
context = "\n\n".join(context_entries)

prompt_template = """
You are a course assistant. Answer the question using only the context provided.
Context: {context}
Question: {question}
""".strip()

prompt = prompt_template.format(context=context, question=query)

response_q3 = openai_client.chat.completions.create(
    model='llama-3.3-70b-versatile',
    messages=[{"role": "user", "content": prompt}]
)
print(f"Q3 Answer (Input tokens sent): {response_q3.usage.prompt_tokens}\n")

# =====================================================================
# Q4. CHUNKING
# =====================================================================
print("--- Running Q4 ---")
chunks = chunk_documents(documents, size=2000, step=1000)
print(f"Q4 Answer (Total chunks generated): {len(chunks)}\n")

# =====================================================================
# Q5. RAG WITH CHUNKING TOKEN COUNTING
# =====================================================================
print("--- Running Q5 ---")
chunk_index = minsearch.Index(text_fields=["content"], keyword_fields=["filename"])
chunk_index.fit(chunks)

chunk_context_entries = [res['content'] for res in chunk_index.search(query, num_results=3)]
chunk_context = "\n\n".join(chunk_context_entries)
chunk_prompt = prompt_template.format(context=chunk_context, question=query)

response_q5 = openai_client.chat.completions.create(
    model='llama-3.3-70b-versatile',
    messages=[{"role": "user", "content": chunk_prompt}]
)
print(f"Q3 Token Count: {response_q3.usage.prompt_tokens}")
print(f"Q5 Token Count (Chunked): {response_q5.usage.prompt_tokens}")
reduction = response_q3.usage.prompt_tokens / response_q5.usage.prompt_tokens
print(f"Q5 Answer: Chunked version is roughly {reduction:.1f}x fewer tokens.\n")

# =====================================================================
# Q6. AGENTIC LOOP COUNTER (FIXED)
# =====================================================================
print("--- Running Q6 ---")
search_call_count = 0

def search_tool_call(query: str) -> str:
    global search_call_count
    search_call_count += 1
    print(f"   -> [Agent Called Tool] Search call #{search_call_count} with query: '{query}'")
    results = chunk_index.search(query, num_results=3)
    return json.dumps([res['content'] for res in results])

# Initialize the message thread manually to ensure stability
messages = [
    {
        "role": "system", 
        "content": "You're a course teaching assistant. Answer the student's question using the search tool. Make multiple searches with different keywords before answering."
    },
    {
        "role": "user", 
        "content": "How does the agentic loop work, and how is it different from plain RAG?"
    }
]

print("Starting Agentic Loop processing natively...")

while True:
    # Query Groq natively using standard definitions
    response = openai_client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=messages,
        tools=[{
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the course lesson database chunks using keyword retrieval.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query keywords"}
                    },
                    "required": ["query"]
                }
            }
        }],
        tool_choice="auto"
    )
    
    assistant_msg = response.choices[0].message
    messages.append(assistant_msg)
    
    # Check if the agent wants to keep searching
    if assistant_msg.tool_calls:
        for tool_call in assistant_msg.tool_calls:
            # Parse arguments safely
            args = json.loads(tool_call.function.arguments)
            # Execute tool and increment counter
            tool_output = search_tool_call(args.get("query", ""))
            
            # Send results back to the agent thread
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": "search",
                "content": tool_output
            })
    else:
        # No more tool calls means the agent is ready to answer!
        print("\nAgent Final Answer Response:\n", assistant_msg.content)
        break

print(f"\nQ6 Answer (Total tool invocations): {search_call_count}")