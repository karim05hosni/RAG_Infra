from app.retrieval import hybrid_search
from google import genai
from dotenv import load_dotenv
from datetime import datetime
import uuid
import os

load_dotenv()  # Load environment variables from .env file

hybrid_search_tool = {
    "type": "function",
    "name": "hybrid_search",
    "description": "A tool for performing hybrid search on knowledge base, It combines keyword search and semantic search to retrieve relevant information based on the user's query.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query string.",
            },
            "language": {
                "type": "string",
                "description": "The language of the query.",
            },
        },
        "required": ["query"],
    },
}
system_instructions = """
You are Dr. Amr's AI assistant that responds to users' queries.
Dr. Amr is an opthalmologist and has video content and publications related to opthalmology. 
You have access to a knowledge base that contains information from Dr. Amr's videos and publications. You can use the hybrid_search tool to retrieve relevant information from the knowledge base based on the user's query. 
When responding to users, you Must avoid providing diagnoses or medical advice. Instead, you should provide the relevant information from the knowledge base as an educational resource and suggest that the user consult Dr. Amr with a good CTA.
The knowledge base content is in both English and French, so you will need to search with 'english' and 'french' as the language parameter in the hybrid_search tool since keyword search  

Your tone should be professional, informative, and empathetic but not overly formal. Don't provide very detailed explanation, pick the most relevant information from the knowledge base and provide it in a concise manner. If you don't know the answer or it's not available in the knowledge base, you should say "I don't know" and suggest that the user consult Dr. Amr for more information.
client's input will be provided in the following format:
<userInput>user's input</userInput>
"""
tools = [
    hybrid_search_tool
]
existing_chats = {}

def generate_session_id(IP_address):
    """generate a unique session ID from request IP address"""
    session_id = uuid.uuid5(uuid.NAMESPACE_DNS, IP_address)
    return session_id

def execute_tool(tool_name, **tool_params):
    if tool_name == "hybrid_search":
        return hybrid_search(**tool_params)
    else:
        raise ValueError(f"Tool '{tool_name}' is not recognized.")
    
genAi_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def agent_loop(prompt, IP_address, max_iterations=5):
    session_id = generate_session_id(IP_address)
    print(f"Starting agent loop with session ID: {session_id}")
    if session_id not in existing_chats:
        existing_chats[session_id] = None  # Initialize a new chat session if it doesn't exist
    current_input = '<userInput>' + prompt + '</userInput>'
    agent_final_response = None
    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}/{max_iterations} for session ID: {session_id}")
        calling_llm_start_time = datetime.now()
        response = genAi_client.interactions.create(
            model='gemini-3.5-flash-lite',
            input=current_input,
            tools=tools,
            system_instruction=system_instructions,
            previous_interaction_id=existing_chats[session_id] if existing_chats[session_id] else None,
            generation_config={
                "temperature": 0.4,
            },
        )
        calling_llm_end_time = datetime.now()
        print(f"LLM call duration: {calling_llm_end_time - calling_llm_start_time}")
        existing_chats[session_id] = response.id
        print(f"response: {response.steps}")
        tool_calls = [step for step in response.steps if step.type == "function_call"]
        if not tool_calls:
            agent_final_response = response.output_text
            break  # Exit the loop if no tools were called
        function_results = []
        for step in response.steps:
            if step.type == "function_call":
                tool_name = step.name
                tool_params = step.arguments
                try:
                    tool_result = execute_tool(tool_name, **tool_params)
                    function_results.append({"tool_name": tool_name, "result": tool_result})
                except Exception as e:
                    function_results.append({"tool_name": tool_name, "error": str(e)})
                # Update the current input for the next iteration
                current_input = f"observation from previous tool execution: {function_results[-1]}. Now, please provide the next action or response based on this observation."
        agent_final_response = response.output_text
    return agent_final_response