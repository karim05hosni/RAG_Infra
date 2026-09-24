import json
import os

from google import genai

from app.eval.golden_dataset import add_eval_dataset, get_eval_dataset, pick_chunks_sample

create_eval_dataset_tool = {
    "type": "function",
    "name": "create_eval_dataset",
    "description": "A tool for creating an evaluation dataset.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "eval_data": {
                "type": "ARRAY",
                "description": "The evaluation dataset items to be created.",
                "items": {
                    "type": "OBJECT",
                    "description": "A single evaluation record.",
                    "properties": {
                        "chunk_id": {
                            "type": "STRING",
                            "description": "Unique identifier for the primary chunk."
                        },
                        "corpus_id": {
                            "type": "STRING",
                            "description": "Unique identifier for the corpus."
                        },
                        "question": {
                            "type": "STRING",
                            "description": "The evaluation question or query."
                        },
                        "answer": {
                            "type": "STRING",
                            "description": "The target or ground-truth answer."
                        },
                        "language": {
                            "type": "STRING",
                            "description": "Language of the content (e.g., 'english')."
                        },
                        "cross_lingual": {
                            "type": "BOOLEAN",
                            "description": "Indicates whether the question/answer pair is cross-lingual."
                        },
                        "chunk_text": {
                            "type": "STRING",
                            "description": "Text content of the primary chunk."
                        },
                        "source": {
                            "type": "STRING",
                            "description": "source of the question & answer , yours is 'LLM-model'"
                        },
                        "expected_chunks_ids": {
                            "type": "ARRAY",
                            "description": "List of expected chunk UUID strings relevant to this entry.",
                            "items": {
                                "type": "STRING"
                            }
                        }
                    },
                    "required": [
                        "chunk_id",
                        "corpus_id",
                        "question",
                        "answer",
                        "language",
                        "cross_lingual",
                        "chunk_text",
                        "source",
                        "expected_chunks_ids"
                    ]
                }
            }
        },
        "required": ["eval_data"]
    }
}
get_chunks_sample_tool = {
    "type": "function",
    "name": "get_chunks_sample",
    "description": "A tool for retrieving a sample of chunks for each source from the knowledge base.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "num_chunks_per_source": {
                "type": "INTEGER",
                "description": "The number of sample chunks to retrieve per source."
            }
        },
        "required": ["num_chunks_per_source"]
    }
}
get_eval_dataset_tool = {
    "type": "function",
    "name": "get_eval_dataset",
    "description": "A tool for retrieving evaluation dataset.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "limit": {
                "type": "INTEGER",
                "description": "The maximum number of evaluation dataset chunks to retrieve."
            }
        },
        "required": ["limit"]
    }
}

tools = [
    create_eval_dataset_tool,
    get_chunks_sample_tool,
    get_eval_dataset_tool
]
# session_store: dict[str, list] = {} # session_id => history_steps (list of previous steps in the conversation)


def execute_tool(tool_name, **tool_params):
    if tool_name == "create_eval_dataset":
        return add_eval_dataset(**tool_params)
    if tool_name == "get_eval_dataset":
        return get_eval_dataset(**tool_params)
    if tool_name == "get_chunks_sample":
        return pick_chunks_sample(**tool_params)
    else:
        raise ValueError(f"Tool '{tool_name}' is not recognized.")

genAi_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
system_instructions = """

You are an AI agent that builds a golden dataset by creating Q/As based on the knowledge base for evaluating the performance of a retrieval-augmented generation (RAG) pipeline.
The Questions must be a naive patient for eye doctor questions who is asking doctor's Assistant, a patient can ask about Dr.Amr's experience with treatments, surgeries, procedures and patient care, The answers must be based on the knowledge base ONLY.
You have access to three tools: `create_eval_dataset`, `get_eval_dataset`, and `get_chunks_sample`. Use these tools to create, retrieve, and sample evaluation dataset chunks as needed.
- `create_eval_dataset`: Use this tool to create new evaluation dataset chunks.
- `get_eval_dataset`: Use this tool to retrieve existing evaluation dataset chunks.
- `get_chunks_sample`: Use this tool to retrieve a sample of chunks for each source from the knowledge base.

"""
history = [

]
def eval_agent_loop(prompt, max_iterations=5):
    agent_final_response = None
    history.append({
        "type": "user_input",
        "content": [{"type": "text", "text": prompt}]
    })
    for iteration in range(max_iterations):
        print(f"Agent loop iteration {iteration + 1}/{max_iterations}")
        response = genAi_client.interactions.create(
            model=os.getenv("GEMINI_MODEL"),
            system_instruction=system_instructions,
            tools=tools,
            input=history,
            store=False
        )
        print(f"Agent response: {response}")

        # Always record every step first — including the final model_output
        for step in response.steps:
            history.append(step.model_dump())
            print(f"Step: {step}")

        tool_calls = [step for step in response.steps if step.type == "function_call"]
        if not tool_calls:
            agent_final_response = response.output_text
            break

        for step in tool_calls:
            try:
                print(f"Executing tool '{step.name}' with arguments: {step.arguments}")
                tool_result = execute_tool(step.name, **step.arguments)
                result_text = json.dumps(tool_result, default=str)
            except Exception as e:
                print(f"Error executing tool '{step.name}': {e}")
                result_text = json.dumps({"error": f"Error executing tool '{step.name}': {e}"})
                raise e
            history.append({
                "type": "function_result",
                "name": step.name,
                "call_id": step.id,
                "result": [{"type": "text", "text": result_text}],
            })
        # pass function call result to the next iteration of the agent loop
        
        agent_final_response = response.output_text

    return agent_final_response