import json
from logging import config
import os

from google import genai
from pydantic import BaseModel
from typing import Dict, List
from app.retrieval.service import get_neighboring_chunks

get_neighbor_chunks_tool = {
    "type": "function",
    "name": "get_neighbor_chunks",
    "description": "Fetches neighboring chunks of a given chunk ID. This is useful for retrieving additional context when generating question-answer pairs from a passage. The function returns the neighboring chunks in order, starting with the chunk before the given chunk and ending with the chunk after it.",
    "parameters": {
        "type": "object",
        "properties": {
            "chunk_id": {
                "type": "string",
                "description": "The ID of the chunk for which to fetch neighboring chunks.",
            },
            "num_neighbors": {
                "type": "integer",
                "description": "The number of neighboring chunks to fetch on each side of the given chunk. For example, if num_neighbors is 2, the function will return 2 chunks before and 2 chunks after the given chunk.",
                "default": 2,
            },
        },
        "required": ["chunk_id", "num_neighbors"],
    },

}

genAi_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class QAPair(BaseModel):
    chunk_id: str
    question: str
    answer: str
    language: str = "en"
    cross_lingual: bool = False
    expected_chunks_ids: List[str]

class ChunkResult(BaseModel):
    chunk_id: str
    skip: bool
    reason: str
    qa_pairs: List[QAPair]

class ChunkQAResponse(BaseModel):
    chunks: List[ChunkResult]
system_instructions = """
    You are generating evaluation question-answer pairs for each passage of Dr. Amr's knowledge base.
    The question must sound like a naive patient asking Dr. Amr's assistant about his experience with treatments, surgeries, procedures, or patient care.
    The answer must be grounded ONLY in the passage below — do not add outside information.
    If the passage has no clear question a patient would ask (e.g. it's an intro, outro, or fragment), flag it as "skip" and provide a reason in the "reason" field. Do not generate any Q/A pairs for skipped passages.
    The passage may be in French or English, and the question and answer must be in the same language as the passage.
    
    If this passage alone is enough to ask and answer a complete question, do so. 
    The number of Q/A pairs for each passage should be 1 to 2, unless the passage is too short or too long, in which case you may generate 0 or 4 Q/A pairs respectively.
    If it references something incomplete (e.g. 'as I mentioned earlier', a cut-off sentence, a partial thought), call get_neighbor_chunks to get more context before answering.

    Report exactly which chunk_ids you used "including the current chunk" to build your answer in expected_chunks_ids list of the final answer.
    
    You will receive a batch of chunks in the following format:
    <chunks>
        <chunk>
            <chunk_text>{chunk_text}</chunk_text>
            <chunk_id>{chunk_id}</chunk_id>
        </chunk>
    </chunks>

    Respond only in JSON mapping each chunk_id to its array of Q/A pairs inside the 'chunks' key:
    {
        
        "chunks": [
            {
                "chunk_id": "{chunk_id}",
                "skip": false,
                "reason": "",
                "qa_pairs": [
                    {
                        "chunk_id": "{chunk_id}",
                        "question": "{question}",
                        "answer": "{answer}",
                        "language": "{language}",
                        "cross_lingual": false,
                        "expected_chunks_ids": ["{chunk_id}", "{neighbor_chunk_id_1}", "{neighbor_chunk_id_2}"]
                    }
                ]
            }
        ]
    }
"""
def generate_chunk_QA(chunks: list[dict], max_iterations=2)-> dict:
    print(f"building chunks input for the model .....")
    # build chunks input for the model
    chunks_input = "<chunks>"
    for chunk in chunks:
        chunks_input += f"<chunk><chunk_text>{chunk['text']}</chunk_text><chunk_id>{str(chunk['chunk_id'])}</chunk_id></chunk>"
    chunks_input += "</chunks>"
    print(f"adding chunks input to the model: {chunks_input[:40]}...")  # print first 100 chars for brevity
    history = [
        {"type": "user_input", "content": [{"type": "text", "text": chunks_input}]}
    ]
    final_response = ''
    for iteration in range(max_iterations + 1):
        print(f"calling LLM ...")
        response = genAi_client.interactions.create(
            model=os.getenv("GEMINI_MODEL"),
            system_instruction=system_instructions,
            tools=[get_neighbor_chunks_tool],
            store=False,
            input=history,
            response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": ChunkQAResponse.model_json_schema()
            }
        )
        for step in response.steps:
            history.append(step.model_dump())
        
        tool_calls = [step for step in response.steps if step.type == "function_call"]
        if not tool_calls:
            final_response = response.output_text
            break
        for step in tool_calls:
            try:
                tool_result = execute_tool(step.name, **step.arguments)
                result_text = json.dumps(tool_result, default=str)
                history.append({
                    "type": "function_result",
                    "name": step.name,
                    "call_id": step.id,
                    "result": [{"type": "text", "text": result_text}],
                })
            except Exception as e:
                print(f"Error executing tool {step.name}: {e}")
                history.append({
                    "type": "function_result",
                    "name": step.name,
                    "call_id": step.id,
                    "result": [{"type": "text", "text": f"Error: {str(e)}"}],
                })
    final_response = response.output_text
    return final_response

def execute_tool(tool_name, **tool_params):
    if tool_name == "get_neighbor_chunks":
        return get_neighboring_chunks(**tool_params)
    else:
        raise ValueError(f"Tool '{tool_name}' is not recognized.")
