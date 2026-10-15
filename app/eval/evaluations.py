"""
Evaluation functions for the RAG system.
- faithfulness(llm_answer, retrieved_chunks) -> float, is LLM answer relevant to the retrieved chunks?
- context_recall(gt_answer, retrieved_chunks) -> float, was all relevant context retrieved for the question?
- context_precision(question, retrieved_chunks) -> float, were relevant chunks retrieved for the question ranked high?
"""
import json
import re
from transformers import pipeline
import outlines
from pydantic import BaseModel

from app.shared import transformer

class ClaimList(BaseModel):
    claims: list[str]

from .golden_dataset import get_eval_dataset
from app.clients.ollama import ollama_client


# Specify your remote server's IP address and port
# client = Client(host='http://192.168.1.9:11434')
client = ollama_client
from sentence_transformers import CrossEncoder
nli_model = CrossEncoder("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
def test_ollama_connection():
    response = client.chat(
        model='hf.co/Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M',
        messages=[{'role': 'user', 'content': 'Hello server!'}],
        format='json'
    )
    print(response['message']['content'])


def extract_claims(text: str, max_retries: int = 2) -> list[str]:
    system_prompt = f"""You are a helpful assistant that extracts claims from text. 
    Break into the smallest possible standalone factual claims.
    Each claim must state exactly ONE fact. Split compound sentences (joined by "and", "but", "in addition to") into separate claims.
    Remove any markdown formatting (bullets, asterisks).
    Exclude calls to action, greetings, links, or instructions to the reader.
    
    Return the claims as a JSON array of strings.
    example:
    input: "The sky is blue. Water is wet."
    output: ["The sky is blue.", "Water is wet."]
    """
    user_prompt = f"""Extract all claims from the following text:\n\n{text}\n\n"""
    for attempt in range(max_retries):
        response = client.chat(
            model='hf.co/Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            format=ClaimList.model_json_schema()
        )
        try:
            claims = ClaimList.model_validate_json(response.message.content).claims
            print(f"claims length: {len(claims)}, Extracted claims: {claims}")
            return claims
        except json.JSONDecodeError as e:
            print(f"Attempt {attempt + 1}: Failed to parse claims from response: {e}")
            continue
    print("Failed to extract claims after all retries.")
    return text

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

tokenizer = AutoTokenizer.from_pretrained(model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(model_name)

def check_entailment(premise: str, hypothesis: str) -> str:
    # Step 1: tokenize the pair together — the model expects premise + hypothesis as one input
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)

    # Step 2: run the model, get raw scores (logits) for the 3 classes
    with torch.no_grad():
        logits = nli_model(**inputs).logits

    # Step 3: turn logits into probabilities (optional, but useful for thresholds later)
    probs = torch.softmax(logits, dim=1)[0]
    print(f"Entailment probabilities: {probs.tolist()}")  # for debugging
    # Step 4: figure out which index means what — DON'T assume order, read it from the nli_model config
    id2label = nli_model.config.id2label
    predicted_class_id = probs.argmax().item()
    return id2label[predicted_class_id]  # returns "entailment", "neutral", or "contradiction"

def clean_for_nli(chunk_text: str) -> str:
    # strip "From X.pdf: " / "From X.mp4: " prefix before NLI scoring
    return re.sub(r"^From [^:]+:\s*", "", chunk_text)

def verify_claim_against_chunks(claim: str, chunks: list[str]) -> tuple[bool, str | None]:
    """Returns (is_supported, supporting_chunk_or_None)."""
    for chunk in chunks:
        cleaned_chunk = clean_for_nli(chunk)
        print(f"Checking entailment for claim: '{claim}' against chunk: '{cleaned_chunk}'")
        label = check_entailment(premise=cleaned_chunk, hypothesis=claim)
        print(f"Entailment result: {label}")
        if label == "entailment":
            return True, chunk  # found support, stop early — no need to check remaining chunks
        # if label == "neutral":
            # TODO: route to a second check
            # pass
    print(f"No supporting chunk found for claim: '{claim}'")
    return False, None  # no chunk entailed this claim

def decompose_chunks_into_atomic(chunks: list[str]) -> list[str]:
    single_fact_chunks = []
    system_prompt = f"""
    Break the multi-fact passage into single-fact statements,  
    you should return a JSON list of strings, each string is a single-fact statement.
    if the passage is already a single-fact statement, return it as a single-item list.
    """
    for chunk in chunks:
        user_prompt = f"""Break the following passage into single-fact statements:\n\n{chunk}\n\n"""
        response = client.chat(
            model='hf.co/Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            format=ClaimList.model_json_schema()
        )
        try:
            atomic_claims = ClaimList.model_validate_json(response.message.content).claims
            print(f"Decomposed chunk into {len(atomic_claims)} atomic claims.")
            single_fact_chunks.extend(atomic_claims)
        except json.JSONDecodeError as e:
            print(f"Failed to parse atomic claims from response: {e}")
            single_fact_chunks.append(chunk)  # fallback to original chunk if parsing fails
    return single_fact_chunks
def faithfulness(answer: str, chunks: list[str]) -> dict:
    claims = extract_claims(answer)  # still the one LLM call for decomposition
    if not claims:
        return {"score": 1.0, "details": []}

    # decompose multi-fact chunks into atomic single-fact statements for more granular evaluation
    atomic_chunks = decompose_chunks_into_atomic(chunks)
    details = []
    for claim in claims:
        supported, source_chunk = verify_claim_against_chunks(claim, atomic_chunks)
        details.append({
            "claim": claim,
            "supported": supported,
            "source_chunk": source_chunk,
        })
    print(f"Faithfulness evaluation details: {details}")
    score = sum(d["supported"] for d in details) / len(details)
    return {"score": score, "details": details}