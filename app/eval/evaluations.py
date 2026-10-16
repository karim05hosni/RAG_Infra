"""
Evaluation functions for the RAG system.
- faithfulness(llm_answer, retrieved_chunks) -> float, is LLM answer relevant to the retrieved chunks?
- context_recall(gt_answer, retrieved_chunks) -> float, was all relevant context retrieved for the question?
- context_precision(question, retrieved_chunks) -> float, were relevant chunks retrieved for the question ranked high?
"""
import json
import re
import time
from transformers import pipeline
import outlines
from pydantic import BaseModel, Field

from app.shared import transformer

class ClaimList(BaseModel):
    claims: list[str]
class ClaimVerification(BaseModel):
    entailments: list[str] = Field(default_factory=list)
    neutral: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)

class NeutralClaim(BaseModel):
    entailments: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    
class LLMJudge(BaseModel):
    judgments: dict[str, NeutralClaim]  # {claim->{entailments->[], contradiction->[]}, ...}

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

def neutral_score_judge(claims_context_pairs: dict[str, list]) -> dict[str, NeutralClaim]:
    # Build a stable id for each claim, so we don't have to match on
    # text the LLM might retype slightly differently.
    claim_id_to_text = {}
    payload = {}

    claim_index = 0
    for claim_text, chunks in claims_context_pairs.items():
        claim_id = f"claim_{claim_index}"
        claim_id_to_text[claim_id] = claim_text
        payload[claim_id] = chunks
        claim_index += 1

    system_prompt = """
    you'll be given a list of claim ids, each mapped to a list of context snippets.
    the NLI model already scored all given pairs as "neutral", but you need to judge if the claim is actually supported by its context.
    it's a multi-lingual medical context, so be careful and check if the claim is actually supported.

    return JSON shaped like:
    {"claim_0": {"entailments": [...], "contradictions": [...]}, "claim_1": {...}, ...}
    Use the exact claim ids given to you as keys. Do not invent new ids and do not skip any.
    """
    user_prompt = f"Judge the following claims:\n\n{json.dumps(payload)}\n\n"

    max_retries = 3
    response = None
    for attempt in range(max_retries):
        try:
            response = client.chat(
                model='hf.co/Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M',
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                format=LLMJudge.model_json_schema()
            )
            print(f"Neutrality judge response: {response.message}")
            break
        except Exception as e:
            print(f"Attempt {attempt + 1}: Error during LLM call: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(2 ** attempt)
            else:
                raise e

    try:
        parsed = LLMJudge.model_validate_json(response.message.content)
        judgments_by_id = parsed.judgments
    except json.JSONDecodeError as e:
        print(f"Failed to parse neutrality judgments from response: {e}")
        raise e

    # Map the judge's answers back to the original claim text ourselves,
    # instead of trusting the model to echo the claim text correctly.
    results_by_claim_text = {}
    for claim_id, claim_text in claim_id_to_text.items():
        if claim_id in judgments_by_id:
            results_by_claim_text[claim_text] = judgments_by_id[claim_id]
        else:
            print(f"WARNING: judge did not return a result for {claim_id!r} ({claim_text!r})")
            results_by_claim_text[claim_text] = NeutralClaim()

    return results_by_claim_text
def verify_claim_against_chunks(claim: str, chunks: list[str]) -> ClaimVerification:
    """Returns (is_supported, supporting_chunk_or_None)."""
    entailments = []
    contradictions = []
    neutral = []
    for chunk in chunks:
        cleaned_chunk = clean_for_nli(chunk)
        print(f"Checking entailment for claim: '{claim}' against chunk: '{cleaned_chunk}'")
        label = check_entailment(premise=cleaned_chunk, hypothesis=claim)
        print(f"Entailment result: {label}")
        if label == "entailment":
            entailments.append(chunk)
        elif label == "contradiction":
            contradictions.append(chunk)
        else:
            neutral.append(chunk)
    return ClaimVerification(entailments=entailments, contradictions=contradictions, neutral=neutral)

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
    claims = extract_claims(answer) 
    if not claims:
        return {"score": 1.0, "details": []}

    # decompose multi-fact chunks into atomic single-fact statements for more granular evaluation
    atomic_chunks = decompose_chunks_into_atomic(chunks)
    details = []
    claims_verification_results: dict[str, ClaimVerification] = {}
    for claim in claims:
        claim_verification = verify_claim_against_chunks(claim, atomic_chunks)
        claims_verification_results[claim] = claim_verification
    # prepare neutral claims for LLM judge
    neutral_claims_context_pairs = {}
    for claim, verification in claims_verification_results.items():
        neutral_claims_context_pairs[claim] = verification.neutral
    # use LLM judge to filter in the supported neutral claims
    judge_results = {}
    if neutral_claims_context_pairs:
        judge_results = neutral_score_judge(neutral_claims_context_pairs)
    # erase neutrals and add entailments & contradictions for every claim in claims_verification_results
    print(f"claims verification results: {len(claims_verification_results)}")
    print(f"claims: {len(claims)}")
    for claim, verification in claims_verification_results.items():
        NLI_supported  = verification.entailments
        NLI_contradictions = verification.contradictions
        LLM_supported = {}
        LLM_contradictions = {}
        if judge_results:
            LLM_supported = judge_results[claim].entailments
            LLM_contradictions = judge_results[claim].contradictions
        # combine NLI and LLM results
        supported = list(set(NLI_supported + LLM_supported))
        contradictions = list(set(NLI_contradictions + LLM_contradictions))
        # erase neutrals
        claims_verification_results[claim] = ClaimVerification(entailments=supported, contradictions=contradictions, neutral=[])
        details.append({
            "claim": claim,
            "supported": len(supported) > 0,
            "entailments": supported,
            "contradictions": contradictions
        })
        
    print(f"Faithfulness evaluation details: {details}")
    score = sum(d["supported"] for d in details) / len(details)
    return {"score": score, "details": details}