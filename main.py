import os
import secrets
from typing import Annotated, List
from app import retrieval
from app.eval import service as eval_service
from app.eval import evaluations
from app.eval.golden_dataset import add_eval_dataset, get_eval_dataset, pick_chunks_sample
from contextlib import asynccontextmanager

from app.shared import helpers
import uvicorn
from litellm import token_counter

from fastapi.middleware.cors import CORSMiddleware
from fastapi import Body, FastAPI, File, Request, Response, UploadFile
from app.repositories import delete_all_collections
from app.ingestion import transcribe_video
from app.ingestion.service import ingest_files
from app.ingestion import service as ingestion_service
from app.clients.postgres import init_pool, close_pool, get_conn
from app.retrieval import keywordSearch, RRF, semantic_search, hybrid_search
from app.agent import agent_loop
from app.shared.session_manager import session_exists, create_session
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting application...")
    
    init_pool()
    print("Postgres pool initialized")

    yield

    print("Closing application...")
    close_pool()

app = FastAPI(lifespan=lifespan)
# 1. Define the allowed origins (front-end URLs)
origins = [
    "http://127.0.0.1:5173",   # Vite local development
    "http://localhost:5173/"
]

# 2. Add the CORSMiddleware to your application
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # Allow specific origins
    allow_methods=["*"],             # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],             # Allow all request headers
)
# gets documents inputs from http request and ingests them into the system
@app.post("/ingest")
def ingest_documents(language: str, files: Annotated[List[UploadFile], File()]):
    try:
        file_paths= []
        for file in files:
            # create doc id
            doc_id = file.filename
            # save file to docs
            file_path = f"docs/{doc_id}"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(file.file.read())
            file_paths.append(file_path)
        # ingest files
        print(f"Ingesting files: {file_paths} with language: {language}")
        ingest_files(file_paths)
        return {"message": f"Successfully ingested {len(file_paths)} documents."}
    except Exception as e:
        raise e
    finally:
        for file_path in file_paths:
            if os.path.exists(file_path):
                os.remove(file_path)

@app.get("/search")
def search(query: str):
    try:
        return hybrid_search(query)
    except Exception as  e:
        print(f"Error occurred while searching documents: {e}")
        return {"error": "error accured while searching"}
    

@app.get("/format_qdrant_collections")
def format_qdrant_collections():
    delete_all_collections()
    
@app.get("/semantic_search")
def semantic_search(query):
    return retrieval.semantic_search(query)

@app.get("/keyword_search")
def keyword_search(query, language, top_k=70):
    return retrieval.keywordSearch(query, language, top_k)
@app.post("/agent")
def agent(request: Request, response: Response, prompt: str = Body(..., embed=True)):
    # mock sessionId for testing
    # sessionId = "test_session_1"
    session_id = request.cookies.get("session_id")
    if not session_id or not session_exists(session_id):
        session_id = secrets.token_urlsafe(32)
        create_session(session_id)
        response.set_cookie(
            "session_id", session_id,
            httponly=True, secure=True, samesite="lax",
            max_age=7 * 24 * 3600
        )
    result = agent_loop(prompt, session_id, 10)
    return {"response": result}

@app.get("/eval")
def eval(num_samples_per_source: int = 5):
    sampled_chunks = pick_chunks_sample(num_samples_per_source)
    print(f"Sampled Chunks: {sampled_chunks}")
    return sampled_chunks

@app.post("/eval/add")
def add_eval_dataset(eval_data: list[dict] = Body(..., embed=True)):
    inserted = add_eval_dataset(eval_data)
    return {"inserted": inserted}

@app.get("/eval/get")
def eval_dataset(limit: int = 10):
    return get_eval_dataset(limit)

@app.post("/eval/generate")
def generate_eval_dataset(jobId: str = None):
    return eval_service.generate_eval_dataset(jobId=jobId)

@app.get("/get-neighbor-chunks")
def get_neighbor_chunks(chunk_id: str, num_neighbors: int = 2):
    return retrieval.get_neighboring_chunks(chunk_id, num_neighbors)

@app.get("/extract-claims")
def get_claims(text):
    return evaluations.extract_claims(text)

@app.get("/test-ollama-connection")
def test_ollama_connection():
    evaluations.test_ollama_connection()

@app.post("/test-faithfulness")
def test_faithfulness(llm_answer: str = Body(...), retrieved_chunks: list[str] = Body(...)):
    return evaluations.faithfulness(llm_answer, retrieved_chunks)

@app.get("/reembedd-chunks")
def reembedd_chunks(embedding_model: str):
    return ingestion_service.reembedd_chunks(embedding_model)

@app.get("/count-tokens")
def count_tokens(input = Body(..., embed=True)):
    token_count = token_counter(model='bge-m3', text=input)
    return {"token_count": token_count}

@app.get("/sanitize-fr-text")
def sanitize_fr_text(text: str):
    return {"sanitized_text": helpers.sanitize_fr_text(text)}