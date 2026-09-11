import os
from typing import Annotated, List

from contextlib import asynccontextmanager
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from fastapi import FastAPI, File, UploadFile
from app.repositories import delete_all_collections
from app.ingestion import transcribe_video
from app.ingestion.service import ingest_files
from app.clients.postgres import init_pool, close_pool, get_conn
from app.retrieval import keywordSearch, RRF, semantic_search, hybrid_search
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting application...")
    
    init_pool()
    print("Postgres pool initialized")

    yield

    print("Closing application...")
    close_pool()

app = FastAPI(lifespan=lifespan)

@app.get('/')
def main():
    print(">>> ENTERED /")
    return {"message": "Hello from rag infra"}

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
        ingest_files(file_paths, language=language, chunk_size=10)
        return {"message": f"Successfully ingested {len(file_paths)} documents."}
    except Exception as e:
        raise e
    finally:
        for file_path in file_paths:
            if os.path.exists(file_path):
                os.remove(file_path)

@app.get("/search")
def search(query: str, language: str):
    try:
        return hybrid_search(query, language)
    except Exception as  e:
        print(f"Error occurred while searching documents: {e}")
        return {"error": "error accured while searching"}
    

@app.get("/format_qdrant_collections")
def format_qdrant_collections():
    delete_all_collections()


if __name__ == "__main__":
    main()
