import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from fastapi import FastAPI
from app.repositories import vector_repository
from app.ingestion import transcribe_video
app = FastAPI()

@app.get('/')
def main():
    return {"message": "Hello from rag infra"}

def agent(prompt: str):
    return ''


if __name__ == "__main__":
    main()
