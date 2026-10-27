import time

from ollama import Client, EmbedResponse
import ollama
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from app.shared.transformer import sentence_transformer_embedding_models
load_dotenv()

ollama_client = Client(host=os.getenv("Ollama_url"))
print(f"Connected to Ollama server at {os.getenv('Ollama_url')}")

def embed_text(chunks: list[str], model: str = 'bge-m3', runtime: str = 'ollama') -> list[float]:
    max_retries = 3
    backoff = 5
    response: EmbedResponse = {}
    final_embeddings: list[float] = []
    if runtime == 'ollama':
        for attempt in range(max_retries):
        # check connection to Ollama server
            try:
                print("Checking connection to Ollama server...")
                print(ollama_client.list())
            except Exception as e:
                print(f"Error connecting to Ollama server: {e}")
                raise e
            try:
                ollama_embed_start_time = time.time()
                response = ollama_client.embed(
                    model='bge-m3',
                    input=chunks
                )
                ollama_embed_end_time = time.time()
                print(f"Ollama embed call duration: {ollama_embed_end_time - ollama_embed_start_time} seconds")
                print(f"Embedding res: {len(response.embeddings[0])}")
                break  # Exit the retry loop if successful
            except ollama._types.ResponseError as e:
                print(f"Error connecting to Ollama server: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {backoff} seconds...")
                    time.sleep(backoff * (attempt + 1))  # Exponential backoff
                    # backoff *= 2  # Exponential backoff
                else:
                    raise e
        final_embeddings.extend(response.embeddings)
        return final_embeddings
    if runtime == 'sentence-transformers':
        model = sentence_transformer_embedding_models.get('bf16').get('BAAI/bge-m3')
        if model is None:
            raise ValueError(f"Model {model} not found in sentence_transformer_embedding_models.")
        final_embeddings = model.encode(chunks)
        return final_embeddings
