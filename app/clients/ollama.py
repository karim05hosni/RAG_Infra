import time

from ollama import Client, EmbedResponse
import ollama
import os
from dotenv import load_dotenv
load_dotenv()

ollama_client = Client(host=os.getenv("Ollama_url"))
print(f"Connected to Ollama server at {os.getenv('Ollama_url')}")

def embed_text(chunks: list[str]) -> list[float]:
    max_retries = 3
    backoff = 5
    response: EmbedResponse = {}
    
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
    return response.embeddings