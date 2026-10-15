import time

from ollama import Client, EmbedResponse
import ollama


ollama_client = Client(host='http://192.168.1.9:11434')

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
            response = ollama_client.embed(
                model='bge-m3',
                input=chunks
            )
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