from app.repositories.chunk_repository import fetch_chunks_by_source, fetch_all_sources
from .golden_dataset import pick_chunks_sample

def run_eval():
    result = {}
    # Placeholder for the evaluation logic
    print("Running evaluation...")
    # retrieve sources meta
    sources_meta = fetch_all_sources()
    # retrieve all chunks
    for source in sources_meta:
        source_id = source['source_id']
        all_chunks = fetch_chunks_by_source(source_id=source_id)
        picked_chunks = pick_chunks_sample(all_chunks, n=5)
        result[source_id] = picked_chunks
    return result