import random
from app.repositories.chunk_repository import fetch_all_sources, fetch_chunks_by_source
from app.repositories.eval_set_repository import insert_eval_dataset, fetch_eval_dataset

def pick_chunks_sample(num_chunks_per_source=5):
    """
    Pick n structurally diverse chunks from one document's chunk list.
    Diversity is across: position (early/middle/late) and length.
    """
    sources = fetch_all_sources()
    result = {}
    for source in sources:
        source_id = str(source['source_id'])
        doc_chunks = fetch_chunks_by_source(source_id)
        if not doc_chunks:
            result[source_id] = []
            continue
        if len(doc_chunks) <= num_chunks_per_source:
            result[source_id] = doc_chunks
            continue

        sorted_by_pos = sorted(doc_chunks, key=lambda x: x['chunk_index'])
        total = len(sorted_by_pos)

        # Step 1: positional picks — use a dict keyed by chunk_id to avoid duplicates
        picks = {}
        indices = [0, total // 4, total // 2, (3 * total) // 4, total - 1]
        for idx in indices:
            chunk = sorted_by_pos[idx]
            picks[chunk['chunk_id']] = chunk
            if len(picks) == num_chunks_per_source:
                continue  # already have enough picks

        # Step 2: swap one pick for the longest chunk if not already included
        longest = max(doc_chunks, key=lambda c: len(c['text'].split()))
        if longest['chunk_id'] not in picks and len(picks) == num_chunks_per_source:
            # replace middle pick to preserve positional spread
            middle = sorted_by_pos[total // 2]
            picks.pop(middle['chunk_id'], None)
            picks[longest['chunk_id']] = longest
        result[source_id] = list(picks.values())[:num_chunks_per_source]

    return result

def add_eval_dataset(eval_data: list[dict]):
    inserted = insert_eval_dataset(eval_data)
    return inserted

def get_eval_dataset(limit=10):
    return fetch_eval_dataset(limit)
