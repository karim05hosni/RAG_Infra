import random
from app.repositories.eval_set_repository import insert_eval_dataset, fetch_eval_dataset


def pick_chunks_sample(doc_chunks, n=5):
    """
    Pick n structurally diverse chunks from one document's chunk list.
    Diversity is across: position (early/middle/late) and length.
    """
    if not doc_chunks:
        return []
    if len(doc_chunks) <= n:
        return doc_chunks

    sorted_by_pos = sorted(doc_chunks, key=lambda x: x['chunk_index'])
    total = len(sorted_by_pos)

    # Step 1: positional picks — use a dict keyed by chunk_id to avoid duplicates
    picks = {}
    indices = [0, total // 4, total // 2, (3 * total) // 4, total - 1]
    for idx in indices:
        chunk = sorted_by_pos[idx]
        picks[chunk['chunk_id']] = chunk
        if len(picks) == n:
            break
        # If we haven't filled n picks yet, add random chunks to the list
        if idx == len(indices) - 1 and len(picks) < n:
            remaining_chunks = [c for c in sorted_by_pos if c['chunk_id'] not in picks]
            random.shuffle(remaining_chunks)
            for chunk in remaining_chunks:
                picks[chunk['chunk_id']] = chunk
                if len(picks) == n:
                    break

    # Step 2: swap one pick for the longest chunk if not already included
    longest = max(doc_chunks, key=lambda c: len(c['text'].split()))
    if longest['chunk_id'] not in picks and len(picks) == n:
        # replace middle pick to preserve positional spread
        middle = sorted_by_pos[total // 2]
        picks.pop(middle['chunk_id'], None)
        picks[longest['chunk_id']] = longest

    return list(picks.values())[:n]

def add_eval_dataset(eval_data: list[dict]):
    inserted = insert_eval_dataset(eval_data)
    return inserted

def get_eval_dataset(limit=10):
    return fetch_eval_dataset(limit)
