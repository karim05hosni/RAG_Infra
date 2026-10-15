import random
from app.repositories.chunk_repository import fetch_chunks_by_ids, fetch_chunks_by_source, fetch_all_sources
from app.repositories.eval_dataset_wal import clean_wal, read_from_wal, write_to_wal
from app.repositories.chunk_repository import fetch_all_sources, fetch_chunks_by_source
from app.repositories.eval_set_repository import insert_eval_dataset, fetch_eval_dataset
from .QA_agent import ChunkResult , generate_chunk_QA, ChunkQAResponse
from app.repositories.eval_set_repository import insert_eval_dataset, fetch_eval_dataset
from app.repositories.eval_chunk_status import fetch_chunks_data_by_status, is_chunk_already_generated, mark_chunk_status, is_chunk_already_done,fetch_chunks_status
from pydantic_core import ValidationError
import json

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

def pick_chunks_sample_by_source(source_id, num_chunks_per_source=5):
    """
    Pick n structurally diverse chunks from one document's chunk list.
    Diversity is across: position (early/middle/late) and length.
    """
    doc_chunks = fetch_chunks_by_source(source_id)
    if not doc_chunks:
        return []
    if len(doc_chunks) <= num_chunks_per_source:
        return doc_chunks

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

    return list(picks.values())[:num_chunks_per_source]

def add_eval_dataset(eval_data: list[dict]):
    inserted = insert_eval_dataset(eval_data)
    return inserted

def get_eval_dataset(limit=10):
    return fetch_eval_dataset(limit)

def register_eval_job(jobId, num_samples_per_source=5):
    resources = fetch_all_sources()
    for resource in resources:
        source_id = resource['source_id']
        chunk_sample = pick_chunks_sample_by_source(source_id, num_samples_per_source)
        for chunk in chunk_sample:
            # check if the chunk is already done
            if is_chunk_already_done(jobId, chunk['chunk_id']):
                print(f"Chunk {chunk['chunk_id']} already done for job {jobId}, skipping.")
                continue
            if is_chunk_already_generated(jobId, chunk['chunk_id']):
                print(f"Chunk {chunk['chunk_id']} already generated for job {jobId}, skipping.")
                continue
            chunk_id = chunk['chunk_id']
            mark_chunk_status(jobId, chunk_id, "pending")
    return True



# map the parsed response back to the chunks and build rows for insertion into eval_dataset
def build_eval_rows(job_id: str, chunk_batch: dict, parsed_response: ChunkQAResponse) -> list[dict]:

    chunk_response_map = {}
    for chunk_result in parsed_response.chunks:
        chunk_response_map[chunk_result.chunk_id] = chunk_result
    rows = []
    for chunk_id, chunk_data in chunk_batch.items():
        evaluation = chunk_response_map.get(chunk_id)
        if evaluation is None:
            print(f"Warning: no evaluation returned for chunk {chunk_id}")
            mark_chunk_status(job_id, chunk_id, "failed", reason="Missing evaluation in LLM response")
            continue
        if evaluation.skip:
            clean_wal(job_id, chunk_id)
            mark_chunk_status(job_id, chunk_id, "skipped", reason=evaluation.reason)
            continue
        for qa in evaluation.qa_pairs:
            rows.append({
                "job_id": job_id,
                "chunk_id": chunk_id,
                "source_id": chunk_data["source_id"],
                "source": "LLM-model",
                "question": qa.question,
                "answer": qa.answer,
                "language": qa.language,
                "cross_lingual": qa.cross_lingual,
                "chunk_text": chunk_data["text"],
                "expected_chunks_ids": qa.expected_chunks_ids,
            })
    return rows

# retrying the failed chunks from WAL log
def retry_from_wal(job_id: str) -> None:
    for entry in read_from_wal(job_id):
        chunk_ids = entry["chunk_ids"]
        # parsing the raw_response to validate
        parsed = None
        try:
            parsed = ChunkQAResponse.model_validate_json(entry["raw_response"])
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"Error parsing QA response for job {job_id}, chunks {chunk_ids}: {e}")
            for chunk_id in chunk_ids:
                mark_chunk_status(job_id, chunk_id, "failed", reason=f"QA parsing failed: {e}")
        if parsed is None:
            continue
        # mapping the parsed response back to the chunks and inserting into eval_dataset
        chunks_data = fetch_chunks_by_ids(chunk_ids)
        chunk_batch = {}
        for chunk in chunks_data:
            chunk_batch[str(chunk["chunk_id"])] = chunk
        rows = build_eval_rows(job_id, chunk_batch, parsed)
        if rows:
            insert_eval_dataset(rows) # marks persisted + cleans WAL per chunk


# Generates QA for a batch of chunks and stages the result.
def generate_and_stage(job_id: str, chunk_batch: dict) -> str | None:

    chunk_ids = list(chunk_batch.keys())
    try:
        raw_response = generate_chunk_QA(list(chunk_batch.values()))
        write_to_wal(job_id, chunk_ids, raw_response)
        for chunk_id in chunk_ids:
            mark_chunk_status(job_id, chunk_id, "generated")
        return raw_response
    except Exception as e:
        print(f"Error generating QA for job {job_id}, chunks {chunk_ids}: {e}")
        for chunk_id in chunk_ids:
            mark_chunk_status(job_id, chunk_id, "failed", reason=f"QA generation failed: {e}")
        return None