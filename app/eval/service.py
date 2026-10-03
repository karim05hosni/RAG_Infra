import json
from typing import List
from uuid import uuid4

from pydantic_core import ValidationError

from app.repositories.chunk_repository import fetch_chunks_by_ids, fetch_chunks_by_source, fetch_all_sources
from app.repositories.eval_dataset_wal import clean_wal, read_from_wal, write_to_wal
from .golden_dataset import pick_chunks_sample, pick_chunks_sample_by_source
from .QA_agent import ChunkResult , generate_chunk_QA, ChunkQAResponse
from app.repositories.eval_set_repository import insert_eval_dataset, fetch_eval_dataset
from app.repositories.eval_chunk_status import fetch_chunks_data_by_status, is_chunk_already_generated, mark_chunk_status, is_chunk_already_done,fetch_chunks_status

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
# Runs one batch through generate -> parse -> map -> persist. No-op on empty batch.
def process_batch(job_id: str, chunk_batch: dict) -> None:
    if not chunk_batch:
        print(f"Batch for job {job_id} is empty, skipping.")
        return
    raw_response = generate_and_stage(job_id, chunk_batch)
    if raw_response is None:
        return
    chunk_ids = list(chunk_batch.keys())
    # parsing the raw_response to validate
    parsed_response = None
    try:
        parsed_response = ChunkQAResponse.model_validate_json(raw_response)
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"Error parsing QA response for job {job_id}, chunks {chunk_ids}: {e}")
        for chunk_id in chunk_ids:
            mark_chunk_status(job_id, chunk_id, "failed", reason=f"QA parsing failed: {e}")
    rows = build_eval_rows(job_id, chunk_batch, parsed_response)
    if rows:
        insert_eval_dataset(rows)  # marks persisted + cleans WAL per chunk

def generate_eval_dataset(jobId: str | None = None, num_samples_per_source: int = 5, batch_size: int = 10) -> dict:
    if jobId is None:
        jobId = str(uuid4())
        print(f"Generated new jobId: {jobId}")
    else:
        print(f"Using provided jobId: {jobId}")
    
    register_eval_job(jobId, num_samples_per_source)

    chunk_batch: dict[str, dict] = {}
    for chunk in fetch_chunks_data_by_status(jobId, status_filter=["pending", "failed"]):
        chunk_batch[str(chunk["chunk_id"])] = chunk
        if len(chunk_batch) >= batch_size:
            process_batch(jobId, chunk_batch)
            chunk_batch = {}
    process_batch(jobId, chunk_batch)  # flush remainder, same path as full batches
    retry_from_wal(jobId)  # recover anything staged but not persisted

    inserted_chunks = len(fetch_chunks_status(jobId, status_filter=["persisted"]))
    skipped_chunks = len(fetch_chunks_status(jobId, status_filter=["skipped"]))
    return {
        "job_id": jobId,
        "message": f"inserted {inserted_chunks} chunks into eval_dataset, skipped {skipped_chunks} chunks",
        "total_chunks": len(fetch_chunks_status(jobId)),
    }