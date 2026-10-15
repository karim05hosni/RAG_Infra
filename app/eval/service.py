import json
from typing import List
from uuid import uuid4
from pydantic_core import ValidationError
from app.eval import golden_dataset
from .QA_agent import ChunkResult , generate_chunk_QA, ChunkQAResponse
from app.repositories.eval_set_repository import insert_eval_dataset, fetch_eval_dataset
from app.repositories.eval_chunk_status import fetch_chunks_data_by_status, is_chunk_already_generated, mark_chunk_status, is_chunk_already_done,fetch_chunks_status

# Runs one batch through generate -> parse -> map -> persist. No-op on empty batch.
def process_batch(job_id: str, chunk_batch: dict) -> None:
    if not chunk_batch:
        print(f"Batch for job {job_id} is empty, skipping.")
        return
    raw_response = golden_dataset.generate_and_stage(job_id, chunk_batch)
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
    rows = golden_dataset.build_eval_rows(job_id, chunk_batch, parsed_response)
    if rows:
        insert_eval_dataset(rows)  # marks persisted + cleans WAL per chunk

def generate_eval_dataset(jobId: str | None = None, num_samples_per_source: int = 5, batch_size: int = 10) -> dict:
    if jobId is None:
        jobId = str(uuid4())
        print(f"Generated new jobId: {jobId}")
    else:
        print(f"Using provided jobId: {jobId}")
    
    golden_dataset.register_eval_job(jobId, num_samples_per_source)

    chunk_batch: dict[str, dict] = {}
    for chunk in fetch_chunks_data_by_status(jobId, status_filter=["pending", "failed"]):
        chunk_batch[str(chunk["chunk_id"])] = chunk
        if len(chunk_batch) >= batch_size:
            process_batch(jobId, chunk_batch)
            chunk_batch = {}
    process_batch(jobId, chunk_batch)  # flush remainder, same path as full batches
    golden_dataset.retry_from_wal(jobId)  # recover anything staged but not persisted

    inserted_chunks = len(fetch_chunks_status(jobId, status_filter=["persisted"]))
    skipped_chunks = len(fetch_chunks_status(jobId, status_filter=["skipped"]))
    return {
        "job_id": jobId,
        "message": f"inserted {inserted_chunks} chunks into eval_dataset, skipped {skipped_chunks} chunks",
        "total_chunks": len(fetch_chunks_status(jobId)),
    }
    
def evaluate_sample():
    pass