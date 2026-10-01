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

def generate_eval_dataset(jobId=None, num_samples_per_source=5):
    if (not jobId):
        jobId = str(uuid4())
        print(f"Generated new jobId: {jobId}")
    else:
        print(f"Using provided jobId: {jobId}")
    register_eval_job(jobId, num_samples_per_source)
    chunk_batch: dict[str, dict] =  {}
    for chunk in fetch_chunks_data_by_status(jobId, status_filter=["pending", "failed"]):
        chunk_id: str = str(chunk['chunk_id'])
        source_id = chunk['source_id']
        chunk_batch[chunk_id] = chunk
        if len(chunk_batch) < 8:
            continue
        LLM_response = "{}"
        try:
            print(f"Generating QA from source {source_id}")
            LLM_response = generate_chunk_QA(list(chunk_batch.values()))
            print(f"Writing to WAL for job {jobId} with chunks {list(chunk_batch.keys())}")
            write_to_wal(jobId, list(chunk_batch.keys()), LLM_response)
            for chunk_id in chunk_batch.keys():
                mark_chunk_status(jobId, chunk_id, "generated")
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"Error parsing QA generation response for source {source_id}: {e}")
            for chunk_id in chunk_batch.keys():
                mark_chunk_status(jobId, chunk_id, "failed", reason=f"QA generation failed: {e}")
            chunk_batch.clear()
            continue
        
        try:
            parsed_response = ChunkQAResponse.model_validate_json(LLM_response)
            print(f"generated batch QA for source {source_id}: {parsed_response}...")
            # chunk_id -> response mapping
            chunk_response_map = {}
            for chunk_result in parsed_response.chunks:
                chunk_id = str(chunk_result.chunk_id)
                chunk_response_map[chunk_id] = chunk_result

            eval_dataset = []

            for chunk_id, chunk_data in chunk_batch.items():
                chunk_data['evaluation'] = chunk_response_map[chunk_id]
                print(f"chunk_data after evaluation: {chunk_data}")
                # check if the chunk was marked as skipped
                if chunk_data['evaluation'].skip:
                    # clean from WAL log for the jobId and chunk_id
                    clean_wal(jobId, chunk_id)
                    mark_chunk_status(jobId, chunk_id, "skipped", reason=chunk_data['evaluation'].reason)
                    continue

                for qa_pair in chunk_data["evaluation"].qa_pairs:
                    eval_dataset.append({
                        "job_id": jobId,
                        "chunk_id": chunk_id,
                        "source_id": chunk_data['source_id'],
                        "source": 'LLM-model',
                        "question": qa_pair.question,
                        "answer": qa_pair.answer,
                        "language": qa_pair.language,
                        "cross_lingual": qa_pair.cross_lingual,
                        "chunk_text": chunk_data['text'],
                        "expected_chunks_ids": qa_pair.expected_chunks_ids
                    })
        except Exception as e:
            print(f"Error processing generated QA for source {source_id}: {e}")
            for chunk_id in chunk_batch.keys():
                mark_chunk_status(jobId, chunk_id, "failed", reason=f"QA processing failed: {e}")
            continue
        finally:
            chunk_batch.clear()  # clear the batch after processing

        insert_eval_dataset(eval_dataset) # already marks the chunk as persisted in eval_chunk_status, cleans WAL log for the jobId and chunk_id

    # TODO: process the remainder
    for chunk_id, chunk_data in chunk_batch.items():
        try:
            print(f"Generating QA for remaining chunk {chunk_id} from source {chunk_data['source_id']}")
            LLM_response = generate_chunk_QA([chunk_data])
            print(f"Writing to WAL for job {jobId} with chunk {chunk_id}")
            write_to_wal(jobId, [chunk_id], LLM_response)
            mark_chunk_status(jobId, chunk_id, "generated")
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"Error parsing QA generation response for chunk {chunk_id}: {e}")
            mark_chunk_status(jobId, chunk_id, "failed", reason=f"QA generation failed: {e}")
            continue
        
        try:
            parsed_response = ChunkQAResponse.model_validate_json(LLM_response)
            print(f"generated QA for chunk {chunk_id}: {parsed_response}...")
            # check if the chunk was marked as skipped
            if parsed_response.chunks[0].skip:
                # clean from WAL log for the jobId and chunk_id
                clean_wal(jobId, chunk_id)
                mark_chunk_status(jobId, chunk_id, "skipped", reason=parsed_response.chunks[0].reason)
                continue

            eval_dataset = []
            for qa_pair in parsed_response.chunks[0].qa_pairs:
                eval_dataset.append({
                    "job_id": jobId,
                    "chunk_id": chunk_id,
                    "source_id": chunk_data['source_id'],
                    "source": 'LLM-model',
                    "question": qa_pair.question,
                    "answer": qa_pair.answer,
                    "language": qa_pair.language,
                    "cross_lingual": qa_pair.cross_lingual,
                    "chunk_text": chunk_data['text'],
                    "expected_chunks_ids": qa_pair.expected_chunks_ids
                })
        except Exception as e:
            print(f"Error processing generated QA for chunk {chunk_id}: {e}")
            mark_chunk_status(jobId, chunk_id, "failed", reason=f"QA processing failed: {e}")
            continue

        insert_eval_dataset(eval_dataset) # already marks the chunk as persisted in eval_chunk_status

    # retry any failed chunks from WAL log
    not_persisted_chunks = read_from_wal(jobId)
    for not_persisted_chunk in not_persisted_chunks:
        print(f"Retrying to persist chunks for job {jobId} from WAL log: {not_persisted_chunk}")
        # read from WAL log, parse the raw response, and try to persist the chunks again
        parsed_response = ChunkQAResponse.model_validate_json(not_persisted_chunk['raw_response'])
        print(f"Retrying to insert into eval_dataset for job {jobId}: {parsed_response}...")
        # chunk_id -> response mapping
        chunk_response_map = {}
        for chunk_result in parsed_response.chunks:
            chunk_id = str(chunk_result.chunk_id)
            chunk_response_map[chunk_id] = chunk_result
        eval_dataset = []
        # retrie chunk text and source_id
        chunks_data = fetch_chunks_by_ids(list(chunk_response_map.keys()))
        for chunk_data in chunks_data:
            chunk_id = str(chunk_data['chunk_id'])
            chunk_data['evaluation'] = chunk_response_map[chunk_id]
            print(f"chunk_data after evaluation: {chunk_data}")
            # check if the chunk was marked as skipped
            if chunk_data['evaluation'].skip:
                # clean from WAL log for the jobId and chunk_id
                clean_wal(jobId, chunk_id)
                mark_chunk_status(jobId, chunk_id, "skipped", reason=chunk_data['evaluation'].reason)
                continue

            for qa_pair in chunk_data["evaluation"].qa_pairs:
                eval_dataset.append({
                    "job_id": jobId,
                    "chunk_id": chunk_id,
                    "source_id": chunk_data['source_id'],
                    "source": 'LLM-model',
                    "question": qa_pair.question,
                    "answer": qa_pair.answer,
                    "language": qa_pair.language,
                    "cross_lingual": qa_pair.cross_lingual,
                    "chunk_text": chunk_data['text'],
                    "expected_chunks_ids": qa_pair.expected_chunks_ids
                })
        insert_eval_dataset(eval_dataset) # already marks the chunk as persisted in eval_chunk_status
    inserted_chunks = len(fetch_chunks_status(jobId, status_filter=["persisted"]))
    skipped_chunks = len(fetch_chunks_status(jobId, status_filter=["skipped"]))
    print(f"Inserted {inserted_chunks} chunks into eval_dataset, skipped {skipped_chunks} chunks for job {jobId}")
    return {
        "job_id": jobId,
        "message": f"inserted {inserted_chunks} chunks into eval_dataset, skipped {skipped_chunks} chunks",
        "total_chunks": len(fetch_chunks_status(jobId)),
    }
