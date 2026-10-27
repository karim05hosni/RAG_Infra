import re
import time

from app.clients.ollama import embed_text
from app.repositories.chunk_repository import get_source_meta, insert_sources_chunks_trans, fetch_all_chunks
from app.repositories.vector_repository import create_collection, delete_all_collections
from app.shared.helpers import sanitize_fr_text

from .readers import read_pdf, read_docx, read_txt, transcribe_video, validate_video_file
import os
from app.shared import generate_doc_id, generate_chunk_id, split_list_by_max_bytes
from app.repositories import add_to_qdrant
from .chunker import chunk_text, chunk_video_transcription
from sentence_transformers import SentenceTransformer
from langdetect import detect, detect_langs

from app.ingestion import chunker

def read_document(file_path):
    file_name, ext = os.path.splitext(file_path)
    file_ext = ext.lower()
    if file_ext == ".pdf":
        return read_pdf(file_path)
    if file_ext == ".docx":
        return read_docx(file_path)
    if file_ext == ".txt":
        return read_txt(file_path)
    return None

def read_video(file_path):
    return transcribe_video(file_path)

def prepare_text_context(source_title: str, text: str):
    return f"From {source_title}: {text}"


def ingest_files(file_paths, chunk_size=10):
    all_chunks_meta, all_chunks_text, sources_meta, sources_id_names = [], [], [], {}
    language = None
    for file_path in file_paths:
        # read file "if video, transcribe & chunk"
        # read document
        text = read_document(file_path)
        file_name = os.path.basename(file_path)
        doc_id = str(generate_doc_id(file_name))
        sources_id_names[doc_id] = file_name
        # check if document already exists in Postgres
        source_meta = get_source_meta(doc_id)
        if source_meta:
            print(f"Document {doc_id} already exists in Postgres.")
            continue
        # if docuement is a video, transcribe it
        if validate_video_file(file_path):
            sources_meta.append({
                'source_id': doc_id,
                'source_type': 'video',
                'filename': file_name,
                'file_path': file_path
            })
            print(f"Transcribing video: {file_path}")
            segments = read_video(file_path)
            chunked_segments = chunk_video_transcription(segments)
            for i, chunk in enumerate(chunked_segments):
                metadata = {
                    'start_time': chunk['start_time'],
                    'end_time': chunk['end_time'],
                }
                # detect chunk language
                if detect(chunk['text']) == 'fr':
                    language = 'french'
                elif detect(chunk['text']) == 'en':
                    language = 'english'
                else:
                    raise ValueError(f"Unsupported language detected in chunk {i} of video {file_name}: {detect(chunk['text'])}")
                print(f"Detected language for chunk {i} of video {file_name}: {language}")
                chunk_data = {
                    'chunk_id': generate_chunk_id(doc_id, i),
                    'source_id': doc_id,
                    'text': prepare_text_context(file_name, chunk['text']),
                    'chunk_index': i,
                    'metadata': metadata,
                    'language': language
                }
                all_chunks_meta.append(chunk_data)
        else:
            sources_meta.append({
                'source_id': doc_id,
                'source_type': 'document',
                'filename': file_name,
                'file_path': file_path
            })
            # detect document language
            if detect(text) == 'fr':
                language = 'french'
            elif detect(text) == 'en':
                language = 'english'
            else:
                raise ValueError(f"Unsupported language detected in document {file_name}: {detect(text)}")
            print(f"Detected language for document {file_name}: {language}")
            chunks = chunker.chunk_text(text, chunk_size)
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    'chunk_id': generate_chunk_id(doc_id, i),
                    'source_id': doc_id,
                    'text': prepare_text_context(file_name, chunk),
                    'chunk_index': i,
                    'metadata': {},
                    'language': language
                }
                all_chunks_meta.append(chunk_data)

    # all_chunks_text = [prepare_text_context(file_name, chunk['text']) for chunk in all_chunks_meta]
    all_chunks_text = []
    for chunk in all_chunks_meta:
        chunk_text = chunk['text']
        chunk_with_context = prepare_text_context(sources_id_names[chunk['source_id']], chunk_text)
        # sanitize text if language is French
        if chunk['language'] == 'french':
            print(f"Sanitizing French text for chunk {chunk['chunk_id']}")
            chunk_with_context = sanitize_fr_text(chunk_with_context)
        all_chunks_text.append(chunk_with_context)
    # stop if no chunks to ingest
    if not all_chunks_meta:
        print("No chunks to ingest. Exiting.")
        return
    # embed in batches
    # source_vectors = embedding_model.encode(all_chunks_text, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    batch_size = len(all_chunks_text) // 10 if len(all_chunks_text) > 10 else len(all_chunks_text)
    source_vectors = []
    for i in range(0, len(all_chunks_text), batch_size):
        batch_chunks = all_chunks_text[i:i + batch_size]
        print(f"Embedding batch {i // batch_size + 1} of {len(all_chunks_text) // batch_size + 1}...")
        batch_vectors = embed_text(batch_chunks)
        source_vectors.extend(batch_vectors)
        time.sleep(1)  # sleep for 1 second to avoid rate limiting
    chunk_objects = [
        {**meta, "vector": vec} for meta, vec in zip(all_chunks_meta, source_vectors)
    ]
    # add to qdrant
    grouped_source_chunks = split_list_by_max_bytes(chunk_objects, max_bytes=30 * 1024 * 1024)

    try:
        add_to_qdrant(grouped_source_chunks)
    except Exception as e:
        print(f"Error occurred while adding chunks to Qdrant: {e}")
        raise e
    try:
        insert_sources_chunks_trans(sources_meta, all_chunks_meta)
    except Exception as e:
        print(f"Error occurred while inserting documents and chunks to Postgres: {e}")
        raise e
    

import json

    # reembedd existing chunks with new embedding model (DELETES EXISTING VECTORS IN QDRANT AND RE-ADDS THEM)
def reembedd_chunks(embedding_model):
    all_vectors = []
    all_chunks = fetch_all_chunks()
    # check if temp_vectors.json exists, if so, load it
    if os.path.exists("temp_vectors.json"):
        with open("temp_vectors.json", "r") as f:
            all_vectors = json.load(f)
        print(f"Loaded {len(all_vectors)} vectors from temp_vectors.json")
    else:
        # fetch all chunks from Postgres
        # all_chunks = fetch_all_chunks()
        all_chunks_text = [chunk['text'] for chunk in all_chunks]
        print(f"Re-embedding {len(all_chunks)} chunks with new embedding model...")
        print(f"chunks text sample: {all_chunks_text[:2]}")
        try:
            # process embedding in batches to avoid memory issues
            batch_size = len(all_chunks_text) // 10 if len(all_chunks_text) > 10 else len(all_chunks_text)
            for i in range(0, len(all_chunks_text), batch_size):
                batch_chunks = all_chunks_text[i:i + batch_size]
                print(f"Embedding batch {i // batch_size + 1} of {len(all_chunks_text) // batch_size + 1}...")
                batch_vectors = embed_text(batch_chunks)
                all_vectors.extend(batch_vectors) 
                # time.sleep(1)  # sleep for 1 second to avoid rate limiting
            
            print(f"Successfully embedded {len(all_vectors)} chunks., type {type(all_vectors)}")
            print(f"type of first vector: {type(all_vectors[0])}, length: {len(all_vectors[0])}")
            # save in temp json file
            with open("temp_vectors.json", "w") as f:
                json.dump(all_vectors, f)
        except Exception as e:
            print(f"Error occurred while embedding chunks: {e}")
            raise e
    # all_vectors = embedding_model.encode(all_chunks_text, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    # store in qdrant
    print(f"constructing chunk objects with vectors...")
    all_chunks_meta = []
    for chunk in fetch_all_chunks():
        all_chunks_meta.append({
            'chunk_id': str(chunk['chunk_id']),
            'source_id': chunk['source_id'],
            'text': chunk['text'],
            'chunk_index': chunk['chunk_index'],
            'metadata': chunk['metadata'],
            'language': chunk['language']
        })
    chunk_objects = [
        {**chunk, "vector": vec} for chunk, vec in zip(all_chunks_meta, all_vectors)
    ]
    grouped_chunks = split_list_by_max_bytes(chunk_objects, max_bytes=30 * 1024 * 1024)
    try:
        # delete
        print(f"Deleting existing collections in Qdrant...")
        delete_all_collections()
        print(f"Creating new collection in Qdrant...")
        create_collection("my_rag", vector_size=len(all_vectors[0]))
        print(f"Re-adding chunks to Qdrant...")
        add_to_qdrant(grouped_chunks)
    except Exception as e:
        print(f"Error occurred while re-adding chunks to Qdrant: {e}")
        raise e
    return {"message": f"Successfully re-embedded {len(all_chunks)} chunks."}
