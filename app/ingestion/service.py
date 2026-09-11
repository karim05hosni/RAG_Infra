from app.repositories.chunk_repository import get_source_meta, insert_sources_chunks_trans

from .readers import read_pdf, read_docx, read_txt, transcribe_video, validate_video_file
import os
from app.shared import generate_doc_id, generate_chunk_id, split_list_by_max_bytes, embedding_model
from app.repositories import add_to_qdrant
from .chunker import chunk_text, chunk_video_transcription
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


def ingest_files(file_paths, language, chunk_size=10):
    all_chunks_meta, all_chunks_text, sources_meta = [], [], []
    for file_path in file_paths:
        # read file "if video, transcribe & chunk"
        # read document
        text = read_document(file_path)
        file_name = os.path.basename(file_path)
        doc_id = generate_doc_id(file_name)
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
            chunks = chunk_text(text, chunk_size)
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

    all_chunks_text = [prepare_text_context(file_name, chunk['text']) for chunk in all_chunks_meta]

    # stop if no chunks to ingest
    if not all_chunks_meta:
        print("No chunks to ingest. Exiting.")
        return
    # embed
    source_vectors = embedding_model.encode(all_chunks_text, normalize_embeddings=True, batch_size=64, show_progress_bar=True)

    chunk_objects = [
        {**meta, "vector": vec.tolist()} for meta, vec in zip(all_chunks_meta, source_vectors)
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
