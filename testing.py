import os
import shutil
import PyPDF2
import json
import docx
import whisper
import uuid
from moviepy import VideoFileClip
import subprocess

def convert_video_to_mp3(input_file: str, output_file: str):
    # Construct the FFmpeg command as a list of strings
    cmd = [
        "ffmpeg",
        "-i", input_file,
        "-vn",
        "-acodec", "libmp3lame",
        "-ab", "192k",
        "-y",  # Overwrite output file without asking
        output_file
    ]
    
    try:
        # Run the command inside the uv environment
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Successfully converted {input_file} to {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e.stderr.decode()}")



def transcribe_video(file_path):
    try:
        file_id = str(uuid.uuid4())
        temp_audio_path = f"temp/audio/{file_id}.mp3"
        # create the temp/audio directory if it doesn't exist
        os.makedirs(os.path.dirname(temp_audio_path), exist_ok=True)
        print(f"Converting video to audio: {file_path} -> {temp_audio_path}")
        convert_video_to_mp3(file_path, temp_audio_path)
        
        print("Loading Whisper model...")
        model = whisper.load_model("base", device="cpu")  # Load the model on CPU
        
        print("Transcribing audio...")
        transcription = model.transcribe(temp_audio_path, fp16=False )  # Set fp16 to False for CPU inference
        segments = transcription.get("segments", [])
        result = []
        for segment in segments:
            result.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })
        return result
    except Exception as e:
        raise e
    finally:
        os.remove(temp_audio_path)


result = transcribe_video('D:\\Projects\\RAG_infra\\1. Peut-on perdre la vision sans s’en rendre compte_1.mp4')
print(f"Transcription result: {result}")


from app.ingestion.readers import read_pdf, read_docx, read_txt, transcribe_video, validate_video_file
import os
from app.shared import generate_doc_id, generate_chunk_id, split_list_by_max_bytes
from app.repositories import add_to_qdrant, insert_documents_chunks_trans, fetch_chunks_by_ids, get_document_meta
from app.shared import embedding_model
from app.ingestion.chunker import chunk_text, chunk_video_transcription
def read_document(file_path):
    file_name, ext = os.path.splitext(file_path)
    file_ext = ext.lower()
    print(f"Reading corpus: {file_path} with extension: {file_ext}")
    if file_ext == ".pdf":
        return read_pdf(file_path)
    if file_ext == ".docx":
        return read_docx(file_path)
    if file_ext == ".txt":
        return read_txt(file_path)
    return None

def read_video(file_path):
    return transcribe_video(file_path)



def ingest_files(file_paths, chunk_size=10):
    all_chunks, all_meta = [], []
    for file_path in file_paths:
        # read file "if video, transcribe & chunk"
        # read document
        text = read_document(file_path)
        file_name = os.path.basename(file_path)
        doc_id = generate_doc_id(file_name)
        # check if document already exists in Postgres
        doc_meta = get_document_meta(doc_id)
        if doc_meta:
            print(f"Document {doc_id} already exists in Postgres.")
            continue
        # storage.insert_or_get_document_meta_to_postgres(file_name, doc_id)
        print(f"Read document: {file_path} with length: {len(text)} characters.")

        # if docuement is a video, transcribe it
        if validate_video_file(file_path):
            print(f"Transcribing video: {file_path}")
            segments = read_video(file_path)
            chunked_segments = chunk_video_transcription(segments)
            for i, chunk in enumerate(chunked_segments):
                all_chunks.append(chunk["text"])
                all_meta.append({"chunk_id": generate_chunk_id(doc_id, i), "doc_id": doc_id, "text": chunk["text"]})
        else:
            chunks = chunk_text(text, chunk_size)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_meta.append({"chunk_id": generate_chunk_id(doc_id, i), "doc_id": doc_id, "text": chunk})
    # stop if no chunks to ingest
    if not all_chunks:
        print("No new chunks to ingest.")
        return
    # embed
    vectors = embedding_model.encode(all_chunks, normalize_embeddings=True, batch_size=64, show_progress_bar=True)

    chunk_objects = [
        {**meta, "vector": vec.tolist()} for meta, vec in zip(all_meta, vectors)
    ]
    # add to qdrant
    grouped_chunks = split_list_by_max_bytes(chunk_objects, max_bytes=30 * 1024 * 1024)

    try:
        add_to_qdrant(grouped_chunks)
    except Exception as e:
        print(f"Error occurred while adding chunks to Qdrant: {e}")
        raise e
    try:
        document_objects = [{"doc_id": meta["doc_id"], "filename": file_name} for meta in all_meta]
        insert_documents_chunks_trans(document_objects, chunk_objects)
    except Exception as e:
        print(f"Error occurred while inserting documents and chunks to Postgres: {e}")
        raise e
