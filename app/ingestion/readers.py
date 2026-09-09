import os
import PyPDF2
import json
import docx
import whisper
import uuid
import whisper
import uuid
import subprocess

def validate_video_file(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    _, ext = os.path.splitext(file_path)
    if ext.lower() not in valid_extensions:
        raise ValueError(f"Invalid file extension: {ext}. Supported extensions are: {valid_extensions}")
    # validate size
    max_size_mb = 300  # Maximum file size in MB
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        raise ValueError(f"File size exceeds the maximum allowed size of {max_size_mb} MB.")
    
    return True
def read_pdf(file_path):
    text = ""
    with open(file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + '\n'
    return text

def read_docx(file_path):
    text = ""
    doc = docx.Document(file_path)
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

def read_txt(file_path):
    text = ""
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text

def read_corpus(file_path):
    file_name, ext = os.path.splitext(file_path)
    file_ext = ext.lower()
    print(f"Reading corpus: {file_path} with extension: {file_ext}")
    if file_ext == ".pdf":
        return read_pdf(file_path)
    if file_ext == ".docx":
        return read_docx(file_path)
    if file_ext == ".txt":
        return read_txt(file_path)
    if validate_video_file(file_path):
        return transcribe_video(file_path)
    return None

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

def transcribe_video(file_path, device="cpu", model_name="base", fp16=False):
    try:
        file_id = str(uuid.uuid4())
        temp_audio_path = f"temp/audio/{file_id}.mp3"
        # create the temp/audio directory if it doesn't exist
        os.makedirs(os.path.dirname(temp_audio_path), exist_ok=True)
        print(f"Converting video to audio: {file_path} -> {temp_audio_path}")
        convert_video_to_mp3(file_path, temp_audio_path)
        
        print("Loading Whisper model...")
        model = whisper.load_model(model_name, device=device)  # Load the model on the specified device
        
        print("Transcribing audio...")
        transcription = model.transcribe(temp_audio_path, fp16=fp16 )  # Set fp16 to False for CPU inference
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
        print(f"Error during transcription: ")
        raise e
    finally:
        os.remove(temp_audio_path)
