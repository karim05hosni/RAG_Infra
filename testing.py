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