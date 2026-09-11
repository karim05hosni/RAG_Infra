"""
Chunking

JSON format:
[
    {
        "chunk_id": "alice_doc__chunk_0",
        "doc_id": "alice_doc",
        "text": "Alice likes cats.",
        "vector": [...]
    }
]
"""
def clean_bullet_points(text: str):
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        # print(f"Cleaning bullet points for line: {line}")
        if line.startswith(("-", "*", "•")):
            line = line[1:].strip()
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

def chunk_text(text, chunk_size=10):
    text = clean_bullet_points(text)
    sentences = text.replace('\n', ' ').split('. ')
    chunks = []
    
    # sliding window
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        sentence_size = len(sentence)

        if current_size + sentence_size <= chunk_size:
            # print(f"Adding sentence: {sentence}")
            current_chunk.append(sentence)
            current_size += sentence_size
        else:
            # print(f"Starting new chunk with sentence: {sentence}")
            chunks.append('. '.join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size

    # Append any remaining text as a chunk
    if current_chunk:
        chunks.append('. '.join(current_chunk))

    return chunks
def content_hash(text):
    return hash(text)

# [{ "start": float, "end": float, "text": str }]
def chunk_video_transcription(transcription):
    """
    chunk every 30s of speech
    """
    chunk_size = 30.0
    chunks = []
    current_chunk = []
    current_start = None
    current_end = None

    for segment in transcription:
        start = segment["start_time"]
        end = segment["end_time"]
        text = segment["text"]
        chunk_index = len(chunks)

        if current_start is None:
            current_start = start
            current_end = end
            current_chunk.append(text)
        elif end - current_start <= chunk_size:
            current_end = end
            current_chunk.append(text)
        else:
            chunks.append({
                "start_time": current_start,
                "end_time": current_end,
                "text": ' '.join(current_chunk),
                "chunk_index": chunk_index
            })
            current_start = start
            current_end = end
            current_chunk = [text]

    if current_chunk:
        chunks.append({
            "start_time": current_start,
            "end_time": current_end,
            "text": ' '.join(current_chunk),
            "chunk_index": len(chunks)
        })

    return chunks