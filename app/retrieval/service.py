from datetime import datetime

import langid
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.clients.ollama import embed_text
from app.repositories.chunk_repository import fetch_chunk_by_id, fetch_chunks_by_ids, fetch_chunks_by_source, fetch_chunks_in_index_range_by_source
from app.repositories import text_search_ts_rank, qdrant_search
from app.shared import transformer
import re
# Lock down the classifier to only test for English and French
langid.set_languages(['en', 'fr'])

def detect_search_language(prompt):
    # Clean the prompt: Remove numbers, punctuation, and extra whitespace
    cleaned_prompt = re.sub(r'[^a-zA-Z\s]', '', prompt).strip()
    
    # Handle empty strings or single-character edges
    if not cleaned_prompt or len(cleaned_prompt) < 2:
        return 'en' # Default fallback
        
    lang, score = langid.classify(cleaned_prompt)
    return lang

def keywordSearch(query, language, top_k=40):
    keyword_search_results: list[dict] = []
    ts_rank = text_search_ts_rank(query, language, top_k)
    keyword_search_results = ts_rank
    return keyword_search_results

def semantic_search(query, top_k=40):
    query_embedding = embed_text([query], model='bge-m3', runtime='sentence-transformers')[0]
    qdrant_results = qdrant_search(query_embedding, top_k).points
    print(f"Retrieved {len(qdrant_results)} results from Qdrant for query: {query}")
    chunks_scores = []
    for result in qdrant_results:
        chunk_id = result.id
        score = result.score
        chunks_scores.append({
            'chunk_id': chunk_id,
            'score': score
        })
    return chunks_scores

def RRF(keyword_search_scores: list[dict], keyword_search_scores_translated: list[dict], semantic_search_scores: list[dict], k=60, w_kw=1.0, w_sem=1.0, w_translated_kw=1.0, top_k=25):
    fused_scores = {}
    counter = 0
    for doc in keyword_search_scores:
        chunk_id = str(doc['chunk_id'])
        score = doc['score']
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + w_kw / (k+ counter + 1)
        counter += 1
    counter = 0
    for doc in keyword_search_scores_translated:
        chunk_id = str(doc['chunk_id'])
        score = doc['score']
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + w_translated_kw / (k+ counter + 1)
        counter += 1
    counter = 0
    for doc in semantic_search_scores:
        chunk_id = str(doc['chunk_id'])
        score = doc['score']
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + w_sem / (k + counter + 1)
        counter += 1
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

def hybrid_search(query: str, top_k: int = 12):
    language = detect_search_language(query)
    if language not in ['en', 'fr']:
        raise ValueError(f"Unsupported language detected: {language}. Only English and French are supported.")

    semantic_search_starts = datetime.now()
    semantic_search_scores = semantic_search(query)
    semantic_search_ends = datetime.now()
    
    target_lang = 'english' if language != 'en' else 'french'
    source_lang = 'english' if language == 'en' else 'french'
    translation_starts = datetime.now()
    translated_query = MT_translate(query, source_lang, target_lang=target_lang)
    translation_ends = datetime.now()
    
    print(f"Semantic search time: {semantic_search_ends - semantic_search_starts}")
    print(f"Translation time: {translation_ends - translation_starts}")
    
    keyword_search_starts = datetime.now()
    keyword_search_scores = keywordSearch(query, source_lang)
    print(f"retrieved {len(keyword_search_scores)} keyword search results for query: {query} in language: {source_lang}")
    keyword_search_ends = datetime.now()
    
    print(f"Keyword search time: {keyword_search_ends - keyword_search_starts}")
    keyword_search_scores_translated = keywordSearch(translated_query, target_lang)
    print(f"retrieved {len(keyword_search_scores_translated)} keyword search results for translated query: {translated_query} in language: {target_lang}")
    RRF_scores = RRF(keyword_search_scores, keyword_search_scores_translated, semantic_search_scores)
    
    # get chunks texts from postgres
    
    chunk_ids = [chunk_id for chunk_id, score in RRF_scores]
    print(f"retrieved {len(chunk_ids)} chunk ids from RRF fusion")
    chunks = fetch_chunks_by_ids(chunk_ids)
    print(f"retrieved {len(chunks)} chunks from postgres")
    chunk_lookup = {}
    for chunk in chunks:
        chunk_lookup[str(chunk['chunk_id'])] = {
            'text': chunk['text'],
            'chunk_index': chunk['chunk_index'],
            'source_id': chunk['source_id']
        }
    print(f"created chunk lookup for {len(chunk_lookup)} chunks")
    final_results = [
        {"chunk_id": str(chunk_id), "score": score, "chunk_data": chunk_lookup.get(str(chunk_id))}
        for chunk_id, score in RRF_scores
    ]
    return {
        'hybrid_search_results': final_results[:top_k]
    }

def get_chunks_in_index_range_by_source(source_id, start_index, end_index):
    chunks = fetch_chunks_in_index_range_by_source(source_id, start_index, end_index)
    print(f"Retrieved {len(chunks)} chunks from source_id: {source_id} in index range: {start_index} to {end_index}")
    print(f"Chunks: {chunks}")
    return chunks

def get_neighboring_chunks(chunk_id, num_neighbors=2):
    # get the chunk from the database
    chunk = fetch_chunk_by_id(chunk_id)
    if not chunk:
        raise ValueError(f"Chunk with ID {chunk_id} not found.")

    # get the neighboring chunks
    neighboring_chunks = []
    neighboring_chunks.extend(fetch_chunks_in_index_range_by_source(str(chunk['source_id']), chunk['chunk_index'] - num_neighbors, chunk['chunk_index'] + num_neighbors))

    return neighboring_chunks

def MT_translate(text, source_lang='english', target_lang='french'):
    model = None
    tokenizer = None
    if source_lang == 'english' and target_lang == 'french':
        model = transformer.en_to_fr_translation_model
        tokenizer = transformer.en_to_fr_tokenizer
    elif source_lang == 'french' and target_lang == 'english':
        model = transformer.fr_to_en_translation_model
        tokenizer = transformer.fr_to_en_tokenizer
    else:
        raise ValueError(f"Unsupported language pair: {source_lang} to {target_lang}")
    inputs = tokenizer(text, return_tensors="pt", padding=True)

    # 4. Generate the translation tokens
    translated_tokens = model.generate(**inputs)

    # 5. Decode the tokens back into human-readable text
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text
