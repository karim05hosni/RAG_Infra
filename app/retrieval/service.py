from datetime import datetime

import langid
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.repositories.chunk_repository import fetch_chunks_by_ids, fetch_chunks_in_index_range_by_source
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

def keywordSearch(query, language, top_k=70):
    keyword_search_results = []
    ts_rank = text_search_ts_rank(query, language, top_k)
    keyword_search_results = ts_rank
    return keyword_search_results

def semantic_search(query_embedding, top_k=70):
    qdrant_results = qdrant_search(query_embedding, top_k=70).points
    dot_product_scores: list[tuple[str, float]] = sorted([(result.payload['chunk_id'], result.score) for result in qdrant_results], key=lambda x: x[1], reverse=True)
    return dot_product_scores

def RRF(keyword_search_scores, keyword_search_scores_translated, semantic_search_scores, k=60, w_kw=1.0, w_sem=1.0, w_translated_kw=1.0, top_k=40):
    fused_scores = {}
    for i, (chunk_id, score) in enumerate(keyword_search_scores):
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + w_kw / (k + i + 1)
    for i, (chunk_id, score) in enumerate(keyword_search_scores_translated):
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + w_translated_kw / (k + i + 1)
    for i, (chunk_id, score) in enumerate(semantic_search_scores):
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + w_sem / (k + i + 1)
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

def hybrid_search(query: str):
    detect_language_starts = datetime.now()
    language = detect_search_language(query)
    detect_language_ends = datetime.now()
    print(f"Language detection time: {detect_language_ends - detect_language_starts}")
    if language not in ['en', 'fr']:
        raise ValueError(f"Unsupported language detected: {language}. Only English and French are supported.")
    query_embedding = transformer.embedding_model.encode(query, normalize_embeddings=True)

    semantic_search_starts = datetime.now()
    semantic_search_scores = semantic_search(query_embedding)
    semantic_search_ends = datetime.now()
    target_lang = 'english' if language != 'en' else 'french'
    source_lang = 'english' if language == 'en' else 'french'
    translation_starts = datetime.now()
    translated_query = MT_translate(query, source_lang, target_lang=target_lang)
    translation_ends = datetime.now()
    
    print(f"Semantic search time: {semantic_search_ends - semantic_search_starts}")
    print(f"Translation time: {translation_ends - translation_starts}")
    
    keyword_search_starts = datetime.now()
    keyword_search_scores = keywordSearch(query, language)
    keyword_search_ends = datetime.now()
    
    print(f"Keyword search time: {keyword_search_ends - keyword_search_starts}")
    keyword_search_scores_translated = keywordSearch(translated_query, target_lang)
    
    fusion_starts = datetime.now()
    RRF_scores = RRF(keyword_search_scores, keyword_search_scores_translated, semantic_search_scores)
    fusion_ends = datetime.now()
    
    print(f"fusion time: {fusion_ends - fusion_starts}")
    # get chunks texts from postgres
    
    response_formatting_starts = datetime.now()
    chunk_ids = [chunk_id for chunk_id, score in RRF_scores]
    chunks = fetch_chunks_by_ids(chunk_ids)
    chunk_lookup = {}
    for chunk in chunks:
        chunk_lookup[chunk['chunk_id']] = {
            'text': chunk['text'],
            'chunk_index': chunk['chunk_index'],
            'source_id': chunk['source_id']
        }
    final_results = [
        {"chunk_id": chunk_id, "score": score, "chunk_data": chunk_lookup.get(chunk_id)}
        for chunk_id, score in RRF_scores
    ]
    response_formatting_ends = datetime.now()
    print(f"response formatting time: {response_formatting_ends - response_formatting_starts}")
    return {
        'hybrid_search_results': final_results
    }

def get_chunks_in_index_range_by_source(source_id, start_index, end_index):
    chunks = fetch_chunks_in_index_range_by_source(source_id, start_index, end_index)
    return chunks

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
