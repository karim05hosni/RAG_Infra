from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.repositories.chunk_repository import fetch_chunks_by_ids
from app.repositories import text_search_ts_rank, qdrant_search
from app.shared import transformer



def keywordSearch(query, language, use_bm25=False, use_ts_postgres=True):
    keyword_search_results = []
    ts_rank = text_search_ts_rank(query, language, 70)
    keyword_search_results = ts_rank
    return keyword_search_results

def semantic_search(query_embedding, use_qdrant_sem_search=True):
    qdrant_results = qdrant_search(query_embedding, top_k=70).points
    dot_product_scores: list[tuple[str, float]] = sorted([(result.payload['chunk_id'], result.score) for result in qdrant_results], key=lambda x: x[1], reverse=True)
    return dot_product_scores




def RRF(keyword_search_scores, keyword_search_scores_translated, semantic_search_scores, k=60, w_bm25=1.0, w_sem=1.0):
    fused_scores = {}
    for i, (chunk_id, score) in enumerate(keyword_search_scores):
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + w_bm25 / (k + i + 1)
    for i, (chunk_id, score) in enumerate(keyword_search_scores_translated):
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + w_bm25 / (k + i + 1)
    for i, (chunk_id, score) in enumerate(semantic_search_scores):
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + w_sem / (k + i + 1)
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:40]

def hybrid_search(query: str, language: str, use_qdrant_sem_search=True, use_bm25=False, use_ts_postgres=True):
    query_embedding = transformer.embedding_model.encode(query, normalize_embeddings=True)

    semantic_search_scores = semantic_search(query_embedding)
    target_lang = 'english' if language != 'english' else 'french'
    translated_query = MT_translate(query, source_lang=language, target_lang=target_lang)
    keyword_search_scores = keywordSearch(query, language)
    keyword_search_scores_translated = keywordSearch(translated_query, target_lang)
    RRF_scores = RRF(keyword_search_scores, keyword_search_scores_translated, semantic_search_scores)

    # get chunks texts from postgres
    chunk_ids = [chunk_id for chunk_id, score in RRF_scores]
    chunks = fetch_chunks_by_ids(chunk_ids)
    chunk_lookup = {c['chunk_id']: c['text'] for c in chunks}
    final_results = [
        {"chunk_id": chunk_id, "score": score, "text": chunk_lookup.get(chunk_id)}
        for chunk_id, score in RRF_scores
    ]
    return {
        'semantic_search_scores': semantic_search_scores,
        'keyword_search_scores': keyword_search_scores,
        'keyword_search_scores_translated': keyword_search_scores_translated,
        'RRF_scores': RRF_scores,
        'final_results': final_results
    }

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
