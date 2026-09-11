from app.repositories.chunk_repository import fetch_chunks_by_ids
from app.repositories import text_search_ts_rank, qdrant_search
from app.shared import transformer



def keywordSearch(query, language, use_bm25=False, use_ts_postgres=True):
    keyword_search_results = []
    ts_rank = text_search_ts_rank(query, language)
    keyword_search_results = ts_rank
    return keyword_search_results

def semantic_search(query_embedding, use_qdrant_sem_search=True):
    qdrant_results = qdrant_search(query_embedding, top_k=50).points
    dot_product_scores: list[tuple[str, float]] = sorted([(result.payload['chunk_id'], result.score) for result in qdrant_results], key=lambda x: x[1], reverse=True)
    return dot_product_scores


def RRF(bm25_scores, dot_product_scores, k=60, w_bm25=1.0, w_sem=1.0):
    fused_scores = {}
    for i, (chunk_id, score) in enumerate(bm25_scores):
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + w_bm25 / (k + i + 1)
    for i, (chunk_id, score) in enumerate(dot_product_scores):
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + w_sem / (k + i + 1)
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

def hybrid_search(query: str, language: str, use_qdrant_sem_search=True, use_bm25=False, use_ts_postgres=True):
    query_embedding = transformer.embedding_model.encode(query, normalize_embeddings=True)

    semantic_search_scores = semantic_search(query_embedding)
    keyword_search_scores = keywordSearch(query, language)
    RRF_scores = RRF(keyword_search_scores, semantic_search_scores)

    # get chunks texts from postgres
    chunk_ids = [chunk_id for chunk_id, score in RRF_scores]
    chunks = fetch_chunks_by_ids(chunk_ids)
    final_results = []
    for chunk_id, score in RRF_scores:
        chunk_text = next((chunk['text'] for chunk in chunks if chunk['chunk_id'] == chunk_id), None)
        final_results.append({"chunk_id": chunk_id, "score": score, "text": chunk_text})
    return {
        'semantic_search_scores': semantic_search_scores,
        'keyword_search_scores': keyword_search_scores,
        'RRF_scores': RRF_scores,
        'final_results': final_results
    }
