import math
def BM25_v3(query, chunks, inverted_index: dict, doc_lengths:dict, avg_doc_len):
    N = len(chunks)
    scores_map = {} # chunkId -> total query score
    for term in query.lower().split():
        inverted_index_entry = inverted_index.get(term, {}) # term -> {chunk_id: term_freq}
        df = len(inverted_index_entry) # number of documents containing the term
        if df == 0:
            continue
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
        for chunk_id, term_freq in inverted_index_entry.items():
            denominator = term_freq + 1.5 * (1-0.75 + 0.75 * (doc_lengths[chunk_id] / avg_doc_len))
            scores_map[chunk_id] = scores_map.get(chunk_id, 0) + idf * ((term_freq * (1.5 + 1)) / denominator)
    return scores_map