import psycopg

from clients import get_conn

def insert_documents_chunks_trans(documents, chunks):
    with get_conn() as conn:
        with conn.transaction() as trans:
            with conn.cursor() as cur:
                try:
                    for document in documents:
                        cur.execute(
                            "INSERT INTO documents (doc_id, filename) VALUES (%s, %s) ON CONFLICT (doc_id) DO NOTHING",
                            (document['doc_id'], document['filename'])
                        )
                    for chunk in chunks:
                        cur.execute(
                            "INSERT INTO chunks (chunk_id, doc_id, text) VALUES (%s, %s, %s) ON CONFLICT (chunk_id) DO NOTHING",
                            (chunk['chunk_id'], chunk['doc_id'], chunk['text'])
                        )
                except Exception as e:
                    print(f"Error occurred while inserting documents and chunks: {e}")
                    raise e
                

def text_search_ts_rank(query, top_k=20):
    print(f"Performing text search for query: '{query}' with top_k={top_k}")
    try:
        terms = query.split()
        if not terms:
            return []

        # websearch_to_tsquery uses plain-search syntax: the literal word
        # "or" means OR, not the "|" operator (that belongs to to_tsquery,
        # a different function). Using "|" here was silently ignored/broken.
        or_query = " OR ".join(terms)

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT chunk_id, ts_rank(text_search, query) AS score
                    FROM chunks, websearch_to_tsquery('english', %s) AS query
                    WHERE text_search @@ query
                    ORDER BY score DESC
                    LIMIT %s
                    """,
                    (or_query, top_k)
                )
                return cur.fetchall()
    except Exception as e:
        print(f"Error occurred while performing text search: {e}")
        return []


def fetch_chunks_by_ids(chunk_ids):
    with get_conn() as conn:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(
                "SELECT chunk_id, text FROM chunks WHERE chunk_id = ANY(%s)",
                (chunk_ids,)
            )
            rows = cur.fetchall()
    return rows