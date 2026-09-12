import psycopg

from app.clients import get_conn
from psycopg.types.json import Json
def insert_sources_chunks_trans(sources, chunks):
    with get_conn() as conn:
        with conn.transaction() as trans:
            with conn.cursor() as cur:
                try:
                    for source in sources:
                        cur.execute(
                            "INSERT INTO sources (source_id, source_type, file_name, file_path) VALUES (%s, %s, %s, %s) ON CONFLICT (source_id) DO NOTHING",
                            (source['source_id'], source['source_type'], source['filename'], source['file_path'])
                        )
                    for chunk in chunks:
                        cur.execute(
                            "INSERT INTO chunks (chunk_id, source_id, text, chunk_index, metadata, language) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (chunk_id) DO NOTHING",
                            (chunk['chunk_id'], chunk['source_id'], chunk['text'], chunk['chunk_index'], Json(chunk['metadata']), chunk['language'])
                        )
                except Exception as e:
                    print(f"Error occurred while inserting documents and chunks: {e}")
                    raise e



def text_search_ts_rank(query, language, top_k=20):
    print(f"Performing text search for query: '{query}' with top_k={top_k}")
    try:
        terms = query.split()
        if not terms:
            return []

        # websearch_to_tsquery uses plain-search syntax: the literal word
        # "or" means OR, not the "|" operator (that belongs to to_tsquery,
        # a different function). Using "|" here was silently ignored/broken.
        or_query = " OR ".join(terms)
        print(f"searching in language: {language} with query: {or_query}")
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT chunk_id, ts_rank(text_search, query) AS score
                    FROM chunks, websearch_to_tsquery(%s, %s) AS query
                    WHERE text_search @@ query and language = %s
                    ORDER BY score DESC
                    LIMIT %s
                    """,
                    (language, or_query, language, top_k)
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

def get_source_meta(doc_id):
    with get_conn() as conn:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute("SELECT source_id, file_name FROM sources WHERE source_id = %s", (doc_id,))
            row = cur.fetchone()
    return row