from uuid import uuid4

import psycopg

from app.clients import get_conn

"""
eval_dataset (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id    text REFERENCES chunks(chunk_id),
    corpus_id      uuid REFERENCES sources(source_id),
    question    TEXT NOT NULL,
    answer      TEXT NOT NULL,
    language    text,  -- 'fr' or 'en'
    cross_lingual BOOLEAN DEFAULT FALSE,
    chunk_text TEXT,              -- snapshot of chunk at generation time
    source source_type NOT NULL,
    generated_at TIMESTAMPTZ DEFAULT NOW(),
    reviewed    BOOLEAN DEFAULT FALSE,  -- doctor sign-off flag
    review_note TEXT
    )
"""

def insert_eval_dataset(eval_data: list[dict]):
    try:
        with get_conn() as conn:
            with conn.transaction() as trans:
                with conn.cursor() as cur:
                    for data in eval_data:
                        # generate UUID
                        data['id'] = str(uuid4())
                        cur.execute(
                            """
                            INSERT INTO eval_dataset (id, chunk_id, corpus_id, question, answer, language, cross_lingual, chunk_text, source)
                            VALUES ( %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                data['id'],
                                data['chunk_id'],
                                data['corpus_id'],
                                data['question'],
                                data['answer'],
                                data['language'],
                                data.get('cross_lingual', False),
                                data.get('chunk_text', None),
                                data['source']
                            )
                        )
                        for expected_chunk in data.get('expected_chunks_ids'):
                            cur.execute(
                                """
                                INSERT INTO eval_expected_chunks (eval_dataset_id, chunk_id)
                                VALUES (%s, %s)
                                """,
                                (data['id'], expected_chunk)
                            )
        return True
    except Exception as e:
        print(f"Error inserting eval dataset: {e}")
        return False
    
def fetch_eval_dataset(limit=10):
    try:
        with get_conn() as conn:
            with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                cur.execute(
                    """
                    SELECT * FROM eval_dataset
                    ORDER BY generated_at DESC
                    LIMIT %s
                    """,
                    (limit,)
                )
                rows = cur.fetchall()
                for row in rows:
                    cur.execute(
                        """
                        SELECT chunk_id FROM eval_expected_chunks
                        WHERE eval_dataset_id = %s
                        """,
                        (row['id'],)
                    )
                    expected_chunks = cur.fetchall()
                    row['expected_chunks_ids'] = [ec['chunk_id'] for ec in expected_chunks]
                return rows
    except Exception as e:
        print(f"Error fetching eval dataset: {e}")
        return []
