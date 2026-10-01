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
    print(f"Inserting {eval_data}  into eval_dataset")
    try:
        with get_conn() as conn:
            with conn.transaction() as trans:
                with conn.cursor() as cur:
                    for data in eval_data:
                        # generate UUID
                        data['id'] = str(uuid4())
                        cur.execute(
                            """
                            INSERT INTO eval_dataset (id, chunk_id, corpus_id, question, answer, language, cross_lingual, chunk_text, source, job_id)
                            VALUES ( %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                data['id'],
                                data['chunk_id'],
                                data['source_id'],
                                data['question'],
                                data['answer'],
                                data['language'],
                                data.get('cross_lingual', False),
                                data.get('chunk_text', None),
                                data['source'],
                                data['job_id']
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
                        # mark the chunk as persisted in eval_chunk_status
                        cur.execute(
                            """
                            UPDATE eval_chunk_status
                            SET status = 'persisted', updated_at = now()
                            WHERE chunk_id = %s AND job_id = %s
                            """,
                            (data['chunk_id'], data['job_id'])
                        )
                        
                        # clean WAL log for the jobId and chunk_id
                        cur.execute(
                            """
                            DELETE FROM eval_dataset_wal
                            WHERE job_id = %s AND %s = ANY(chunks_ids)
                            """,
                            (data['job_id'], data['chunk_id'])
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

def batch_mark_chunk_status(jobId, chunk_ids, status, reason=None):
    try:
        with get_conn() as conn:
            with conn.transaction() as trans:
                with conn.cursor() as cur:
                    for chunk_id in chunk_ids:
                        cur.execute(
                            """
                            UPDATE eval_chunk_status
                            SET status = %s, reason = %s, updated_at = now()
                            WHERE chunk_id = %s AND job_id = %s
                            """,
                            (status, reason, chunk_id, jobId)
                        )
        return True
    except Exception as e:
        print(f"Error batch marking status: {e}")
        return False