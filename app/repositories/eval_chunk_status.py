"""
eval_chunk_status (
    chunk_id TEXT PRIMARY KEY,
    status TEXT CHECK (status IN ('generated', 'skipped', 'pending', 'failed')),
    reason TEXT,
    updated_at TIMESTAMP DEFAULT now()
);
"""

import psycopg

from app.clients import get_conn

def mark_chunk_status(job_id, chunk_id, status, reason=None):
    try:
        with get_conn() as conn:
            with conn.transaction() as trans:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO eval_chunk_status (job_id, chunk_id, status, reason)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (job_id, chunk_id) DO UPDATE SET status = EXCLUDED.status, reason = EXCLUDED.reason, updated_at = now()
                        """,
                        (job_id, chunk_id, status, reason)
                    )
    except Exception as e:
        print(f"Error marking chunk status: {e}")
        raise e
    return True

def is_chunk_already_done(jobId,chunk_id):
    try:
        with get_conn() as conn:
            with conn.transaction() as trans:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT status FROM eval_chunk_status WHERE chunk_id = %s AND job_id = %s
                        """,
                        (chunk_id, jobId)
                    )
                    result = cur.fetchone()
                    if result and result[0] == 'persisted':
                        return True
    except Exception as e:
        print(f"Error checking chunk status: {e}")
    return False

def fetch_chunks_status(jobId, status_filter=None):
    try:
        with get_conn() as conn:
                with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                    if status_filter:
                        cur.execute(
                            """
                            SELECT chunk_id, status, reason, updated_at FROM eval_chunk_status WHERE job_id = %s AND status = ANY(%s)
                            """,
                            (jobId, status_filter)
                        )
                    else:
                        cur.execute(
                            """
                            SELECT chunk_id, status, reason, updated_at FROM eval_chunk_status WHERE job_id = %s
                            """,
                            (jobId,)
                        )
                    return cur.fetchall()
    except Exception as e:
        print(f"Error fetching chunks status: {e}")
        return []
    
def fetch_chunks_data_by_status(jobId, status_filter:list[str]):
    """
    Fetches chunk data from eval_chunk_status table & chunks table for a given jobId and status.
    """
    try:
        with get_conn() as conn:
            with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                cur.execute(
                    """
                    SELECT ecs.chunk_id, ecs.status, ecs.reason, ecs.updated_at, c.text, c.chunk_index, c.source_id
                    FROM eval_chunk_status ecs
                    JOIN chunks c ON ecs.chunk_id = c.chunk_id
                    WHERE ecs.job_id = %s AND ecs.status = ANY(%s)
                    """,
                    (jobId, status_filter)
                )
                return cur.fetchall()
    except Exception as e:
        print(f"Error fetching chunks data by status: {e}")
        return []
    
    
def is_chunk_already_generated(jobId,chunk_id):
    try:
        with get_conn() as conn:
            with conn.transaction() as trans:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT status FROM eval_chunk_status WHERE chunk_id = %s AND job_id = %s
                        """,
                        (chunk_id, jobId)
                    )
                    result = cur.fetchone()
                    if result and result[0] in ['generated', 'skipped']:
                        return True
    except Exception as e:
        print(f"Error checking chunk generation status: {e}")
    return False