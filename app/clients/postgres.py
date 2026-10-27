"""
Shared Postgres connection pool.

Import `get_conn()` anywhere in the codebase to borrow a connection:

    from db import get_conn

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM chunks WHERE doc_id = %s", (doc_id,))
            rows = cur.fetchall()

Call `init_pool()` once at app startup (e.g. FastAPI startup event) and
`close_pool()` once at shutdown, so connection failures happen predictably
at boot instead of randomly on first request.
"""

import os
from contextlib import contextmanager

import psycopg
from psycopg_pool import ConnectionPool
import os
from dotenv import load_dotenv

load_dotenv()

_pool: ConnectionPool | None = None

def init_pool(min_size: int = 2, max_size: int = 10) -> None:
    """Create the pool. Call once at app startup."""
    global _pool
    if _pool is not None:
        return  # already initialized, don't double-open

    conninfo = os.getenv("POSTGRES_URL")
    print(f"Initializing Postgres connection pool with conninfo: {conninfo}")
    if not conninfo:
        raise RuntimeError(
            "DATABASE_URL env var is not set. "
            "Expected format: postgresql://user:password@host:port/dbname"
        )

    _pool = ConnectionPool(
        conninfo=conninfo,
        min_size=min_size,
        max_size=max_size,
        reconnect_failed=True,
        open=True,
        check=ConnectionPool.check_connection,
        max_idle=300
    )


def close_pool() -> None:
    """Close the pool. Call once at app shutdown."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


@contextmanager
def get_conn() -> psycopg.Connection:
    """
    Borrow a connection from the pool for the duration of a `with` block.
    Automatically returns it to the pool afterward, and rolls back any
    open transaction if an exception occurs inside the block.

    Usage:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(...)
    """
    if _pool is None:
        raise RuntimeError(
            "Connection pool not initialized. Call init_pool() at app startup first."
        )
    with _pool.connection() as conn:
        yield conn
