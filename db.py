"""
db.py — Database abstraction layer.

Automatically uses:
  - PostgreSQL  when DATABASE_URL env var is set  (production on Render)
  - SQLite      otherwise                          (local development)

Both backends expose the same functions so api.py never touches raw SQL directly.
"""

import os
from contextlib import contextmanager

DATABASE_URL = os.getenv("DATABASE_URL")
_ROOT        = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════════════════════
#  PostgreSQL backend
# ══════════════════════════════════════════════════════════════════════════════


if DATABASE_URL:
    import psycopg2
    import psycopg2.extras

    @contextmanager
    def get_db():
        conn = psycopg2.connect(DATABASE_URL)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db() -> None:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS agents (
                        id           SERIAL PRIMARY KEY,
                        name         TEXT UNIQUE NOT NULL,
                        description  TEXT,
                        method       TEXT,
                        input_format TEXT,
                        file_path    TEXT,
                        source_code  TEXT,
                        synced_at    TEXT,
                        is_active    BOOLEAN NOT NULL DEFAULT TRUE
                    )
                """)
                # Migration: add column to existing tables that predate this field
                cur.execute("""
                    ALTER TABLE agents
                    ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE
                """)

    def fetchall(conn, sql: str, params: tuple = ()) -> list[dict]:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    def fetchone(conn, sql: str, params: tuple = ()) -> dict | None:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return dict(row) if row else None

    def execute(conn, sql: str, params: tuple = ()) -> None:
        with conn.cursor() as cur:
            cur.execute(sql, params)

    def upsert_agent(conn, meta: dict, synced_at: str) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO agents
                    (name, description, method, input_format, file_path, source_code, synced_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (name) DO UPDATE SET
                    description  = EXCLUDED.description,
                    method       = EXCLUDED.method,
                    input_format = EXCLUDED.input_format,
                    file_path    = EXCLUDED.file_path,
                    source_code  = EXCLUDED.source_code,
                    synced_at    = EXCLUDED.synced_at
                """,
                (
                    meta["name"], meta["description"], meta["method"],
                    meta["input_format"], meta["file_path"], meta["source_code"],
                    synced_at,
                ),
            )

    def count_agents(conn) -> int:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM agents")
            return cur.fetchone()[0]

# ══════════════════════════════════════════════════════════════════════════════
#  SQLite backend  (local development)
# ══════════════════════════════════════════════════════════════════════════════

else:
    import sqlite3

    DB_PATH = os.path.join(_ROOT, "agents.db")

    @contextmanager
    def get_db():
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db() -> None:
        with get_db() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agents (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    name         TEXT UNIQUE NOT NULL,
                    description  TEXT,
                    method       TEXT,
                    input_format TEXT,
                    file_path    TEXT,
                    source_code  TEXT,
                    synced_at    TEXT,
                    is_active    INTEGER NOT NULL DEFAULT 1
                )
            """)
            # Migration: add column to existing tables that predate this field
            try:
                conn.execute("ALTER TABLE agents ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1")
            except Exception:
                pass  # column already exists

    def fetchall(conn, sql: str, params: tuple = ()) -> list[dict]:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def fetchone(conn, sql: str, params: tuple = ()) -> dict | None:
        row = conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def execute(conn, sql: str, params: tuple = ()) -> None:
        conn.execute(sql, params)

    def upsert_agent(conn, meta: dict, synced_at: str) -> None:
        conn.execute(
            """
            INSERT INTO agents
                (name, description, method, input_format, file_path, source_code, synced_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (name) DO UPDATE SET
                description  = excluded.description,
                method       = excluded.method,
                input_format = excluded.input_format,
                file_path    = excluded.file_path,
                source_code  = excluded.source_code,
                synced_at    = excluded.synced_at
            """,
            (
                meta["name"], meta["description"], meta["method"],
                meta["input_format"], meta["file_path"], meta["source_code"],
                synced_at,
            ),
        )

    def count_agents(conn) -> int:
        return conn.execute("SELECT COUNT(*) FROM agents").fetchone()[0]
