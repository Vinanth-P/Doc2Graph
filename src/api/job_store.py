"""SQLite-backed job status store (async via aiosqlite).

Concurrency Architecture:
  - Local single-process deployment uses SQLite in Write-Ahead Logging (WAL)
    mode with a generous busy timeout (30.0s) to allow concurrent API reads
    (status polling) alongside background pipeline writes without SQLITE_BUSY locks.
  - Multi-container scaling: When scaling beyond a single container replica in
    docker-compose, SQLite and local file locks should be replaced with distributed
    primitives: either storing job state in Neo4j metadata nodes (:Job {id, status, ...})
    or backing jobs with Redis / Celery.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from typing import AsyncIterator

import aiosqlite

DB_PATH = Path("data/jobs.sqlite")
SQLITE_TIMEOUT = 30.0


class JobStatus(str, Enum):
    QUEUED  = "queued"
    RUNNING = "running"
    DONE    = "done"
    FAILED  = "failed"


@asynccontextmanager
async def get_db_connection() -> AsyncIterator[aiosqlite.Connection]:
    """Provide an aiosqlite connection configured with WAL and busy timeout."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH, timeout=SQLITE_TIMEOUT) as db:
        await db.execute("PRAGMA journal_mode=WAL;")
        await db.execute("PRAGMA busy_timeout = 30000;")
        await db.execute("PRAGMA synchronous = NORMAL;")
        yield db


async def init_db() -> None:
    """Initialize jobs schema and ensure WAL mode is active."""
    async with get_db_connection() as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id          TEXT PRIMARY KEY,
                status      TEXT NOT NULL,
                filename    TEXT,
                created_at  TEXT,
                finished_at TEXT,
                result      TEXT,
                error       TEXT
            )
            """
        )
        # Clean up any leftover jobs from prior interrupted server run
        await db.execute(
            "UPDATE jobs SET status = 'failed', error = 'Interrupted by server restart' WHERE status IN ('running', 'queued')"
        )
        await db.commit()


async def create_job(job_id: str, filename: str) -> None:
    now_iso = datetime.now(timezone.utc).isoformat()
    async with get_db_connection() as db:
        await db.execute(
            """
            INSERT INTO jobs (id, status, filename, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (job_id, JobStatus.QUEUED, filename, now_iso),
        )
        await db.commit()


async def update_job(job_id: str, **kwargs) -> None:
    if not kwargs:
        return
    # Normalize finished_at if passed as SQL literal string
    if kwargs.get("finished_at") == "datetime('now')":
        kwargs["finished_at"] = datetime.now(timezone.utc).isoformat()

    # Serialize result dict to JSON string if provided
    if "result" in kwargs and isinstance(kwargs["result"], dict):
        kwargs["result"] = json.dumps(kwargs["result"])

    cols = ", ".join(f"{k} = ?" for k in kwargs)
    async with get_db_connection() as db:
        await db.execute(
            f"UPDATE jobs SET {cols} WHERE id = ?",
            (*kwargs.values(), job_id),
        )
        await db.commit()


async def get_job(job_id: str) -> dict | None:
    async with get_db_connection() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM jobs WHERE id = ?", (job_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            data = dict(row)
            if data.get("result"):
                try:
                    data["result"] = json.loads(data["result"])
                except json.JSONDecodeError:
                    pass
            return data


async def list_jobs(limit: int = 50) -> list[dict]:
    async with get_db_connection() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ) as cursor:
            rows = await cursor.fetchall()
            results = []
            for row in rows:
                data = dict(row)
                if data.get("result"):
                    try:
                        data["result"] = json.loads(data["result"])
                    except json.JSONDecodeError:
                        pass
                results.append(data)
            return results
