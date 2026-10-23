"""SQLite-backed job status store (async via aiosqlite)."""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

import aiosqlite

DB_PATH = Path("data/jobs.sqlite")


class JobStatus(str, Enum):
    QUEUED  = "queued"
    RUNNING = "running"
    DONE    = "done"
    FAILED  = "failed"


async def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
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
        await db.commit()


async def create_job(job_id: str, filename: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO jobs (id, status, filename, created_at)
            VALUES (?, ?, ?, datetime('now'))
            """,
            (job_id, JobStatus.QUEUED, filename),
        )
        await db.commit()


async def update_job(job_id: str, **kwargs) -> None:
    if not kwargs:
        return
    # Serialize result dict to JSON string if provided
    if "result" in kwargs and isinstance(kwargs["result"], dict):
        kwargs["result"] = json.dumps(kwargs["result"])
    cols = ", ".join(f"{k} = ?" for k in kwargs)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            f"UPDATE jobs SET {cols} WHERE id = ?",
            (*kwargs.values(), job_id),
        )
        await db.commit()


async def get_job(job_id: str) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
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
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]
