"""Background worker — dispatches pipeline to thread pool with file-lock guard."""
from __future__ import annotations

import asyncio
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor

from src.api import job_store
from src.pipeline import run_pipeline

logger = logging.getLogger(__name__)

# Single-thread executor — one pipeline at a time
_executor = ThreadPoolExecutor(max_workers=1)


async def dispatch_pipeline(
    job_id: str,
    raw_bytes: bytes,
    filename: str,
    ftype: str,
) -> None:
    """Run the pipeline in a background thread, update job store on completion."""
    await job_store.update_job(job_id, status=job_store.JobStatus.RUNNING)

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _executor,
            run_pipeline,
            raw_bytes,
            filename,
            ftype,
            job_id,
        )
        await job_store.update_job(
            job_id,
            status=job_store.JobStatus.DONE,
            finished_at="datetime('now')",
            result=result,
        )
        logger.info("[%s] Pipeline completed: %s", job_id, result)
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("[%s] Pipeline failed: %s\n%s", job_id, exc, tb)
        await job_store.update_job(
            job_id,
            status=job_store.JobStatus.FAILED,
            finished_at="datetime('now')",
            error=str(exc),
        )
