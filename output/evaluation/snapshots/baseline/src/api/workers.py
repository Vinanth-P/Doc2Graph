"""Background worker — dispatches pipeline to thread pool with file-lock guard.

The pipeline is split into two phases to avoid asyncio event-loop conflicts:

1. run_pipeline_nlp  — CPU-bound NLP work (spaCy, NER, relation extraction,
                       entity resolution). Runs in a ThreadPoolExecutor so it
                       never blocks the FastAPI event loop.

2. run_pipeline_write — async Neo4j write. Awaited directly on the FastAPI
                        event loop AFTER the NLP thread finishes, so the
                        async neo4j driver is always bound to the correct loop.
                        (asyncio.run() inside a thread executor raises
                        'Future attached to a different loop' on the driver's
                        internal connection handles.)
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor

from src.api import job_store
from src.pipeline import run_pipeline_nlp, run_pipeline_write

logger = logging.getLogger(__name__)

# Single-thread executor — one pipeline at a time
_executor = ThreadPoolExecutor(max_workers=1)


async def dispatch_pipeline(
    job_id: str,
    raw_bytes: bytes,
    filename: str,
    ftype: str,
) -> None:
    """Run the pipeline in two phases, update job store on completion."""
    await job_store.update_job(job_id, status=job_store.JobStatus.RUNNING)

    loop = asyncio.get_running_loop()
    try:
        # Phase 1: CPU-bound NLP — runs in a thread to avoid blocking the loop
        nlp_result = await loop.run_in_executor(
            _executor,
            run_pipeline_nlp,
            raw_bytes,
            filename,
            ftype,
            job_id,
        )

        # Phase 2: async Neo4j write — runs on the FastAPI event loop so the
        # driver's connection pool is correctly bound to this loop.
        result = await run_pipeline_write(nlp_result, job_id=job_id)

        now_iso = datetime.now(timezone.utc).isoformat()
        await job_store.update_job(
            job_id,
            status=job_store.JobStatus.DONE,
            finished_at=now_iso,
            result=result,
        )
        logger.info("[%s] Pipeline completed: %s", job_id, result)
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("[%s] Pipeline failed: %s\n%s", job_id, exc, tb)
        now_iso = datetime.now(timezone.utc).isoformat()
        await job_store.update_job(
            job_id,
            status=job_store.JobStatus.FAILED,
            finished_at=now_iso,
            error=str(exc),
        )
