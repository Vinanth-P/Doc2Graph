"""Ingest routes — POST /upload, GET /jobs/{id}, GET /jobs."""
from __future__ import annotations

import io
import zipfile
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool
from langdetect import detect, DetectorFactory, LangDetectException

from src.api import job_store, workers
from src.api.auth import require_api_key
from src.api.schemas import JobStatusResponse, UploadResponse
from src.config import Settings, get_settings
from src.ingestion.loader import load_document, load_txt

DetectorFactory.seed = 0  # deterministic langdetect

router = APIRouter(prefix="/api", tags=["ingest"])

CHUNK_SIZE = 8192


async def _read_and_validate(
    file: UploadFile, settings: Settings
) -> tuple[bytes, str]:
    """Stream file, enforce size limit, detect type via magic bytes."""
    max_bytes = settings.max_upload_mb * 1024 * 1024

    if file.size and file.size > max_bytes:
        raise HTTPException(413, "File exceeds size limit")

    chunks, total = [], 0
    while True:
        chunk = await file.read(CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(413, "File exceeds size limit during upload")
        chunks.append(chunk)

    content = b"".join(chunks)

    # Magic-byte type detection
    if content.startswith(b"%PDF"):
        ftype = "pdf"
    elif content.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                if {"[Content_Types].xml", "word/document.xml"} <= set(zf.namelist()):
                    ftype = "docx"
                else:
                    raise HTTPException(415, "ZIP is not a DOCX")
        except zipfile.BadZipFile:
            raise HTTPException(415, "Corrupt file")
    else:
        try:
            load_txt(content)  # Validate the entire payload, not just its header.
            ftype = "txt"
        except ValueError as exc:
            raise HTTPException(415, str(exc)) from exc

    return content, ftype


def _sample_text(content: bytes, ftype: str) -> str:
    """Use the real loader; a parsing failure must never decode raw binary."""
    return load_document(content, ftype)[:3000]


@router.post("/upload", response_model=UploadResponse, dependencies=[Depends(require_api_key)])
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
) -> UploadResponse:
    """Upload a document (PDF / DOCX / TXT) for knowledge graph extraction."""
    content, ftype = await _read_and_validate(file, settings)

    # Language gate — must run BEFORE background task dispatch
    try:
        sample = await run_in_threadpool(_sample_text, content, ftype)
    except Exception as exc:
        raise HTTPException(422, f"Could not extract readable document text: {exc}") from exc
    try:
        lang = detect(sample)
    except LangDetectException:
        lang = "unknown"
    if lang != "en":
        raise HTTPException(422, f"Language '{lang}' not supported. English only.")

    job_id = str(uuid4())
    await job_store.create_job(job_id, file.filename or "unknown")

    background_tasks.add_task(
        workers.dispatch_pipeline,
        job_id, content, file.filename or "unknown", ftype,
    )
    return UploadResponse(job_id=job_id, status="queued")


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    row = await job_store.get_job(job_id)
    if row is None:
        raise HTTPException(404, f"Job {job_id!r} not found")
    return JobStatusResponse(**row)


@router.get("/jobs", response_model=list[JobStatusResponse])
async def list_jobs() -> list[JobStatusResponse]:
    rows = await job_store.list_jobs()
    return [JobStatusResponse(**r) for r in rows]
