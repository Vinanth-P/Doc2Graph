"""Exercise the web upload boundary with real PDF bytes and malformed input."""
import io
import zipfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api import job_store, workers
from src.config import get_settings
from src.ingestion.loader import load_document, load_txt
from src.ingestion.preprocessor import preprocess


@pytest.fixture
def client(monkeypatch):
    settings = SimpleNamespace(max_upload_mb=50, api_key="")
    app.dependency_overrides[get_settings] = lambda: settings
    monkeypatch.setattr(job_store, "create_job", AsyncMock())
    monkeypatch.setattr(workers, "dispatch_pipeline", AsyncMock())
    yield TestClient(app)  # No lifespan: do not touch the live database.
    app.dependency_overrides.clear()


def test_pdf_upload_extracts_text_instead_of_binary(client):
    import fitz
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "Elon Musk founded Tesla. Google acquired Microsoft.")
        raw = document.tobytes()
    response = client.post("/api/upload", files={"file": ("example.pdf", raw, "application/pdf")})
    assert response.status_code == 200
    assert workers.dispatch_pipeline.call_args.args[-1] == "pdf"
    text = load_document(raw, "pdf")
    assert "Elon Musk founded Tesla" in text
    assert not text.startswith("%PDF")
    parsed = preprocess(raw, "example.pdf", text=text)
    assert all("%PDF" not in c.text and "\ufffd" not in c.text for c in parsed.chunks)


@pytest.mark.parametrize("payload", [
    b"%PDF-1.4\nThis is an English document, but the PDF is malformed.",
    b"The company is based in London. " * 40 + b"\x00\x01\xff\x81",
    "This English text contains corrupt data \ufffd".encode(),
])
def test_bad_upload_never_queues_a_job(client, payload):
    response = client.post("/api/upload", files={"file": ("broken.txt", payload)})
    assert response.status_code in {415, 422}
    job_store.create_job.assert_not_called()
    workers.dispatch_pipeline.assert_not_called()


def test_other_office_zip_is_not_docx(client):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("xl/workbook.xml", "<workbook/>")
    response = client.post("/api/upload", files={"file": ("book.xlsx", stream.getvalue())})
    assert response.status_code == 415


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16", "cp1252"])
def test_unicode_text_upload_uses_valid_decoding(client, encoding):
    text = "René works for Google in London. The company invests in clean energy."
    raw = text.encode(encoding)
    assert load_txt(raw) == text
    response = client.post("/api/upload", files={"file": ("unicode.txt", raw)})
    assert response.status_code == 200


def test_preprocessor_rejects_raw_pdf_even_when_it_looks_english():
    raw = b"%PDF-1.4\nReportLab Generated PDF document in English\n1 0 obj\nendobj"
    with pytest.raises(ValueError, match="binary data"):
        preprocess(raw, "wrong.pdf")
    with pytest.raises(ValueError, match="binary data"):
        preprocess(raw, "wrong.pdf", text=raw.decode())


@pytest.mark.asyncio
async def test_job_list_deserializes_results(tmp_path, monkeypatch):
    monkeypatch.setattr(job_store, "DB_PATH", tmp_path / "jobs.sqlite")
    await job_store.init_db()
    await job_store.create_job("test", "test.pdf")
    await job_store.update_job("test", status="done", result={"entities": 2})
    rows = await job_store.list_jobs()
    assert rows[0]["result"] == {"entities": 2}
