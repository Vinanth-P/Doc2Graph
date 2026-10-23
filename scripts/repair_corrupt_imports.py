"""Archive binary-decoded imports and rebuild from verified original PDFs.

Read-only preview by default. --apply preserves old nodes and relationships
under ArchivedEntity/ArchivedChunk/ArchivedDocument labels, exports a JSON
snapshot, then re-ingests the originals. This repair deliberately refuses a
mixed database containing documents outside the supplied affected set.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import pipeline_lock
from src.config import get_settings
from src.graph.neo4j_client import create_driver
from src.ingestion.loader import load_document
from src.ingestion.pdf_cleaner import clean_pdf_text
from src.ingestion.preprocessor import preprocess
from src.pipeline import run_pipeline


async def archive_imports(sources: dict[str, Path], apply: bool) -> dict:
    settings = get_settings()
    driver = create_driver(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    try:
        async with driver.session() as session:
            result = await session.run("MATCH (d:Document) RETURN d.id AS id, d.filename AS filename")
            documents = await result.data()
            if {d["id"] for d in documents} != set(sources):
                raise RuntimeError("Repair requires the exact affected active document set; no changes made.")
            result = await session.run(
                "MATCH (c:Chunk) WHERE c.sent_idx = 0 RETURN c.doc_id AS id, c.text AS text"
            )
            first_chunks = {r["id"]: r["text"] for r in await result.data()}
            if not all(first_chunks.get(doc_id, "").lstrip().startswith("%PDF-") for doc_id in sources):
                raise RuntimeError("Not every supplied document has confirmed raw-PDF corruption; no changes made.")
            result = await session.run(
                "MATCH (n) WHERE n:Entity OR n:Chunk OR n:Document "
                "RETURN elementId(n) AS element_id, labels(n) AS labels, properties(n) AS properties"
            )
            nodes = await result.data()
            # Refuse to archive nodes whose provenance cannot be traced to the
            # verified files (canonical entity IDs start with the document hash).
            for node in nodes:
                identifier = node["properties"].get("id", "")
                if not any(identifier == doc_id or identifier.startswith(doc_id + ":") for doc_id in sources):
                    raise RuntimeError("Found nodes outside the affected imports; no changes made.")
            result = await session.run(
                "MATCH (a)-[r]->(b) WHERE (a:Entity OR a:Chunk OR a:Document) "
                "AND (b:Entity OR b:Chunk OR b:Document) "
                "RETURN elementId(a) AS source, elementId(b) AS target, type(r) AS type, properties(r) AS properties"
            )
            relationships = await result.data()
            report = {"documents": documents, "nodes": len(nodes), "relationships": len(relationships)}
            if not apply:
                return report
            batch = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            directory = Path("data/backups")
            directory.mkdir(parents=True, exist_ok=True)
            backup = directory / f"corrupt-imports-{batch}.json"
            with backup.open("x", encoding="utf-8") as handle:
                json.dump({"batch": batch, "nodes": nodes, "relationships": relationships}, handle, ensure_ascii=True, default=str)

            async def archive(tx):
                for label in ("Entity", "Chunk", "Document"):
                    await tx.run(
                        f"MATCH (n:{label}) WHERE elementId(n) IN $ids "
                        f"SET n:Archived{label}, n.archive_batch = $batch, "
                        "n.archive_reason = 'Raw PDF bytes were processed as text' "
                        f"REMOVE n:{label}",
                        ids=[n["element_id"] for n in nodes], batch=batch,
                    )
            await session.execute_write(archive)
            return {**report, "archive_batch": batch, "backup": str(backup)}
    finally:
        await driver.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    sources = {}
    for path in args.files:
        raw = path.read_bytes()
        text = clean_pdf_text(load_document(raw, "pdf"))
        # Validate extraction and English segmentation before touching storage.
        preprocess(raw, path.name, text=text)
        sources[hashlib.sha256(raw).hexdigest()] = path
    pipeline_lock.acquire()
    try:
        report = asyncio.run(archive_imports(sources, args.apply))
        print(json.dumps(report))
        if args.apply:
            for path in sources.values():
                result = run_pipeline(path.read_bytes(), path.name, "pdf", job_id="repair")
                print(json.dumps({"filename": path.name, "result": result}))
    finally:
        pipeline_lock.release()


if __name__ == "__main__":
    main()
