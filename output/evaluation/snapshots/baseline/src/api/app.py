"""FastAPI application — lifespan, CORS, routers, static UI."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api import job_store
from src.api.routes import ingest, graph, search
from src.config import get_settings
from src.graph.neo4j_client import get_driver, run_schema, close_driver


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB + Neo4j schema. Shutdown: close driver."""
    settings = get_settings()

    # Init SQLite job store
    await job_store.init_db()

    # Init Neo4j schema (constraints + indexes) — idempotent IF NOT EXISTS
    try:
        driver = await get_driver(
            settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password
        )
        await run_schema(driver)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            "Neo4j schema init failed (Neo4j may not be ready yet): %s", exc
        )

    yield

    await close_driver()


app = FastAPI(
    title="Doc2Graph",
    description="Knowledge Graph Builder from Documents — Classical ML Edition",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routers
app.include_router(ingest.router)
app.include_router(graph.router)
app.include_router(search.router)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# Serve the D3.js UI at /
app.mount("/", StaticFiles(directory="src/api/static", html=True), name="static")
