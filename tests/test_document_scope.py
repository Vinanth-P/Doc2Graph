"""Document scope propagation and real Cypher isolation checks."""
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from src.api.app import app
from src.api.routes import graph, search
from src.config import get_settings
from src.search.query_engine import QueryEngine


@pytest.mark.parametrize("path,method,result", [
    ("/api/graph/visualization?entity_id=shared", "get_visualization_data", {"nodes": [], "links": []}),
    ("/api/graph/stats?", "get_graph_stats", {"entity_count": 0, "edge_count": 0, "doc_count": 0}),
    ("/api/search?q=Google", "search_entities", []),
    ("/api/graph/entities/shared?", "get_entity_relations", {"id": "shared", "name": "Shared", "type": "ORG", "outgoing": []}),
])
def test_web_endpoints_forward_document_scope(monkeypatch, path, method, result):
    engine = SimpleNamespace(**{method: AsyncMock(return_value=result)})
    for module in (graph, search):
        monkeypatch.setattr(module, "get_driver", AsyncMock())
        monkeypatch.setattr(module, "QueryEngine", lambda driver: engine)
    response = TestClient(app).get(path + "&doc_id=uploaded-document")
    assert response.status_code == 200
    assert getattr(engine, method).call_args.kwargs["doc_id"] == "uploaded-document"


@pytest.mark.skipif(os.environ.get("DOC2GRAPH_NEO4J_TESTS") != "1", reason="Set DOC2GRAPH_NEO4J_TESTS=1 for rollback-only Neo4j checks")
@pytest.mark.asyncio
async def test_shared_entities_do_not_leak_other_documents():
    """Use actual Neo4j queries, rolling back all fixture data even on failure."""
    from src.graph.neo4j_client import create_driver
    settings = get_settings()
    driver = create_driver(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    prefix = "scope-test-" + uuid4().hex
    a, b = prefix + "-a", prefix + "-b"
    person, org, only_a, only_b, lone = [prefix + name for name in ("-person", "-org", "-only-a", "-only-b", "-lone")]
    try:
        async with driver.session() as session:
            tx = await session.begin_transaction()
            try:
                await tx.run(
                    """
                    CREATE (a:Document {id: $a}), (b:Document {id: $b}),
                           (ca:Chunk {id: $a + ':0', doc_id: $a}),
                           (ca2:Chunk {id: $a + ':1', doc_id: $a}),
                           (cb:Chunk {id: $b + ':0', doc_id: $b}),
                           (p:Entity {id: $person, name: 'QuasarPerson', type: 'PERSON'}),
                           (o:Entity {id: $org, name: 'QuasarOrg', type: 'ORG'}),
                           (oa:Entity {id: $only_a, name: 'QuasarAlpha', type: 'ORG'}),
                           (ob:Entity {id: $only_b, name: 'QuasarBeta', type: 'ORG'}),
                           (l:Entity {id: $lone, name: 'QuasarLone', type: 'ORG'}),
                           (p)-[:MENTIONED_IN]->(ca), (p)-[:MENTIONED_IN]->(cb),
                           (o)-[:MENTIONED_IN]->(ca), (o)-[:MENTIONED_IN]->(cb),
                           (oa)-[:MENTIONED_IN]->(ca), (ob)-[:MENTIONED_IN]->(cb),
                           (l)-[:MENTIONED_IN]->(ca),
                           (p)-[:EMPLOYED_BY {chunk_ids: [$a + ':0', $a + ':1', $b + ':0'], confidence: 0.9, evidence_count: 3}]->(o),
                           (p)-[:FOUNDED {chunk_ids: [$a + ':0'], confidence: 0.8, evidence_count: 1}]->(oa),
                           (p)-[:FOUNDED {chunk_ids: [$b + ':0'], confidence: 0.8, evidence_count: 1}]->(ob)
                    """, a=a, b=b, person=person, org=org, only_a=only_a, only_b=only_b, lone=lone,
                )

                class TransactionSession:
                    async def __aenter__(self):
                        return tx
                    async def __aexit__(self, *args):
                        pass

                engine = QueryEngine(SimpleNamespace(session=lambda: TransactionSession()))
                first = await engine.get_visualization_data(doc_id=a)
                second = await engine.get_visualization_data(doc_id=b)
                assert {n["id"] for n in first["nodes"]} == {person, org, only_a, lone}
                assert {n["id"] for n in second["nodes"]} == {person, org, only_b}
                assert len(first["links"]) == len(second["links"]) == 2
                assert next(e for e in first["links"] if e["type"] == "EMPLOYED_BY")["evidence_count"] == 2
                assert next(e for e in second["links"] if e["type"] == "EMPLOYED_BY")["evidence_count"] == 1
                assert await engine.get_graph_stats(doc_id=a) == {"entity_count": 4, "edge_count": 2, "doc_count": 1}
                focused = await engine.get_visualization_data(entity_id=org, doc_id=b)
                assert [(e["source"], e["target"]) for e in focused["links"]] == [(person, org)]
                detail = await engine.get_entity_relations(person, doc_id=a)
                assert {r["target_id"] for r in detail["outgoing"]} == {org, only_a}
                assert (await engine.get_entity_relations(lone, doc_id=a))["outgoing"] == []
                assert await engine.get_entity_relations(only_b, doc_id=a) == {}
                assert await engine.search_entities("QuasarBeta", doc_id=a) == []
                assert {n["id"] for n in await engine.search_entities("QuasarBeta", doc_id=b)} == {only_b}
                assert await engine.get_visualization_data(doc_id=prefix + "-missing") == {"nodes": [], "links": []}
                assert await engine.get_graph_stats(doc_id=prefix + "-missing") == {"entity_count": 0, "edge_count": 0, "doc_count": 0}
            finally:
                await tx.rollback()
    finally:
        await driver.close()
