"""Unit tests for QueryEngine and RankingEngine."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.search.query_engine import QueryEngine
from src.search.ranking import RankingEngine


@pytest.mark.asyncio
async def test_shortest_path_interpolates_safe_hops():
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__aenter__.return_value = mock_session

    class AsyncResult:
        def __aiter__(self):
            return self
        async def __anext__(self):
            raise StopAsyncIteration

    executed_queries = []
    async def mock_run(query, **kwargs):
        executed_queries.append((query, kwargs))
        return AsyncResult()

    mock_session.run = mock_run

    engine = QueryEngine(mock_driver)
    
    # Test safe_hops upper bound clamp (15 -> 10)
    await engine.shortest_path("src_1", "tgt_1", max_hops=15)
    query, kwargs = executed_queries[-1]
    assert "[*1..10]" in query
    assert kwargs == {"src": "src_1", "tgt": "tgt_1"}

    # Test safe_hops lower bound clamp (-5 -> 1)
    await engine.shortest_path("src_1", "tgt_1", max_hops=-5)
    query, kwargs = executed_queries[-1]
    assert "[*1..1]" in query

    # Test nominal bound (3)
    await engine.shortest_path("src_1", "tgt_1", max_hops=3)
    query, kwargs = executed_queries[-1]
    assert "[*1..3]" in query


@pytest.mark.asyncio
async def test_get_graph_stats_cypher_subqueries():
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__aenter__.return_value = mock_session

    class AsyncRecord:
        def data(self):
            return {"entity_count": 5, "edge_count": 8, "doc_count": 2}

    class AsyncResult:
        def __init__(self):
            self._yielded = False
        def __aiter__(self):
            return self
        async def __anext__(self):
            if not self._yielded:
                self._yielded = True
                return AsyncRecord()
            raise StopAsyncIteration

    executed_queries = []
    async def mock_run(query, **kwargs):
        executed_queries.append((query, kwargs))
        return AsyncResult()

    mock_session.run = mock_run

    engine = QueryEngine(mock_driver)
    stats = await engine.get_graph_stats(doc_id="doc_123")
    assert stats == {"entity_count": 5, "edge_count": 8, "doc_count": 2}
    query, kwargs = executed_queries[0]
    assert "CALL () {" in query
    assert kwargs == {"doc_id": "doc_123"}


@pytest.mark.asyncio
async def test_top_entities_by_degree_restricts_to_entities():
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__aenter__.return_value = mock_session

    class AsyncResult:
        def __aiter__(self):
            return self
        async def __anext__(self):
            raise StopAsyncIteration

    executed_queries = []
    async def mock_run(query, **kwargs):
        executed_queries.append((query, kwargs))
        return AsyncResult()

    mock_session.run = mock_run

    engine = RankingEngine(mock_driver)
    await engine.top_entities_by_degree(limit=5)
    query, kwargs = executed_queries[0]
    assert "MATCH (e:Entity)-[r]-(:Entity)" in query
    assert kwargs == {"limit": 5}


@pytest.mark.asyncio
async def test_top_entities_by_pagerank_lifecycle_and_cleanup():
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__aenter__.return_value = mock_session

    class AsyncRecord:
        def data(self):
            return {"id": "e1", "name": "Tesla", "type": "ORG", "score": 1.25}

    class AsyncResult:
        def __init__(self):
            self._yielded = False
        def __aiter__(self):
            return self
        async def __anext__(self):
            if not self._yielded:
                self._yielded = True
                return AsyncRecord()
            raise StopAsyncIteration

    executed_queries = []
    async def mock_run(query, **kwargs):
        executed_queries.append((query, kwargs))
        return AsyncResult()

    mock_session.run = mock_run

    engine = RankingEngine(mock_driver)
    results = await engine.top_entities_by_pagerank(limit=5)
    assert len(results) == 1
    assert results[0]["name"] == "Tesla"

    # Verify project, stream, and drop queries were called
    assert len(executed_queries) == 3
    proj_query, proj_kw = executed_queries[0]
    stream_query, stream_kw = executed_queries[1]
    drop_query, drop_kw = executed_queries[2]

    assert "CALL gds.graph.project" in proj_query
    assert "'Entity'" in proj_query
    graph_name = proj_kw["graph_name"]
    assert graph_name.startswith("entity_graph_")
    assert stream_kw["graph_name"] == graph_name
    assert drop_kw["graph_name"] == graph_name
    assert "CALL gds.graph.drop" in drop_query


@pytest.mark.asyncio
async def test_top_entities_by_pagerank_drops_graph_on_exception():
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__aenter__.return_value = mock_session

    executed_queries = []
    async def mock_run(query, **kwargs):
        executed_queries.append((query, kwargs))
        if "pageRank.stream" in query:
            raise RuntimeError("GDS stream failure")
        return MagicMock()

    mock_session.run = mock_run

    engine = RankingEngine(mock_driver)
    with pytest.raises(RuntimeError, match="GDS stream failure"):
        await engine.top_entities_by_pagerank(limit=5)

    assert len(executed_queries) == 3
    assert "CALL gds.graph.drop" in executed_queries[2][0]
