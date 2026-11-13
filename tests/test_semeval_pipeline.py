from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from src.nlp.semeval_pipeline import extract_pipeline
from src.api.schemas import ResolvedEdgePayload, ResolvedEntityPayload
from src.graph.graph_builder import GraphBuilder


def test_default_upload_pipeline_dispatches_to_stanford_svm(monkeypatch, tmp_path):
    from src import pipeline
    from src.config import Settings
    from src.nlp import semeval_pipeline

    monkeypatch.delenv('RELATION_BACKEND', raising=False)
    monkeypatch.setenv('CONFIG_YAML', str(tmp_path / 'absent.yaml'))
    settings = Settings(_env_file=None)
    assert settings.relation_backend == 'semeval_svm'
    monkeypatch.setattr(pipeline, 'get_settings', lambda: settings)
    monkeypatch.setattr(pipeline.pipeline_lock, 'acquire', lambda: None)
    monkeypatch.setattr(pipeline.pipeline_lock, 'release', lambda: None)

    calls = []
    expected = {'relation_backend': 'semeval_svm'}

    def extract(raw_bytes, filename, text, model_dir):
        calls.append((raw_bytes, filename, text, model_dir))
        return expected

    monkeypatch.setattr(semeval_pipeline, 'extract_pipeline', extract)
    text = 'The water is in the bottle.'
    assert pipeline.run_pipeline_nlp(text.encode(), 'upload.txt', 'txt') is expected
    assert calls == [(text.encode(), 'upload.txt', text, settings.utd_svm_model_path)]


def test_candidate_pipeline_uses_source_spans_and_unknown_confidence():
    text = 'The water is in the bottle.'
    client = SimpleNamespace(annotate=lambda t: dict(sentences=[dict(index=0,char_start=0,char_end=len(t))],
        noun_phrases=[dict(sentence_index=0,char_start=4,char_end=9),
                      dict(sentence_index=0,char_start=20,char_end=26)],entities=[]))
    result = extract_pipeline(text.encode(),'test.txt',text,'models/utd_svm',client)
    assert len(result['resolved_edges']) == 1
    edge = result['resolved_edges'][0]
    assert edge.relation_type == 'CONTENT_CONTAINER'
    assert edge.evidence == text and edge.confidence is None
    assert edge.chunk_ids == [result['chunks'][0].chunk_id]
    ResolvedEdgePayload(**edge.to_dict())
    for entity in result['resolved_entities']:
        ResolvedEntityPayload(**entity.to_dict())


@pytest.mark.asyncio
async def test_svm_edges_reach_graph_writer():
    from src.graph.graph_builder import ResolvedEdge
    edge = ResolvedEdge('COMPONENT_WHOLE','a','b',['chunk'],['semeval_svm_experimental'],None,claim_id='claim')
    session = AsyncMock()
    await GraphBuilder().write_edges(session,[edge])
    assert 'COMPONENT_WHOLE' in session.run.call_args.args[0]
    assert session.run.call_args.kwargs['edges'][0]['confidence'] is None
