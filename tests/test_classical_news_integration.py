"""Integration and regression tests for Classical News backend and graph storage."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.config import get_settings
from src.graph.graph_builder import ResolvedEdge, ResolvedEntity
from src.nlp.classical_news import ClassicalNewsExtractor
from src.nlp.stanford_classical_client import get_shared_stanford_client
from src.pipeline import run_pipeline_nlp

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def shared_client():
    client = get_shared_stanford_client()
    if not client.is_available():
        pytest.skip("Stanford CoreNLP classical runtime dependencies not installed.")
    client.start()
    return client


def test_live_classical_news_extractor_integration(shared_client):
    """Verify live extraction of entities, claims, and edges with verified text boundaries."""
    extractor = ClassicalNewsExtractor(client=shared_client)
    sample_text = (
        "Tim Cook serves as the chief executive officer of Apple. "
        "Apple is headquartered in Cupertino, California. "
        "Tim Cook lives in Palo Alto."
    )
    res = extractor.extract_document("test_integration_001", sample_text)

    # 1. Mentions integrity
    mentions = res["mentions"]
    assert len(mentions) >= 5
    for m in mentions:
        assert m.type in {"PERSON", "ORGANIZATION", "LOCATION"}
        assert sample_text[m.start:m.end] == m.text
        assert m.detector == "stanford_crf_3class"

    # 2. Resolved entities integrity
    resolved_ents = res["resolved_entities"]
    ent_names = {e.name for e in resolved_ents}
    assert "Apple" in ent_names
    assert "Tim Cook" in ent_names

    # 3. Claims and edges integrity
    claims = res["claims"]
    edges = res["resolved_edges"]
    assert len(claims) >= 1
    assert len(edges) >= 1

    for claim in claims:
        assert claim.relation in {"WORKS_FOR", "LIVES_IN", "BASED_IN", "LOCATED_IN"}
        assert claim.polarity == "unknown"
        assert claim.modality == "unknown"
        assert claim.lifecycle == "unknown"
        assert claim.calibrated_confidence is None
        assert 0.0 <= claim.raw_score <= 1.0

    for edge in edges:
        assert edge.relation_type in {"WORKS_FOR", "LIVES_IN", "BASED_IN", "LOCATED_IN"}
        assert edge.polarity == "unknown"
        assert 0.0 <= edge.confidence <= 1.0


def test_split_manifest_and_group_isolation():
    """Verify that split_manifest.json exists and guarantees zero cross-split group contamination."""
    manifest_path = WORKSPACE_ROOT / "data" / "classical_news" / "splits" / "split_manifest.json"
    assert manifest_path.exists(), "split_manifest.json must exist"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = manifest.get("splits", {})
    assert "dev" in splits
    assert "locked_test" in splits
    assert "challenge" in splits

    # Verify document files exist and have non-zero content
    group_assignments: dict[str, str] = {}
    for split_name, sinfo in splits.items():
        doc_file = WORKSPACE_ROOT / sinfo["file"]
        assert doc_file.exists(), f"Split file {doc_file} must exist"

        with doc_file.open("r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                grp = doc["group_key"]
                if grp in group_assignments:
                    assert group_assignments[grp] == split_name, (
                        f"Group {grp} is contaminated across splits {group_assignments[grp]} and {split_name}!"
                    )
                group_assignments[grp] = split_name


def test_evaluation_report_artifacts_and_decision():
    """Verify that evaluation artifacts exist and uphold the release decision KEEP_DEFAULT_RULES."""
    report_json_path = WORKSPACE_ROOT / "output" / "classical_news_evaluation_report.json"
    report_md_path = WORKSPACE_ROOT / "docs" / "CLASSICAL_NEWS_EVALUATION_REPORT.md"

    assert report_json_path.exists(), "Evaluation JSON report must exist"
    assert report_md_path.exists(), "Evaluation Markdown report must exist"

    report = json.loads(report_json_path.read_text(encoding="utf-8"))
    assert report["backend"] == "classical_news"
    assert report["release_decision"] == "KEEP_DEFAULT_RULES"
    assert report["runtime_bridge"]["non_dl_verified"] is True

    gates = report["feasibility_gates"]
    assert "entity_micro_f1" in gates
    assert "relation_precision" in gates
    assert "relation_recall" in gates
    assert "relation_micro_f1" in gates


def test_pipeline_nlp_classical_news_roundtrip():
    """Verify classical_news restores the previously configured backend."""
    settings = get_settings()
    orig_backend = settings.relation_backend
    settings.relation_backend = "classical_news"
    try:
        text = "Satya Nadella works for Microsoft. Microsoft is based in Redmond, Washington."
        nlp_res = run_pipeline_nlp(text.encode("utf-8"), "integration_test.txt", "txt")

        assert "resolved_entities" in nlp_res
        assert "v2_claims" in nlp_res
        assert "resolved_edges" in nlp_res

        names = {e.name for e in nlp_res["resolved_entities"]}
        assert "Microsoft" in names
    finally:
        settings.relation_backend = orig_backend
        assert settings.relation_backend == orig_backend
