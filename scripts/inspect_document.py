"""Run the production NLP pipeline and save an inspectable graph, without Neo4j.

Usage: python scripts/inspect_document.py document.pdf --output output/graph.json
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import run_pipeline_nlp
from src.graph.graph_builder import build_resolved_edges
from src.api.schemas import ResolvedEdgePayload, ResolvedEntityPayload


def inspect_document(path: Path) -> dict:
    result = run_pipeline_nlp(path.read_bytes(), path.name, path.suffix.lstrip(".").lower())
    entities = result["resolved_entities"]
    edges = (result["resolved_edges"] if result.get("relation_backend") == "semeval_svm"
             else build_resolved_edges(result["normalized"], result["id_map"]))
    names = {e.canonical_id: e.name for e in entities}
    linked = {identifier for e in edges for identifier in (e.source_id, e.target_id)}
    adjacency = {identifier: set() for identifier in names}
    for edge in edges:
        adjacency[edge.source_id].add(edge.target_id)
        adjacency[edge.target_id].add(edge.source_id)
    unseen, sizes = set(names), []
    while unseen:
        pending, component = [next(iter(unseen))], set()
        while pending:
            node = pending.pop()
            if node not in component:
                component.add(node)
                pending.extend(adjacency[node] - component)
        unseen.difference_update(component)
        sizes.append(len(component))
    for entity in entities:
        ResolvedEntityPayload(**entity.to_dict())
    for edge in edges:
        ResolvedEdgePayload(**edge.to_dict())
        assert edge.source_id in names and edge.target_id in names
    return {
        "filename": path.name,
        "doc_id": result["preprocessed"].doc_id,
        "summary": {
            "entities": len(entities), "edges": len(edges),
            "isolated_entities": len(set(names) - linked),
            "connected_entity_fraction": round(len(linked) / len(entities), 4) if entities else 0,
            "weakly_connected_components": len(sizes),
            "largest_component_entities": max(sizes, default=0),
            "relation_counts": dict(sorted(Counter(e.relation_type for e in edges).items())),
        },
        "entities": [e.to_dict() for e in entities],
        "relationships": [{**e.to_dict(), "source_name": names[e.source_id],
                           "target_name": names[e.target_id]} for e in edges],
        "isolated_entities": [e.to_dict() for e in entities if e.canonical_id not in linked],
        "chunks": [{"id": c.chunk_id, "text": c.text} for c in result["chunks"]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/graph_diagnostics.json"))
    parser.add_argument("--relation-backend", choices=["rules", "learned", "encoder", "contextual_v2", "classical_news", "semeval_svm"], default="rules")
    parser.add_argument("--model", type=Path, default=Path("models/learned_relations.joblib"))
    args = parser.parse_args()
    from src.config import get_settings
    get_settings().relation_backend = args.relation_backend
    get_settings().learned_relation_model_path = str(args.model.resolve())
    report = inspect_document(args.document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Graph and evidence saved to {args.output}")


if __name__ == "__main__":
    main()
