"""Propose candidate annotations on raw finance documents for human review.

Generates drafts complying with ReviewedDocumentPayload and schema 2.finance.1.
All outputs are explicitly marked 'pending_human_review'. Machine-generated proposals
must never be presented as independent gold truth without human adjudication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from src.ingestion.preprocessor import get_nlp
from src.nlp.ner import NERRunner
from src.nlp.contracts_v2 import ReviewedDocumentPayload, MentionRecord, ReviewedRelationAnnotation


FINANCE_TRIGGER_WORDS = {
    "ACQUIRED": ["acquire", "acquired", "acquisition", "buy", "bought", "purchase", "purchased", "takeover", "merged with"],
    "INVESTED_IN": ["invest", "invested", "investment", "backed", "funded", "financing", "seed round"],
    "ADVISED": ["advised", "advising", "advisor to", "adviser to", "represented"],
    "FINANCED": ["financed", "loan", "borrowed", "credit facility", "debt facility"],
    "FOUNDED": ["founded", "co-founded", "cofounded", "established", "launched"],
    "EMPLOYED_BY": ["ceo", "cto", "cfo", "executive", "works for", "appointed", "leads", "serves as"],
    "MEMBER_OF": ["member of", "joined", "board member", "trustee"],
    "OWNS": ["owns", "owned", "holds a stake", "majority stake", "subsidiary"],
    "SUBSIDIARY_OF": ["subsidiary of", "operating unit of", "division of"],
}


def propose_annotations_for_text(
    doc_id: str,
    title: str,
    text: str,
    doc_group: str,
    ontology: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate candidate mentions and relation annotations for human review."""
    nlp = get_nlp()
    doc = nlp(text)
    ner = NERRunner()
    raw_entities = ner.run(doc, "chunk_0", text)

    mentions = []
    for i, ent in enumerate(raw_entities):
        mentions.append(MentionRecord(
            id=f"m{i}",
            document_id=doc_id,
            text=ent.text,
            type=ent.type,
            start=ent.start_char,
            end=ent.end_char,
            detector="heuristic_ner_runner",
            score=0.9,
        ))

    mentions_by_id = {m.id: m for m in mentions}
    relations_cfg = ontology.get("relations", {})
    annotations = []
    ann_idx = 0

    # Propose relations within sentence co-occurrence containing trigger words
    for sent in doc.sents:
        sent_start = sent.start_char
        sent_end = sent.end_char
        sent_text_lower = sent.text.lower()

        sent_mentions = [
            m for m in mentions if sent_start <= m.start and m.end <= sent_end
        ]

        for sub in sent_mentions:
            for obj in sent_mentions:
                if sub.id == obj.id:
                    continue

                for rel, triggers in FINANCE_TRIGGER_WORDS.items():
                    if rel not in relations_cfg:
                        continue
                    allowed_pairs = relations_cfg[rel].get("entity_type_pairs", [])
                    if [sub.type, obj.type] not in allowed_pairs and (sub.type, obj.type) not in allowed_pairs:
                        continue

                    # Check for trigger match
                    if any(t in sent_text_lower for t in triggers):
                        ann_id = f"ann_{ann_idx}"
                        ann_idx += 1

                        # Heuristic modality / lifecycle suggestion
                        modality = "planned" if any(w in sent_text_lower for w in ["plans to", "proposed", "intends", "bid"]) else "asserted"
                        lifecycle = "announced" if "announced" in sent_text_lower else "signed" if "signed" in sent_text_lower else "completed"
                        polarity = "negated" if any(w in sent_text_lower for w in ["not", "denied", "cancelled", "terminated", "refuted"]) else "affirmed"

                        annotations.append(ReviewedRelationAnnotation(
                            annotation_id=ann_id,
                            subject_mention_id=sub.id,
                            object_mention_id=obj.id,
                            relation=rel,
                            decision="positive",
                            polarity=polarity,
                            modality=modality,
                            lifecycle=lifecycle,
                            evidence_spans=[(sent_start, sent_end)],
                            evidence_text=sent.text,
                            reviewer="proposal_engine_unreviewed",
                            review_status="pending_human_review",
                            notes="Machine proposal; human review required.",
                        ))

    payload = ReviewedDocumentPayload(
        schema_version="2.finance.1",
        document_id=doc_id,
        title=title,
        text=text,
        doc_group=doc_group,
        license_or_rights="pilot_internal",
        reviewer="unassigned",
        review_status="pending_human_review",
        data_origin="machine_draft",
        review_guidelines_version="2.0",
        annotation_scope="proposed_candidate_screening",
        mentions=mentions,
        relation_annotations=annotations,
    )
    return payload.model_dump()


def propose_for_file(
    file_path: Path,
    output_path: Path,
    ontology_path: Path,
    doc_group: str | None = None,
) -> None:
    text = file_path.read_text(encoding="utf-8")
    ontology = yaml.safe_load(ontology_path.read_text(encoding="utf-8"))
    group = doc_group or file_path.stem
    doc_id = f"doc_{file_path.stem}"

    payload = propose_annotations_for_text(
        doc_id=doc_id,
        title=file_path.name,
        text=text,
        doc_group=group,
        ontology=ontology,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Generated draft annotations with {len(payload['relation_annotations'])} proposed relations in {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to input text file")
    parser.add_argument("--output", type=Path, required=True, help="Path to output JSON proposal")
    parser.add_argument("--ontology", type=Path, default=Path("config/ontology_v2.yaml"))
    parser.add_argument("--group", type=str, default=None, help="Document group key")
    args = parser.parse_args()

    propose_for_file(args.input, args.output, args.ontology, args.group)
