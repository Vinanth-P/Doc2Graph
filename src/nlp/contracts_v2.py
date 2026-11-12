"""Version 2 NLP and extraction data contracts.

Enforces attribute-frozen, typed records for documents, entity mentions, identity links,
relation candidates, claims with assertion attributes, and aggregated canonical claims.
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Literal, Optional, Tuple, List, Dict
from pydantic import BaseModel, ConfigDict, Field, model_validator


PolarityType = Literal["affirmed", "negated", "unknown"]
ModalityType = Literal["asserted", "planned", "conditional", "rumored", "unknown"]
LifecycleType = Literal["announced", "signed", "completed", "cancelled", "unknown"]
IdentityDecisionType = Literal["linked", "unlinked", "uncertain"]


class SourceBlock(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    block_id: str
    page_idx: int = 0
    start_char: int
    end_char: int
    raw_start: Optional[int] = None
    raw_end: Optional[int] = None
    bbox: Optional[Tuple[float, float, float, float]] = None


class DocumentRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str
    source_name: str
    extraction_version: str = "2.0"
    raw_text: str
    normalized_text: str
    content_hash: str = ""
    blocks: List[SourceBlock] = Field(default_factory=list)
    language: str = "en"
    ocr_diagnostics: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def set_content_hash(cls, values: Any) -> Any:
        if isinstance(values, dict):
            raw = values.get("raw_text", "")
            if not values.get("content_hash"):
                values["content_hash"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return values


class MentionRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    document_id: str
    text: str
    type: str
    start: int
    end: int
    source_start: Optional[int] = None
    source_end: Optional[int] = None
    page_idx: Optional[int] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    sentence_idx: Optional[int] = None
    block_id: Optional[str] = None
    detector: str = "unknown"
    score: float = Field(default=1.0, ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_offsets(self) -> MentionRecord:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"Invalid offsets: [{self.start}, {self.end})")
        return self


class IdentityLinkRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_mention_id: str
    target_mention_id: str
    decision: IdentityDecisionType
    score: float = Field(ge=0, le=1, allow_inf_nan=False)
    method: str
    evidence: str = ""


class RelationCandidateRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str
    document_id: str
    subject_mention_id: str
    object_mention_id: str
    context_ids: List[str] = Field(default_factory=list)
    context_text: str = ""
    retrieval_reason: str = "sentence_cooccurrence"
    truncated: bool = False


class RelationClaimRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    claim_id: str
    document_id: str
    subject_mention_id: str
    object_mention_id: str
    relation: str
    polarity: PolarityType = "unknown"
    modality: ModalityType = "unknown"
    lifecycle: LifecycleType = "unknown"
    time_qualifiers: Dict[str, Any] = Field(default_factory=dict)
    evidence_spans: List[Tuple[int, int]] = Field(default_factory=list)
    evidence_text: str = ""
    model_version: str = ""
    raw_score: float = 0.0
    calibrated_confidence: Optional[float] = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    abstention_reason: Optional[str] = None
    schema_version: str = "2"

    @model_validator(mode="after")
    def validate_spans(self) -> RelationClaimRecord:
        for span in self.evidence_spans:
            if span[0] < 0 or span[1] < span[0]:
                raise ValueError(f"Invalid evidence span: {span}")
        return self


class CanonicalClaimRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    canonical_claim_id: str
    canonical_subject_id: str
    canonical_object_id: str
    relation: str
    polarity: PolarityType = "unknown"
    modality: ModalityType = "unknown"
    lifecycle: LifecycleType = "unknown"
    time_qualifiers: Dict[str, Any] = Field(default_factory=dict)
    member_claim_ids: List[str] = Field(default_factory=list)
    evidence_spans: List[Tuple[int, int]] = Field(default_factory=list)
    evidence_texts: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0, le=1, allow_inf_nan=False)
    disagreements: List[Dict[str, Any]] = Field(default_factory=list)
    schema_version: str = "2"


class DocumentAnnotationPayload(BaseModel):
    """Interchange payload for annotated or predicted documents."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["2"] = "2"
    document_id: str
    text: str
    mentions: List[MentionRecord]
    claims: List[RelationClaimRecord]

    @model_validator(mode="after")
    def validate_references(self):
        mentions = {m.id: m for m in self.mentions}
        if len(mentions) != len(self.mentions):
            raise ValueError('Duplicate mention IDs')
        if len({c.claim_id for c in self.claims}) != len(self.claims):
            raise ValueError('Duplicate claim IDs')
        for mention in self.mentions:
            if mention.document_id != self.document_id:
                raise ValueError('Mention document ID mismatch')
            if mention.end > len(self.text) or self.text[mention.start:mention.end] != mention.text:
                raise ValueError('Mention text does not match its span')
        for claim in self.claims:
            if claim.document_id != self.document_id:
                raise ValueError('Claim document ID mismatch')
            if claim.subject_mention_id not in mentions or claim.object_mention_id not in mentions:
                raise ValueError('Claim references an unknown mention')
            if claim.subject_mention_id == claim.object_mention_id:
                raise ValueError('Self-referential claim')
            if any(not 0 <= a < b <= len(self.text) for a, b in claim.evidence_spans):
                raise ValueError('Evidence span outside document or empty')
        return self


DecisionType = Literal["positive", "explicit_negative", "unknown"]
ReviewStatusType = Literal["reviewed", "pending_human_review", "adjudicated"]


class ReviewedRelationAnnotation(BaseModel):
    """A human or machine-proposed relation decision for a specific mention pair."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    annotation_id: str
    subject_mention_id: str
    object_mention_id: str
    relation: str
    decision: DecisionType = "unknown"
    polarity: PolarityType = "unknown"
    modality: ModalityType = "unknown"
    lifecycle: LifecycleType = "unknown"
    time_qualifiers: Dict[str, Any] = Field(default_factory=dict)
    evidence_spans: List[Tuple[int, int]] = Field(default_factory=list)
    evidence_text: str = ""
    reviewer: str = "unassigned"
    review_status: ReviewStatusType = "pending_human_review"
    notes: Optional[str] = None

    @model_validator(mode="after")
    def validate_evidence(self) -> ReviewedRelationAnnotation:
        for span in self.evidence_spans:
            if span[0] < 0 or span[1] <= span[0]:
                raise ValueError(f"Invalid evidence span: {span}")
        if self.decision == "positive" and not self.evidence_spans:
            raise ValueError("Positive relation annotations require at least one valid evidence span")
        return self


class ReviewedDocumentPayload(BaseModel):
    """Full document payload with reviewer metadata, mentions, and relation decisions."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["2.finance.1"] = "2.finance.1"
    document_id: str
    title: str
    text: str
    doc_group: str
    license_or_rights: str = "proprietary_internal"
    reviewer: str = "unassigned"
    review_guidelines_version: str = "2.0"
    review_timestamp: Optional[str] = None
    review_status: ReviewStatusType = "pending_human_review"
    data_origin: Literal["human_reviewed_source", "machine_draft", "unverified", "test_fixture"] = "unverified"
    annotation_scope: str = "exhaustive_corporate_finance"
    mentions: List[MentionRecord]
    relation_annotations: List[ReviewedRelationAnnotation]

    @model_validator(mode="after")
    def validate_document(self) -> ReviewedDocumentPayload:
        mentions = {m.id: m for m in self.mentions}
        if len(mentions) != len(self.mentions):
            raise ValueError("Duplicate mention IDs in document")
        annotation_ids = set()
        decisions = {}

        if self.review_status in {"reviewed", "adjudicated"}:
            if not self.reviewer.strip() or self.reviewer in {"unassigned", "proposal_engine_unreviewed"}:
                raise ValueError('Reviewed document requires a named reviewer')
            if not self.review_guidelines_version.strip() or not self.review_timestamp:
                raise ValueError('Reviewed document requires guidelines version and timestamp')
            try:
                timestamp = datetime.fromisoformat(self.review_timestamp.replace('Z', '+00:00'))
                if timestamp.tzinfo is None:
                    raise ValueError('Timezone required')
            except ValueError as exc:
                raise ValueError('Review timestamp must be an ISO timestamp with timezone') from exc

        for mention in self.mentions:
            if mention.document_id != self.document_id:
                raise ValueError(f"Mention {mention.id} document_id mismatch")
            if mention.end > len(self.text) or self.text[mention.start:mention.end] != mention.text:
                raise ValueError(f"Mention {mention.id} text does not match span in document")

        for ann in self.relation_annotations:
            if ann.annotation_id in annotation_ids:
                raise ValueError(f"Duplicate annotation ID: {ann.annotation_id}")
            annotation_ids.add(ann.annotation_id)

            if ann.subject_mention_id not in mentions:
                raise ValueError(f"Annotation {ann.annotation_id} references unknown subject: {ann.subject_mention_id}")
            if ann.object_mention_id not in mentions:
                raise ValueError(f"Annotation {ann.annotation_id} references unknown object: {ann.object_mention_id}")
            if ann.subject_mention_id == ann.object_mention_id:
                raise ValueError(f"Annotation {ann.annotation_id} is self-referential")

            for span in ann.evidence_spans:
                if span[1] > len(self.text):
                    raise ValueError(f"Annotation {ann.annotation_id} evidence span extends past document text")

            pair_key = (ann.subject_mention_id, ann.relation, ann.object_mention_id)
            previous = decisions.setdefault(pair_key, [])
            all_decisions = {a.decision for a in previous} | {ann.decision}
            if {"positive", "explicit_negative"} <= all_decisions:
                raise ValueError(f"Contradictory positive and negative annotations for pair {pair_key}")
            # Event-scoped repeated claims need a richer schema. Never silently
            # choose the last annotation/status for an identical relation pair.
            if previous:
                raise ValueError(f'Duplicate relation-pair annotation requires adjudication: {pair_key}')
            previous.append(ann)

            if ann.review_status in {"reviewed", "adjudicated"} and (
                not ann.reviewer.strip() or ann.reviewer in {"unassigned", "proposal_engine_unreviewed"}
            ):
                raise ValueError('Reviewed annotation requires a named reviewer')

        return self

