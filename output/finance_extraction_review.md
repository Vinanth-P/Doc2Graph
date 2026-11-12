# Finance PDF extraction review

Document: `Doc2Graph_Corporate_Finance_MA_Dummy.pdf` (3 pages).

The same PDF was run through the production NLP pipeline before and after the changes. Both runs stopped before database persistence.

| Metric | Before | After |
| --- | ---: | ---: |
| Entities | 73 | 33 |
| Relationships | 26 | 33 |
| Isolated entities (no semantic edge) | 44 | 2 |
| Entities with at least one edge | 39.7% | 93.9% |

These are document-specific connectivity measurements, not precision/recall or a general model accuracy score. The final graph has five weakly connected components; its largest component contains 27 entities. Not every entity belongs to one connected graph.

## Changes

- Preserve PDF page boundaries, remove repeated page furniture, separate numbered headings from prose, and delimit actual tables before excluding them from narrative extraction. Table labels previously contaminated company names.
- Recover names from document-local company definitions, quoted aliases, executive roles, product descriptions and unambiguous surnames. Preserve original mention offsets and source sentences.
- Extract shared subjects, coordinated predicates, percentage ownership, advisers, reporting lines, financing, executive roles, subsidiary structures, location modifiers, product ownership and assessments.
- Distinguish planned acquisitions, borrowing, integration and subsidiary arrangements from completed/current relationships.
- Reject negated relations, possessive-head substitutions (shareholders are not the company), canonical self-links, and redundant generic edges when a specific edge already captures the same evidence.
- Keep graph storage validation in sync with the additional relation types.

No external dataset was downloaded and no model weights were retrained. The fixes target extraction, normalization and entity resolution failures around the existing spaCy model.

## Verification

`python -m pytest -q`: **166 passed, 1 skipped**. The suite includes 28 new finance regression cases, including alternative names, negation, relationship direction, planned versus completed actions, ambiguous surnames, coordinated company names, PDF layout, and the supplied PDF through the production NLP entry point. The skipped test requires a running Neo4j instance.

The final graph passes the production entity/relationship payload validators; all edge endpoints exist and every edge retains evidence and chunk IDs. Live Neo4j persistence was not exercised or updated.

## Remaining limitations

- `Grant Thornton Bharat` remains isolated and the base NER still labels it PERSON. Its due-diligence sentence does not explicitly name the client/target for that clause; this run does not invent that relationship.
- `Competition Commission Of India` remains isolated. The text describes a required approval, not an approval already granted. Representing this fully needs transaction/event and condition modeling.
- The two smaller connected groups concern the committee/chair and Pocket Labs/director. Cross-sentence links to the main transaction remain incomplete.
- Table-only facts are not converted to relations by this narrative pipeline. Amounts/dates are not graph nodes under the existing type schema.
- Pronoun resolution and some entity types remain heuristic. Future tense is distinguished for the new planned transaction categories; this is not a complete temporal/event model.

## Inspect and refresh

The before/after graphs are saved beside this report as `finance_graph_before.json` and `finance_graph_after.json`. Each relationship includes its source and target names and evidence.

To inspect another document without Neo4j:

```powershell
python scripts/inspect_document.py Doc2Graph_Corporate_Finance_MA_Dummy.pdf --output output/finance_graph_after.json
```

Restart the backend (or let its development reloader pick up the changes), then re-upload the PDF to regenerate its stored graph with the revised extractor. Existing database results do not change merely by editing the extraction code.
