# Doc2Graph: Project Evaluation Summary

## 1. One-minute explanation

Doc2Graph converts written documents into a **knowledge graph**.

For example, from:

> Alice joined Acme Corporation. She works in London.

the system tries to produce:

- entities: `Alice` (PERSON), `Acme Corporation` (ORG), `London` (GPE)
- relationships: `Alice MEMBER_OF Acme Corporation`
- relationships: `Alice LOCATED_IN London`

The result is stored in Neo4j, where entities become nodes and relationships
become labelled edges. A FastAPI backend provides upload, search, graph
statistics, entity details, shortest-path queries, and visualization data for a
D3.js web dashboard.

The project deliberately uses **classical NLP and classical machine learning**.
It does not call an LLM, does not require a GPU, and is designed to be
interpretable: most decisions can be traced to a regex, dictionary entry,
dependency pattern, similarity feature, or configured relation rule.

## 2. Problem statement

Normal documents are unstructured text. Humans can read:

> Microsoft acquired Activision Blizzard.

but a computer needs explicit structure before it can answer graph questions.
Doc2Graph solves this by extracting:

1. **Entities**: important objects such as people, organizations, places,
   products, dates, money values, and email addresses.
2. **Relations**: meaningful connections between entities, such as
   `ACQUIRED`, `FOUNDED`, `EMPLOYED_BY`, or `LOCATED_IN`.
3. **Evidence**: the source sentence and document chunk supporting each fact.

This makes it possible to search connections rather than only search words.

## 3. Complete system flow

```text
PDF / DOCX / TXT
        |
        v
Upload validation and language check
        |
        v
Text extraction, cleaning, normalization, sentence splitting
        |
        v
Named Entity Recognition (NER)
        |
        v
Rule-based coreference resolution
        |
        v
Dependency-pattern relation extraction
        |
        v
Predicate normalization and relation whitelist
        |
        v
Entity resolution / duplicate-name merging
        |
        v
Neo4j graph construction with provenance
        |
        +--> FastAPI search and graph APIs
        |
        +--> D3.js interactive visualization
```

The pipeline is intentionally sequential. Coreference resolution happens before
relation extraction, so a sentence containing “She joined Google” can use the
previously detected person instead of creating an unknown pronoun entity.

## 4. Stage-by-stage working

### 4.1 Upload validation and ingestion

The API accepts PDF, DOCX, and TXT files. It:

- reads uploads in bounded chunks and enforces the maximum size;
- identifies the file type using magic bytes and DOCX ZIP contents;
- rejects corrupt or unsupported files;
- extracts PDF text with `pdfplumber`;
- uses OCR as a fallback for scanned PDF pages;
- extracts paragraphs from DOCX;
- decodes TXT as UTF-8 or Windows-1252;
- rejects non-English documents.

This prevents binary data or damaged documents from silently entering the NLP
pipeline.

### 4.2 Preprocessing

The text is normalized using Unicode NFKC normalization and whitespace cleanup.
spaCy performs sentence splitting. Each document receives a SHA-256 identifier,
and each sentence receives a deterministic chunk ID:

```text
<document_hash>:<sentence_index>
```

Deterministic IDs are useful because reprocessing the same input produces
stable references and avoids unnecessary duplicate evidence.

### 4.3 Named Entity Recognition

NER answers: **“What real-world object is mentioned here, and what type is it?”**

Doc2Graph combines three layers:

1. **Regex rules** detect structured patterns such as dates, money, and emails.
2. **Gazetteers** are project dictionaries containing known people,
   organizations, and locations. Exact and fuzzy matching are supported.
3. **spaCy `en_core_web_sm`** supplies general-purpose NER predictions.

The span merger resolves overlaps using source priority. This is useful because
a domain dictionary may know a company name more reliably than a general model.
Every entity also stores its character offsets, source layer, sentence, and
chunk ID.

### 4.4 Coreference resolution

Coreference means determining what a reference such as “he”, “it”, or “the
company” refers to.

Doc2Graph uses rules rather than a large neural coreference model:

- personal pronouns point to the nearest recent PERSON;
- organizational pronouns point to the nearest recent ORG;
- definite noun phrases such as “the company” use an organization heuristic;
- aliases are compared using containment and Jaro-Winkler similarity.

Example:

```text
Elon Musk founded SpaceX. He leads the company.
```

The resolver links “He” to `Elon Musk` and “the company” to `SpaceX`.
This approach is fast and explainable, but it can be less reliable for complex
conversation, long-distance references, or ambiguous pronouns.

### 4.5 Relation extraction

Relation extraction answers: **“How are two entities connected?”**

spaCy supplies a dependency parse, which represents grammatical relationships
between words. The extractor looks for:

- subject-verb-object patterns;
- appositive patterns;
- prepositional patterns;
- nominal patterns involving roles such as founder, CEO, or director.

For example:

```text
Google hired Alice.
```

The dependency structure identifies `Google` as subject, `hired` as predicate,
and `Alice` as object. The normalizer can reverse the direction to represent
the canonical fact:

```text
Alice EMPLOYED_BY Google
```

The extractor also records negation. Negated statements are not written as
asserted graph facts, because “Alice did not join Google” should not create a
positive `MEMBER_OF` edge.

### 4.6 Predicate normalization

Different words can express the same meaning. “bought”, “purchased”, and
“acquired” can all represent `ACQUIRED`.

The file [`config/relations.yaml`](C:/Users/asus/OneDrive/Desktop/Doc2Graph/config/relations.yaml)
defines the canonical relation vocabulary, accepted verb lemmas, required
prepositions, direction swaps, and valid entity-type pairs.

Examples:

- `join` -> `MEMBER_OF`
- `hire` -> `EMPLOYED_BY` with direction reversal
- `acquire` -> `ACQUIRED`
- `work for` -> `EMPLOYED_BY`
- `found` -> `FOUNDED` only when the object is an organization

The whitelist reduces false positives by rejecting unsupported predicates and
invalid combinations, such as a person being treated as an acquired company.

### 4.7 Relation confidence scoring

The project includes a Logistic Regression scorer for relation confidence.
Features include dependency distance, entity types, predicate information,
negation-related information, and other structured indicators.

Logistic Regression estimates the probability of a class using a weighted
combination of features. It is suitable here because it is lightweight and
interpretable: feature weights can be inspected and the model runs on a CPU.

### 4.8 Entity resolution

The same real entity may appear under several names:

```text
International Business Machines
IBM
IBM Corp.
```

Entity resolution decides whether two mentions refer to the same entity.
Doc2Graph first uses **blocking**, which creates a smaller set of plausible
candidates using shared tokens and initials. It then compares candidates using:

- Jaro-Winkler similarity;
- TF-IDF character n-gram cosine similarity;
- token overlap;
- initials and abbreviation features;
- a Logistic Regression decision model.

Blocking improves efficiency because every entity does not need to be compared
with every other entity. The resolver works within a document and also checks
existing Neo4j entities for cross-document merging.

### 4.9 Graph construction and provenance

In the graph:

- entities are nodes;
- relation types are directed edges;
- documents and sentence chunks are provenance nodes;
- `MENTIONED_IN` links entities to the chunks where they occur;
- relationship `chunk_ids` identify the evidence sentences;
- confidence, extractor type, aliases, and evidence are retained.

Neo4j `MERGE` makes writes idempotent: retrying the same document does not
normally create duplicate nodes or duplicate relationship evidence. Constraints,
indexes, and a full-text entity index support reliable storage and search.

### 4.10 API and interface

FastAPI provides:

- `POST /api/upload` for document ingestion;
- `GET /api/jobs/{job_id}` for background-job status;
- graph statistics and visualization endpoints;
- entity detail and shortest-path endpoints;
- full-text entity search.

SQLite stores job status. A background worker runs the CPU-heavy pipeline, and a
file lock prevents conflicting pipeline executions. The D3.js frontend renders
the returned nodes and links as an interactive force-directed graph.

## 5. Theory for evaluation

### Knowledge graph

A knowledge graph is a graph-shaped representation of knowledge:

```text
node --relationship--> node
```

In Doc2Graph, an example is:

```text
Alice --EMPLOYED_BY--> Acme Corporation
```

Graphs are useful for relationship questions, traversal, shortest paths,
centrality, and visual exploration.

### NLP

Natural Language Processing is the area of computing concerned with language.
This project uses NLP to transform sentences into structured entities and
relations.

### Tokenization and sentence splitting

Tokenization divides text into words and punctuation. Sentence splitting
separates the document into manageable units. Consistent offsets matter because
later stages must map a parsed token back to the original evidence.

### Named Entity Recognition

NER is a sequence-labelling task. It assigns a category to a text span:

```text
[Elon Musk] PERSON founded [SpaceX] ORG.
```

Important metrics are:

- **Precision** = correct predictions / all predictions.
- **Recall** = correct predictions / all gold entities.
- **F1** = harmonic mean of precision and recall.

F1 is useful when both missed entities and incorrect entities matter.

### Dependency parsing

Dependency parsing creates a tree showing grammatical relations. It helps
distinguish the subject, predicate, object, and prepositional context instead
of relying only on word order.

### Coreference

Coreference resolution links different expressions that refer to the same
thing. It is essential for multi-sentence understanding and for avoiding
orphaned “he”, “she”, or “it” nodes.

### TF-IDF and cosine similarity

TF-IDF represents a name using weighted word or character fragments. Common
fragments receive less weight, while distinctive fragments receive more weight.
Cosine similarity measures the angle between two vectors; a value near 1 means
the representations are similar.

### Jaro-Winkler similarity

Jaro-Winkler is a string similarity metric that is especially useful for names
and short strings. It rewards matching characters and common prefixes, making
it useful for spelling variations and aliases.

### Logistic Regression

Logistic Regression is a supervised classification algorithm. It combines
features with learned weights and converts the result into a class probability.
Here it is used for relation confidence and entity-pair resolution.

### Evaluation metrics

For predicted facts:

- **True positive (TP):** predicted correctly.
- **False positive (FP):** predicted but not present in the gold answer.
- **False negative (FN):** present in the gold answer but missed.
- **Precision:** `TP / (TP + FP)`.
- **Recall:** `TP / (TP + FN)`.
- **F1:** `2 * precision * recall / (precision + recall)`.

The benchmark also reports exact relation documents, meaning the complete set of
predicted relations exactly matches the expected set for that document.

## 6. Current evaluation results

The following values were produced by running:

```powershell
python scripts/benchmark.py --eval data/eval/
python scripts/benchmark.py --eval data/eval/ --train
```

| Component | Result | Interpretation |
|---|---:|---|
| NER precision | 0.966 | Most predicted entity spans are correct |
| NER recall | 1.000 | All gold entity spans were found in this set |
| NER F1 | 0.982 | Strong combined NER result |
| Full-text relation precision | 0.857 | Some predicted relations are incorrect |
| Full-text relation recall | 0.857 | Some expected relations are missed |
| Full-text relation F1 | 0.857 | Good regression performance on the included examples |
| Exact relation documents | 13/15 | Complete relation output matched in 13 examples |
| Relation scorer 5-fold CV F1 | 0.920 +/- 0.098 | Model performance on 20 labelled feature rows |
| Entity resolver 5-fold CV F1 | 0.933 +/- 0.133 | Name-matching performance on the resolution gold set |

These are development-set measurements, not a guarantee of general-domain
accuracy. The datasets are small and hand-authored, so a stronger evaluation
would use more documents, independent test data, multiple domains, and human
annotation agreement.

## 7. Strengths and limitations

### Strengths

- modular stages that can be tested independently;
- CPU-only and no paid LLM API dependency;
- interpretable rules and feature-based models;
- supports PDF, DOCX, TXT, and scanned PDFs through OCR;
- relation whitelist reduces unsupported graph facts;
- provenance preserves source evidence;
- deterministic document/chunk IDs;
- idempotent graph writes;
- API, CLI, dashboard, and automated tests are included.

### Limitations

- English only;
- rule-based coreference can fail on ambiguous or distant references;
- dependency patterns cover supported sentence structures rather than all
  natural language;
- gazetteer quality depends on the domain dictionaries;
- classical NER is less flexible than modern transformer models;
- small evaluation sets limit claims about real-world accuracy;
- no vector or semantic embedding search is used;
- OCR quality depends on scan quality and OCR configuration.

## 8. Suggested viva/demo explanation

1. Upload a short English document.
2. Explain that validation checks size, file type, extractability, and language.
3. Show the extracted entities and their types.
4. Explain one relation using its source sentence and dependency pattern.
5. Show how an alias or pronoun is resolved.
6. Open the Neo4j-backed graph visualization.
7. Search for an entity and inspect its neighbors.
8. Demonstrate a shortest path or graph statistics endpoint.
9. Run the benchmark and explain precision, recall, and F1.
10. State clearly that the reported scores are on small development datasets.

## 9. Final conclusion

Doc2Graph is an end-to-end document intelligence system. Its main contribution
is not only extracting isolated names, but connecting entity recognition,
coreference, relation extraction, normalization, duplicate resolution,
provenance-aware graph storage, search, and visualization into one reproducible
pipeline. The design prioritizes transparency and CPU-friendly operation, while
its main future improvement area is broader evaluation and more robust language
understanding.
