# Doc2Graph v2 Extraction Error Audit (Development Split)

Audit Date: 11 November 2026
Dataset: `data/external/redocred/dev_revised.json` (500 documents, 613 mapped gold facts)
Evaluator: `scripts/evaluate_learned_relations.py`

## 1. Executive Summary & Aggregate Error Distribution

| Metric Category | Count | Percentage of Class |
|---|---:|---:|
| **Total Ground Truth Relations** | **613** | 100.0% |
| Correct Extractions (True Positives) | 47 | 7.7% |
| Missed Relations (False Negatives) | 566 | 92.3% |
| — *Missed due to NER / Alignment (recovered with Gold Entities)* | 144 | 25.4% of FNs |
| — *Missed by Relation Model (even with Gold Entities)* | 422 | 74.6% of FNs |
| **Total Predicted Relations** | **927** | 100.0% |
| True Positives | 47 | 5.1% |
| Total False Positives | 880 | 94.9% |
| — *At least one unmatched/misaligned entity endpoint* | 731 | **83.1% of FPs** |
| &nbsp;&nbsp;&nbsp;&nbsp;• Both endpoints unmatched | 255 | 29.0% of FPs |
| &nbsp;&nbsp;&nbsp;&nbsp;• Target endpoint unmatched | 327 | 37.2% of FPs |
| &nbsp;&nbsp;&nbsp;&nbsp;• Source endpoint unmatched | 149 | 16.9% of FPs |
| — *Both endpoints correctly aligned (Pure Semantic False Positives)* | 149 | **16.9% of FPs** |

### Key Takeaways from the Aggregate Numbers
1. **The Strict Metric Severely Penalizes NER Boundary & Linking Mismatches**: 83.1% of all scored false positives are not 'hallucinated relationships' between recognized entities; they are candidate pairs where spaCy NER or within-doc resolution produced an ID that could not be mapped to the dataset's gold cluster.
2. **Relation Classification is Still Fundamentally Weak**: Even when gold entity mentions and gold cluster IDs are supplied directly to the learned model (`gold_entities` mode), precision is only 24.55% and recall is 31.16% (F1 = 27.46%). It makes 587 false positives on gold entities alone. Therefore, fixing NER alone does not solve extraction.
3. **Cross-Sentence & Distant Mention Bottleneck**: 74.6% of gold facts are missed even when gold entities are provided. The linear TF-IDF classifier on local 2-window masked contexts cannot capture multi-hop, long-range, or complex narrative syntax.

## 2. Taxonomy of Root Causes

1. **NER Boundary Inexactness & Subtoken Truncation**: Re-DocRED uses exact token spans. spaCy `en_core_web_sm` often adds leading determiners ('the BBC' vs 'BBC') or splits compound organization names differently. Any 1-character boundary discrepancy causes strict alignment failure.
2. **Premature Within-Doc Resolution Merging**: EntityResolver merges entities before relation extraction using initials and token overlap heuristics. When two different entities share initials or a surname, they merge into one root, converting any potential relation between them into a self-loop (`source == target`), which is filtered out.
3. **Context Truncation and Fixed Sliding Windows**: `build_candidates` selects only up to 2 sentence contexts within distance <= 4. Facts spanning further across the document are never proposed.
4. **Masked Token Collocation Artifacts**: Masking entities as `SUBJECT_ORG` and `OBJECT_ORG` removes all lexical clues about the entities themselves, leaving the TF-IDF model to over-index on generic co-occurrence words (e.g. 'and', 'with', 'party', 'national').
5. **Complete Erasure of Assertion Modality & Polarity**: The pipeline has no representation for planned, conditional, or historical events, nor negative assertions. All extracted edges are flattened into unconditional present assertions.

## 3. Stratified Error Audit: Representative Samples

### Category A: False Positives with Unmatched Endpoints (NER / Linking Failure)
- **Doc 6 ('Soccer Academy')** — Relation: `OWNS`
  - Subject: `1` | Target: `unmatched:08309b391cd65ba6efc2f7b6d70c9470d8e5d6e4fa00335299e292ac24897856:0:0:18` (Unmatched: target)
  - Evidence: *"The Soccer Academy is a children ’s television show produced by Capistrano Films . The Soccer Academy " qualifies as children - friendly programming from an instructional and educational perspective ,..."*
  - *Root Cause*: NER detected mention with boundary mismatch or entity resolver failed to link it to gold cluster.
- **Doc 6 ('Soccer Academy')** — Relation: `OWNS`
  - Subject: `1` | Target: `unmatched:08309b391cd65ba6efc2f7b6d70c9470d8e5d6e4fa00335299e292ac24897856:5:142:144` (Unmatched: target)
  - Evidence: *"The Soccer Academy is a children ’s television show produced by Capistrano Films . The Soccer Academy " qualifies as children - friendly programming from an instructional and educational perspective ,..."*
  - *Root Cause*: NER detected mention with boundary mismatch or entity resolver failed to link it to gold cluster.
- **Doc 6 ('Soccer Academy')** — Relation: `OWNS`
  - Subject: `unmatched:08309b391cd65ba6efc2f7b6d70c9470d8e5d6e4fa00335299e292ac24897856:0:0:18` | Target: `1` (Unmatched: source)
  - Evidence: *"The Soccer Academy is a children ’s television show produced by Capistrano Films . The Soccer Academy " qualifies as children - friendly programming from an instructional and educational perspective ,..."*
  - *Root Cause*: NER detected mention with boundary mismatch or entity resolver failed to link it to gold cluster.
- **Doc 6 ('Soccer Academy')** — Relation: `SUBSIDIARY_OF`
  - Subject: `unmatched:08309b391cd65ba6efc2f7b6d70c9470d8e5d6e4fa00335299e292ac24897856:0:0:18` | Target: `1` (Unmatched: source)
  - Evidence: *"The Soccer Academy is a children ’s television show produced by Capistrano Films . The Soccer Academy " qualifies as children - friendly programming from an instructional and educational perspective ,..."*
  - *Root Cause*: NER detected mention with boundary mismatch or entity resolver failed to link it to gold cluster.
- **Doc 7 ('ROKETSAN')** — Relation: `OWNS`
  - Subject: `0` | Target: `unmatched:5906a58605653369650f13639f9022d443054b48ea8c8de212834ad757fc5811:5:53:59` (Unmatched: target)
  - Evidence: *"Roketsan is best known for its vast range of unguided rockets as well as laser and infrared guided missiles such as Cirit and UMTAS . The company also produces subsystems for Stinger and Rapier missil..."*
  - *Root Cause*: NER detected mention with boundary mismatch or entity resolver failed to link it to gold cluster.

### Category B: Pure Semantic False Positives (Both Endpoints Gold, Relation False)
- **Doc 4 ('ELAM (Latin American School of Medicine) Cuba')** — Relation: `SUBSIDIARY_OF`
  - Subject Cluster: `4` | Object Cluster: `1`
  - Evidence: *"Escuela Latinoamericana de Medicina ( ELAM ) , formerly Escuela Latinoamericana de Ciencias Médicas ( in Spanish ; in English : Latin American School of Medicine ( LASM ) , formerly Latin American Sch..."*
  - *Root Cause*: TF-IDF classifier triggered on co-occurring context words without verifying syntactic or semantic relation direction.
- **Doc 17 ('Carey Business School')** — Relation: `OWNS`
  - Subject Cluster: `0` | Object Cluster: `7`
  - Evidence: *"The Carey Business School is accredited by the Association to Advance Collegiate Schools of Business ( AACSB ) . James Carey ( 1751 - 1834 ) , the namesake of the Carey Business School , is a relative..."*
  - *Root Cause*: TF-IDF classifier triggered on co-occurring context words without verifying syntactic or semantic relation direction.
- **Doc 17 ('Carey Business School')** — Relation: `SUBSIDIARY_OF`
  - Subject Cluster: `0` | Object Cluster: `7`
  - Evidence: *"The Carey Business School is accredited by the Association to Advance Collegiate Schools of Business ( AACSB ) . James Carey ( 1751 - 1834 ) , the namesake of the Carey Business School , is a relative..."*
  - *Root Cause*: TF-IDF classifier triggered on co-occurring context words without verifying syntactic or semantic relation direction.
- **Doc 17 ('Carey Business School')** — Relation: `OWNS`
  - Subject Cluster: `7` | Object Cluster: `0`
  - Evidence: *"The Carey Business School is accredited by the Association to Advance Collegiate Schools of Business ( AACSB ) . James Carey ( 1751 - 1834 ) , the namesake of the Carey Business School , is a relative..."*
  - *Root Cause*: TF-IDF classifier triggered on co-occurring context words without verifying syntactic or semantic relation direction.
- **Doc 17 ('Carey Business School')** — Relation: `FOUNDED`
  - Subject Cluster: `8` | Object Cluster: `0`
  - Evidence: *"The Carey Business School is accredited by the Association to Advance Collegiate Schools of Business ( AACSB ) . James Carey ( 1751 - 1834 ) , the namesake of the Carey Business School , is a relative..."*
  - *Root Cause*: TF-IDF classifier triggered on co-occurring context words without verifying syntactic or semantic relation direction.
- **Doc 19 ('Railroad Revival Tour')** — Relation: `MEMBER_OF`
  - Subject Cluster: `13` | Object Cluster: `12`
  - Evidence: *"The inaugural tour in 2011 included performances by Mumford & Sons , Edward Sharpe and the Magnetic Zeros , and Old Crow Medicine Show . In 2011 , Railroad Revival Tour bands Mumford & Sons , Edward S..."*
  - *Root Cause*: TF-IDF classifier triggered on co-occurring context words without verifying syntactic or semantic relation direction.
- **Doc 19 ('Railroad Revival Tour')** — Relation: `MEMBER_OF`
  - Subject Cluster: `27` | Object Cluster: `26`
  - Evidence: *"The 2012 tour was slated to include performances by Willie Nelson , Band of Horses , Jamey Johnson , and John Reilly and Friends ...."*
  - *Root Cause*: TF-IDF classifier triggered on co-occurring context words without verifying syntactic or semantic relation direction.

### Category C: False Negatives Missed by Relation Model (Gold Entities Supplied)
- **Doc 3 ('Ramey Idriss')** — Missing Gold Fact: `(0, MEMBER_OF, 9)`
  - *Root Cause*: Classifier probability fell below threshold or mention pair was outside distance <= 4 candidate budget.
- **Doc 5 ('Song of Freedom')** — Missing Gold Fact: `(16, MEMBER_OF, 14)`
  - *Root Cause*: Classifier probability fell below threshold or mention pair was outside distance <= 4 candidate budget.
- **Doc 6 ('Soccer Academy')** — Missing Gold Fact: `(1, OWNS, 0)`
  - *Root Cause*: Classifier probability fell below threshold or mention pair was outside distance <= 4 candidate budget.
- **Doc 8 ('The Sacramento Bee')** — Missing Gold Fact: `(0, SUBSIDIARY_OF, 11)`
  - *Root Cause*: Classifier probability fell below threshold or mention pair was outside distance <= 4 candidate budget.
- **Doc 8 ('The Sacramento Bee')** — Missing Gold Fact: `(11, OWNS, 0)`
  - *Root Cause*: Classifier probability fell below threshold or mention pair was outside distance <= 4 candidate budget.
- **Doc 8 ('The Sacramento Bee')** — Missing Gold Fact: `(11, OWNS, 17)`
  - *Root Cause*: Classifier probability fell below threshold or mention pair was outside distance <= 4 candidate budget.
- **Doc 8 ('The Sacramento Bee')** — Missing Gold Fact: `(11, OWNS, 18)`
  - *Root Cause*: Classifier probability fell below threshold or mention pair was outside distance <= 4 candidate budget.

### Category D: True Positive Success Cases
- **Doc 25 ('Toufique Imrose Khalidi')** — Valid Fact: `(0, EMPLOYED_BY, 4)`
  - Evidence: *"..."*
  - *Observation*: Close-distance mention pair in a single declarative sentence with clear pattern words.
- **Doc 43 ('Cry for Help')** — Valid Fact: `(4, MEMBER_OF, 18)`
  - Evidence: *"..."*
  - *Observation*: Close-distance mention pair in a single declarative sentence with clear pattern words.
- **Doc 46 ('List of Square Enix video games')** — Valid Fact: `(11, OWNS, 13)`
  - Evidence: *"..."*
  - *Observation*: Close-distance mention pair in a single declarative sentence with clear pattern words.
- **Doc 47 ('Velocifero')** — Valid Fact: `(15, MEMBER_OF, 2)`
  - Evidence: *"..."*
  - *Observation*: Close-distance mention pair in a single declarative sentence with clear pattern words.
- **Doc 93 ('Il regalo più grande')** — Valid Fact: `(15, MEMBER_OF, 14)`
  - Evidence: *"..."*
  - *Observation*: Close-distance mention pair in a single declarative sentence with clear pattern words.

## 4. Priority Interventions for v2 Redesign

1. **Data Contracts v2 (`src/nlp/contracts_v2.py`)**: Introduce immutable `Mention`, `Document`, `RelationCandidate`, `RelationClaim`, and `CanonicalClaim` contracts preserving exact character offsets, document hashes, assertion attributes (polarity, modality, lifecycle), and evidence spans.
2. **Non-Destructive Entity Resolution**: Do NOT collapse mention identities before relation extraction. Extract relations on mention pairs first (or preserve soft clusters), and aggregate claims onto canonical entities subsequently.
3. **Negative Training Label Correction**: In `train_robust_relations.py`, stop treating unmatched NER spans as hard negative relation examples. Mask unaligned mentions from the loss rather than training false zeros on true relations.
4. **Ontology & Direction Standardization (`config/ontology_v2.yaml`)**: Explicitly declare domain, endpoint types, direction, and assertion attribute schemas for finance and general relations.