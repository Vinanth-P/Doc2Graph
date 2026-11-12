# Copy/paste handoff prompt

You are reviewing and implementing the Doc2Graph extraction redesign in this repository. Read `MODEL_REDESIGN_HANDOFF.md` completely, then verify its claims against the named source files and evaluation artifacts. It is a proposed plan, not an instruction to accept its architecture or conclusions without evidence.

The objective is accurate, evidence-backed document-to-graph extraction beyond the supplied dummy finance PDF. Use learned local entity/relation models, not a generative LLM doing runtime extraction. Current models are poor; no candidate is approved to replace the default. Do not “fix” this by adding document-specific rules, forcing edges onto isolated nodes, lowering evaluation standards, or hiding failures.

Start with:

1. Inventory the working tree and applicable repository instructions. Preserve unrelated edits, old artifacts and reports; do not reset the repository or database.
2. Read `output/evaluation/REPORT.md`, `LEARNED_REPORT.md`, `FOLLOWUP_REPORT.md`, model documentation, training metadata and relevant pipeline code.
3. Critique the handoff: distinguish verified facts from hypotheses, correct errors, and identify missing data or product decisions. Specifically audit mention alignment, relation direction, premature entity merging, incomplete negative labels, checkpoint preprocessing, and claim-status loss.
4. Reproduce the baseline and classify representative failures before selecting a model or rewriting components. Run pipeline evaluations sequentially because of the shared lock.
5. Implement the staged plan, revising it where evidence warrants. Preserve ingestion/storage infrastructure where suitable; introduce versioned mention/claim contracts and an isolated new backend.
6. Train and evaluate using reviewed labels, group-disjoint splits, separate calibration, and a truly unseen final test. Existing Re-DocRED dev/test have already been inspected; do not present them as fresh holdouts. Synthetic or LLM-produced labels are not independent gold truth.
7. Report matched before/after precision, recall, F1, status/evidence accuracy, per-label support, errors and resource costs. Unit tests alone do not establish extraction quality. Keep the default unchanged unless the predeclared acceptance criteria pass.

Your first response should provide a concise verification report and a concrete implementation sequence. Then begin the authorized local, reversible work. If human-reviewed data or additional compute is required, explain exactly what is missing and complete independent engineering work first. Do not invent successful training, benchmark results, annotations or model capabilities.

At each milestone provide changed files, tests/commands, measured results, failure examples and the remaining blocker. Be candid if a model is worse. The goal is a reliable extractor, not a cosmetic graph improvement.
