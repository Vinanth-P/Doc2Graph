"""Compute descriptive audit counts; never infer causal error classes from scores."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def analyze(data):
    counts, samples = Counter(), {}
    for doc in data['documents_detail']:
        gold = set(map(tuple, doc['gold']))
        pred = set(map(tuple, doc['end_to_end']['predicted']))
        oracle = set(map(tuple, doc['gold_entities']['predicted']))
        counts.update(gold=len(gold), predicted=len(pred), tp=len(gold & pred),
                      fp=len(pred - gold), fn=len(gold - pred), oracle_tp=len(gold & oracle),
                      oracle_fp=len(oracle - gold), oracle_fn=len(gold - oracle),
                      fn_recovered_by_oracle=len((gold - pred) & oracle),
                      fn_also_missed_by_oracle=len((gold - pred) - oracle),
                      e2e_tp_missing_in_oracle=len((gold & pred) - oracle))
        evidence = {tuple(item['fact']): item['text'] for item in doc.get('false_positive_evidence', [])}
        for fact in sorted(pred - gold):
            a, _, b = fact
            unmatched = int(a.startswith('unmatched:')) + int(b.startswith('unmatched:'))
            category = ['fp_aligned', 'fp_one_unmatched', 'fp_both_unmatched'][unmatched]
            counts[category] += 1
            rows = samples.setdefault(category, [])
            if len(rows) < 5:
                rows.append({'document': doc['index'], 'title': doc['title'], 'fact': fact,
                             'evidence': evidence.get(fact, ''), 'cause': 'not manually adjudicated'})
    counts['fp_any_unmatched'] = counts['fp_one_unmatched'] + counts['fp_both_unmatched']
    return dict(counts), samples


def build_audit_report(input_path=Path('output/evaluation/handoff_learned_dev.json'),
                       output_path=Path('output/evaluation/v2_error_audit.md')):
    data = json.loads(input_path.read_text(encoding='utf-8'))
    if data.get('errors'):
        raise ValueError('Input report contains pipeline errors; do not present it as complete')
    counts, samples = analyze(data)
    lines = ['# V2 descriptive error audit — corrected', '',
             f'Source: `{input_path.as_posix()}`',
             f'SHA-256: `{hashlib.sha256(input_path.read_bytes()).hexdigest()}`',
             f'Documents: {len(data["documents_detail"])}', '',
             'Counts are computed from fact sets, not hard-coded. Samples are deterministic first examples within each stratum, not a random or human-adjudicated causal audit.', '',
             '| Diagnostic | Count |', '|---|---:|']
    lines += [f'| {key} | {value} |' for key, value in counts.items()]
    percentage = 100 * counts['fp_any_unmatched'] / max(counts['fp'], 1)
    lines += ['', f'{percentage:.1f}% of scored false positives have at least one unmatched endpoint.', '',
              '## Interpretation and corrections', '',
              '- Unmatched means exact typed gold-cluster alignment failed or was ambiguous. Causes may include boundaries, types, spurious entities, missing annotations or incorrect resolution. It does not establish that the predicted relation is semantically correct.',
              '- Oracle-recovered misses identify sensitivity to supplied mentions/clusters/contexts, not NER-only causation. End-to-end and oracle true-positive sets need not be nested; counts use intersections, not subtraction of totals.',
              '- Misses remaining under oracle entities can involve candidate/schema coverage, context, thresholds, mapping, model errors or annotation issues. Their aggregate does not prove long-range reasoning is the cause.',
              '- Candidate generation caps the DISTANCE feature at four. It does not exclude pairs farther than four sentences. The previous audit stated an incorrect cutoff.',
              '- Pre-extraction merging exists in the learned/encoder branch. Its measured contribution requires traced merge examples or an ablation; code order alone does not prove how many facts were lost.',
              '- Strictly unmatched predicted pairs were labeled zero by prior robust retraining. This is a real unknown-label hazard. Its causal contribution to recall collapse is not isolated without a controlled masked-label retraining comparison.',
              '- A sigmoid/dot-product scorer is not inherently incapable of distinguishing negatives. The saved GLiREL probe contains bad predictions, but it does not show founder above 0.70 for every ordered pair. Calibration and integration must be tested independently.',
              '- Claim-status defaults and source mapping require regression tests before building dataset preparation. Automatic taxonomy is not a substitute for manual adjudication.', '',
              '## Samples requiring review', '']
    for category, rows in samples.items():
        lines += [f'### {category}', '']
        for row in rows:
            lines += [f'- Document {row["document"]}: {row["title"]}; fact `{row["fact"]}`',
                      f'  Evidence: {row["evidence"] or "Not available in this artifact."}',
                      '  Causal adjudication: pending.']
        lines.append('')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines), encoding='utf-8')
    output_path.with_suffix('.json').write_text(json.dumps({'counts': counts, 'samples': samples}, indent=2), encoding='utf-8')
    print(json.dumps(counts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('output/evaluation/handoff_learned_dev.json'))
    parser.add_argument('--output', type=Path, default=Path('output/evaluation/v2_error_audit.md'))
    args = parser.parse_args()
    build_audit_report(args.input, args.output)
