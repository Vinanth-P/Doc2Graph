"""Fixed-threshold encoder diagnostic; previously observed dev, not a new holdout."""
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['USE_TF'] = '0'
os.environ['HF_HOME'] = str(ROOT / '.runtime/huggingface')
os.environ['HF_HUB_OFFLINE'] = '1'
import argparse
from collections import Counter
import hashlib
import json
from scripts.evaluate_external import reconstruct, gold_facts, graph_predictions, metrics, summarize
from src.config import get_settings
from src.pipeline import run_pipeline_nlp
from src.nlp.encoder_relations import REPOSITORY, REVISION


def main(args):
    records = json.loads(args.dataset.read_text(encoding='utf-8'))[:args.limit or None]
    get_settings().relation_backend = 'encoder'
    counts, rows, errors = Counter(), [], []
    for i, record in enumerate(records):
        text, _, _, mentions = reconstruct(record)
        gold = gold_facts(record)
        predicted, evidence = set(), {}
        try:
            result = run_pipeline_nlp(text.encode(), f'encoder-{i}.txt', 'txt')
            predicted, _, _, evidence = graph_predictions(result, mentions)
        except Exception as exc:
            errors.append({'index': i, 'error': str(exc)})
            # Do not disguise infrastructure failures as model quality scores.
            if i == 0:
                raise
        counts.update(metrics(predicted, gold))
        rows.append({'index': i, 'title': record['title'], 'gold': sorted(gold), 'predicted': sorted(predicted),
                     'false_positive_evidence': [{'fact': p, 'text': evidence.get(p, '')} for p in sorted(predicted - gold)]})
        if (i + 1) % 10 == 0:
            print(json.dumps({'processed': i + 1, 'errors': len(errors), **summarize(counts)}), flush=True)
    report = {'repository': REPOSITORY, 'revision': REVISION, 'threshold': .75,
              'dataset': str(args.dataset), 'dataset_sha256': hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
              'documents': len(records), 'metrics': summarize(counts), 'errors': errors, 'documents_detail': rows}
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['metrics']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('data/external/redocred/dev_revised.json'))
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--output', type=Path, default=Path('output/evaluation/encoder_dev.json'))
    main(parser.parse_args())
