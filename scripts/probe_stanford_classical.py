"""Probe and smoke test script for Stanford Classical CoreNLP.

Runs 20 varied sentences covering supported relations, no-relation pairs, reversed mentions,
multiple entities, negation, historical claims, and Unicode characters/emoji.
Saves native model outputs to output/classical_news_smoke_20.json.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nlp.stanford_classical_client import StanfordClassicalClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("probe_stanford_classical")

SMOKE_SENTENCES = [
    {"id": "smoke-01", "text": "Barack Obama was born in Hawaii."},
    {"id": "smoke-02", "text": "Satya Nadella is the CEO of Microsoft."},
    {"id": "smoke-03", "text": "Google is headquartered in Mountain View, California."},
    {"id": "smoke-04", "text": "Steve Jobs lived in Palo Alto."},
    {"id": "smoke-05", "text": "Apple was founded in Cupertino."},
    {"id": "smoke-06", "text": "France is located in Europe."},
    {"id": "smoke-07", "text": "Albert Einstein lived in Princeton, New Jersey."},
    {"id": "smoke-08", "text": "Bill Gates does not work for Apple."},
    {"id": "smoke-09", "text": "Amazon did not acquire Netflix."},
    {"id": "smoke-10", "text": "Elon Musk formerly worked for PayPal."},
    {"id": "smoke-11", "text": "Berlin is in Germany, and Tokyo is in Japan."},
    {"id": "smoke-12", "text": "Sundar Pichai works at Google, which is based in Mountain View."},
    {"id": "smoke-13", "text": "Microsoft in Redmond employs Satya Nadella."},
    {"id": "smoke-14", "text": "California contains San Francisco."},
    {"id": "smoke-15", "text": "Tim Cook had lunch with Satya Nadella."},
    {"id": "smoke-16", "text": "Microsoft and Apple compete in the technology market."},
    {"id": "smoke-17", "text": "Yoshua Bengio is a professor at Université de Montréal in Canada."},
    {"id": "smoke-18", "text": "Angela Merkel lived in Berlin 🇩🇪 for many years."},
    {"id": "smoke-19", "text": "OpenAI was based in San Francisco, California."},
    {"id": "smoke-20", "text": "The Eiffel Tower is in Paris."},
]


def run_probe(output_path: Path) -> dict:
    logger.info("Initializing Stanford Classical Client...")
    client = StanfordClassicalClient()

    if not client.is_available():
        logger.error("Stanford Classical runtime dependencies not installed. Run scripts/setup_stanford_classical.py first.")
        sys.exit(1)

    results = {
        "metadata": {
            "annotators": "tokenize,ssplit,pos,lemma,ner,parse,relation",
            "pos_model": "english-left3words-distsim.tagger",
            "ner_model": "english.all.3class.distsim.crf.ser.gz",
            "parse_model": "englishPCFG.ser.gz",
            "relation_model": "roth_relation_model_pipelineNER.ser",
            "neural_components": "NONE",
            "total_sentences": len(SMOKE_SENTENCES),
        },
        "items": [],
    }

    with client:
        results["metadata"]["bridge_handshake"] = client.ready_metadata

        for idx, item in enumerate(SMOKE_SENTENCES):
            sid = item["id"]
            text = item["text"]
            logger.info("Processing [%d/%d] %s: %s", idx + 1, len(SMOKE_SENTENCES), sid, text)

            resp = client.annotate(text, req_id=sid)

            # Assert offset alignments
            for ent in resp.get("entities", []):
                expected = ent.get("mapped_text")
                actual = text[ent["char_start"]:ent["char_end"]]
                assert actual == expected, (
                    f"Span mismatch in {sid} for entity {ent}: expected {expected!r}, got {actual!r}"
                )

            for rel in resp.get("relations", []):
                for arg_key in ("arg0", "arg1"):
                    arg = rel.get(arg_key, {})
                    expected = arg.get("mapped_text")
                    actual = text[arg["char_start"]:arg["char_end"]]
                    assert actual == expected, (
                        f"Span mismatch in {sid} for relation {rel['label']} {arg_key}: expected {expected!r}, got {actual!r}"
                    )

            results["items"].append({
                "id": sid,
                "text": text,
                "sentence_count": len(resp.get("sentences", [])),
                "entities": resp.get("entities", []),
                "relations": resp.get("relations", []),
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Probe completed. Saved 20 smoke outputs to %s", output_path)
    return results


def main():
    out_file = Path(__file__).resolve().parent.parent / "output" / "classical_news_smoke_20.json"
    run_probe(out_file)


if __name__ == "__main__":
    main()
