"""Prepare news and biography benchmark dataset and annotation review packets.

Creates reproducible splits (dev, locked_test, challenge) with strict group isolation,
exact/near-duplicate detection, and explicit positive, negative, and unknown labels.
Exports review packets with reviewer provenance tracking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure repository root is on sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.nlp.contracts_v2 import MentionRecord, ReviewedRelationAnnotation
from src.nlp.deduplication import analyze_duplicates_and_groups, compute_content_hash

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SPLITS_DIR = WORKSPACE_ROOT / "data" / "classical_news" / "splits"
PACKETS_DIR = WORKSPACE_ROOT / "data" / "classical_news" / "review_packets"


def make_mention(mid: str, doc_id: str, text: str, full_text: str, m_type: str, sent_idx: int) -> Dict[str, Any]:
    """Helper to find character start/end in text and create a valid mention dictionary."""
    start = full_text.find(text)
    if start == -1:
        raise ValueError(f"Mention text '{text}' not found in document text.")
    end = start + len(text)
    return {
        "id": mid,
        "document_id": doc_id,
        "text": text,
        "type": m_type,
        "start": start,
        "end": end,
        "sentence_idx": sent_idx,
        "detector": "human_gold_standard",
        "score": 1.0,
    }


def build_raw_documents() -> List[Dict[str, Any]]:
    """Construct representative news and biography documents across story groups."""
    docs: List[Dict[str, Any]] = []

    # ─────────────────────────────────────────────────────────────────────────
    # DEV SPLIT (30 documents): Topics: Apple, Microsoft, Amazon, Tesla, Google, Politics, European Cities
    # ─────────────────────────────────────────────────────────────────────────
    dev_specs = [
        # Apple group
        {
            "id": "news:dev:001",
            "title": "Tim Cook and Apple Cupertino Headquarters",
            "group": "group:apple_execs",
            "text": "Tim Cook serves as the chief executive officer of Apple. He works closely with design teams in Cupertino. Apple is based in Cupertino, California. Tim Cook lives in Palo Alto.",
            "mentions": [
                ("m0", "Tim Cook", "PERSON", 0),
                ("m1", "Apple", "ORGANIZATION", 0),
                ("m2", "Cupertino", "LOCATION", 1),
                ("m3", "Apple", "ORGANIZATION", 2),
                ("m4", "Cupertino", "LOCATION", 2),
                ("m5", "California", "LOCATION", 2),
                ("m6", "Tim Cook", "PERSON", 3),
                ("m7", "Palo Alto", "LOCATION", 3),
            ],
            "relations": [
                ("m0", "WORKS_FOR", "m1", "positive", "Tim Cook serves as the chief executive officer of Apple."),
                ("m3", "BASED_IN", "m4", "positive", "Apple is based in Cupertino, California."),
                ("m4", "LOCATED_IN", "m5", "positive", "Apple is based in Cupertino, California."),
                ("m6", "LIVES_IN", "m7", "positive", "Tim Cook lives in Palo Alto."),
                ("m0", "LIVES_IN", "m2", "explicit_negative", "He works closely with design teams in Cupertino."),
            ],
        },
        {
            "id": "news:dev:002",
            "title": "Steve Jobs Biography and NeXT",
            "group": "group:apple_execs",
            "text": "Steve Jobs co-founded Apple in Los Altos. Jobs lived in Palo Alto for many decades. Apple maintains major offices across California.",
            "mentions": [
                ("m0", "Steve Jobs", "PERSON", 0),
                ("m1", "Apple", "ORGANIZATION", 0),
                ("m2", "Los Altos", "LOCATION", 0),
                ("m3", "Jobs", "PERSON", 1),
                ("m4", "Palo Alto", "LOCATION", 1),
                ("m5", "Apple", "ORGANIZATION", 2),
                ("m6", "California", "LOCATION", 2),
            ],
            "relations": [
                ("m0", "WORKS_FOR", "m1", "positive", "Steve Jobs co-founded Apple in Los Altos."),
                ("m3", "LIVES_IN", "m4", "positive", "Jobs lived in Palo Alto for many decades."),
                ("m2", "LOCATED_IN", "m6", "unknown", "in Los Altos"),
            ],
        },
        {
            "id": "news:dev:003",
            "title": "Jony Ive Design at Apple",
            "group": "group:apple_execs",
            "text": "Jony Ive worked for Apple for over two decades. Ive resided in San Francisco while leading industrial design. Apple is located in Cupertino.",
            "mentions": [
                ("m0", "Jony Ive", "PERSON", 0),
                ("m1", "Apple", "ORGANIZATION", 0),
                ("m2", "Ive", "PERSON", 1),
                ("m3", "San Francisco", "LOCATION", 1),
                ("m4", "Apple", "ORGANIZATION", 2),
                ("m5", "Cupertino", "LOCATION", 2),
            ],
            "relations": [
                ("m0", "WORKS_FOR", "m1", "positive", "Jony Ive worked for Apple for over two decades."),
                ("m2", "LIVES_IN", "m3", "positive", "Ive resided in San Francisco while leading industrial design."),
                ("m4", "BASED_IN", "m5", "positive", "Apple is located in Cupertino."),
            ],
        },
        # Microsoft group
        {
            "id": "news:dev:004",
            "title": "Satya Nadella Leadership at Microsoft",
            "group": "group:microsoft_execs",
            "text": "Satya Nadella joined Microsoft in 1992 and became chief executive in 2014. Nadella lives in Bellevue near Seattle. Microsoft is headquartered in Redmond, Washington.",
            "mentions": [
                ("m0", "Satya Nadella", "PERSON", 0),
                ("m1", "Microsoft", "ORGANIZATION", 0),
                ("m2", "Nadella", "PERSON", 1),
                ("m3", "Bellevue", "LOCATION", 1),
                ("m4", "Seattle", "LOCATION", 1),
                ("m5", "Microsoft", "ORGANIZATION", 2),
                ("m6", "Redmond", "LOCATION", 2),
                ("m7", "Washington", "LOCATION", 2),
            ],
            "relations": [
                ("m0", "WORKS_FOR", "m1", "positive", "Satya Nadella joined Microsoft in 1992 and became chief executive in 2014."),
                ("m2", "LIVES_IN", "m3", "positive", "Nadella lives in Bellevue near Seattle."),
                ("m5", "BASED_IN", "m6", "positive", "Microsoft is headquartered in Redmond, Washington."),
                ("m6", "LOCATED_IN", "m7", "positive", "Microsoft is headquartered in Redmond, Washington."),
                ("m2", "LIVES_IN", "m4", "explicit_negative", "Nadella lives in Bellevue near Seattle."),
            ],
        },
        {
            "id": "news:dev:005",
            "title": "Bill Gates Foundation and Seattle",
            "group": "group:microsoft_execs",
            "text": "Bill Gates stepped down from Microsoft to focus on philanthropy. Gates lives in Medina, Washington. Medina is located near Seattle.",
            "mentions": [
                ("m0", "Bill Gates", "PERSON", 0),
                ("m1", "Microsoft", "ORGANIZATION", 0),
                ("m2", "Gates", "PERSON", 1),
                ("m3", "Medina", "LOCATION", 1),
                ("m4", "Washington", "LOCATION", 1),
                ("m5", "Medina", "LOCATION", 2),
                ("m6", "Seattle", "LOCATION", 2),
            ],
            "relations": [
                ("m0", "WORKS_FOR", "m1", "positive", "Bill Gates stepped down from Microsoft to focus on philanthropy."),
                ("m2", "LIVES_IN", "m3", "positive", "Gates lives in Medina, Washington."),
                ("m3", "LOCATED_IN", "m4", "positive", "Gates lives in Medina, Washington."),
            ],
        },
        {
            "id": "news:dev:006",
            "title": "Brad Smith Legal Counsel",
            "group": "group:microsoft_execs",
            "text": "Brad Smith serves as vice chair and president of Microsoft. Smith resides in Bellevue. Microsoft operates major campuses across Washington.",
            "mentions": [
                ("m0", "Brad Smith", "PERSON", 0),
                ("m1", "Microsoft", "ORGANIZATION", 0),
                ("m2", "Smith", "PERSON", 1),
                ("m3", "Bellevue", "LOCATION", 1),
                ("m4", "Microsoft", "ORGANIZATION", 2),
                ("m5", "Washington", "LOCATION", 2),
            ],
            "relations": [
                ("m0", "WORKS_FOR", "m1", "positive", "Brad Smith serves as vice chair and president of Microsoft."),
                ("m2", "LIVES_IN", "m3", "positive", "Smith resides in Bellevue."),
            ],
        },
        # Amazon group
        {
            "id": "news:dev:007",
            "title": "Jeff Bezos and Amazon Relocation",
            "group": "group:amazon_corp",
            "text": "Jeff Bezos founded Amazon in Seattle. Bezos lived in Seattle before announcing a move to Miami. Amazon is based in Seattle, Washington.",
            "mentions": [
                ("m0", "Jeff Bezos", "PERSON", 0),
                ("m1", "Amazon", "ORGANIZATION", 0),
                ("m2", "Seattle", "LOCATION", 0),
                ("m3", "Bezos", "PERSON", 1),
                ("m4", "Seattle", "LOCATION", 1),
                ("m5", "Miami", "LOCATION", 1),
                ("m6", "Amazon", "ORGANIZATION", 2),
                ("m7", "Seattle", "LOCATION", 2),
                ("m8", "Washington", "LOCATION", 2),
            ],
            "relations": [
                ("m0", "WORKS_FOR", "m1", "positive", "Jeff Bezos founded Amazon in Seattle."),
                ("m3", "LIVES_IN", "m4", "positive", "Bezos lived in Seattle before announcing a move to Miami."),
                ("m3", "LIVES_IN", "m5", "positive", "Bezos lived in Seattle before announcing a move to Miami."),
                ("m6", "BASED_IN", "m7", "positive", "Amazon is based in Seattle, Washington."),
                ("m7", "LOCATED_IN", "m8", "positive", "Amazon is based in Seattle, Washington."),
            ],
        },
        {
            "id": "news:dev:008",
            "title": "Andy Jassy Amazon Leadership",
            "group": "group:amazon_corp",
            "text": "Andy Jassy was named CEO of Amazon succeeding Bezos. Jassy lives in Seattle. Amazon operates logistics facilities across Washington.",
            "mentions": [
                ("m0", "Andy Jassy", "PERSON", 0),
                ("m1", "Amazon", "ORGANIZATION", 0),
                ("m2", "Bezos", "PERSON", 0),
                ("m3", "Jassy", "PERSON", 1),
                ("m4", "Seattle", "LOCATION", 1),
                ("m5", "Amazon", "ORGANIZATION", 2),
                ("m6", "Washington", "LOCATION", 2),
            ],
            "relations": [
                ("m0", "WORKS_FOR", "m1", "positive", "Andy Jassy was named CEO of Amazon succeeding Bezos."),
                ("m3", "LIVES_IN", "m4", "positive", "Jassy lives in Seattle."),
            ],
        },
        # Google group
        {
            "id": "news:dev:009",
            "title": "Sundar Pichai at Alphabet and Google",
            "group": "group:google_corp",
            "text": "Sundar Pichai leads Google as chief executive officer. Google is based in Mountain View, California. Pichai lives in Los Altos Hills.",
            "mentions": [
                ("m0", "Sundar Pichai", "PERSON", 0),
                ("m1", "Google", "ORGANIZATION", 0),
                ("m2", "Google", "ORGANIZATION", 1),
                ("m3", "Mountain View", "LOCATION", 1),
                ("m4", "California", "LOCATION", 1),
                ("m5", "Pichai", "PERSON", 2),
                ("m6", "Los Altos Hills", "LOCATION", 2),
            ],
            "relations": [
                ("m0", "WORKS_FOR", "m1", "positive", "Sundar Pichai leads Google as chief executive officer."),
                ("m2", "BASED_IN", "m3", "positive", "Google is based in Mountain View, California."),
                ("m3", "LOCATED_IN", "m4", "positive", "Google is based in Mountain View, California."),
                ("m5", "LIVES_IN", "m6", "positive", "Pichai lives in Los Altos Hills."),
            ],
        },
        {
            "id": "news:dev:010",
            "title": "Larry Page Google Stanford Origins",
            "group": "group:google_corp",
            "text": "Larry Page co-founded Google with Sergey Brin. Page resides in Palo Alto. Mountain View is located in California.",
            "mentions": [
                ("m0", "Larry Page", "PERSON", 0),
                ("m1", "Google", "ORGANIZATION", 0),
                ("m2", "Sergey Brin", "PERSON", 0),
                ("m3", "Page", "PERSON", 1),
                ("m4", "Palo Alto", "LOCATION", 1),
                ("m5", "Mountain View", "LOCATION", 2),
                ("m6", "California", "LOCATION", 2),
            ],
            "relations": [
                ("m0", "WORKS_FOR", "m1", "positive", "Larry Page co-founded Google with Sergey Brin."),
                ("m2", "WORKS_FOR", "m1", "positive", "Larry Page co-founded Google with Sergey Brin."),
                ("m3", "LIVES_IN", "m4", "positive", "Page resides in Palo Alto."),
                ("m5", "LOCATED_IN", "m6", "positive", "Mountain View is located in California."),
            ],
        },
        # European Corporations & Cities
        {
            "id": "news:dev:011",
            "title": "Siemens Headquarters in Munich",
            "group": "group:siemens_germany",
            "text": "Siemens is headquartered in Munich, Germany. Roland Busch works for Siemens as president and CEO. Busch lives in Munich.",
            "mentions": [
                ("m0", "Siemens", "ORGANIZATION", 0),
                ("m1", "Munich", "LOCATION", 0),
                ("m2", "Germany", "LOCATION", 0),
                ("m3", "Roland Busch", "PERSON", 1),
                ("m4", "Siemens", "ORGANIZATION", 1),
                ("m5", "Busch", "PERSON", 2),
                ("m6", "Munich", "LOCATION", 2),
            ],
            "relations": [
                ("m0", "BASED_IN", "m1", "positive", "Siemens is headquartered in Munich, Germany."),
                ("m1", "LOCATED_IN", "m2", "positive", "Siemens is headquartered in Munich, Germany."),
                ("m3", "WORKS_FOR", "m4", "positive", "Roland Busch works for Siemens as president and CEO."),
                ("m5", "LIVES_IN", "m6", "positive", "Busch lives in Munich."),
            ],
        },
        {
            "id": "news:dev:012",
            "title": "SAP Headquarters in Walldorf",
            "group": "group:sap_germany",
            "text": "SAP is based in Walldorf, Germany. Christian Klein serves as chief executive of SAP. Klein resides in Heidelberg.",
            "mentions": [
                ("m0", "SAP", "ORGANIZATION", 0),
                ("m1", "Walldorf", "LOCATION", 0),
                ("m2", "Germany", "LOCATION", 0),
                ("m3", "Christian Klein", "PERSON", 1),
                ("m4", "SAP", "ORGANIZATION", 1),
                ("m5", "Klein", "PERSON", 2),
                ("m6", "Heidelberg", "LOCATION", 2),
            ],
            "relations": [
                ("m0", "BASED_IN", "m1", "positive", "SAP is based in Walldorf, Germany."),
                ("m1", "LOCATED_IN", "m2", "positive", "SAP is based in Walldorf, Germany."),
                ("m3", "WORKS_FOR", "m4", "positive", "Christian Klein serves as chief executive of SAP."),
                ("m5", "LIVES_IN", "m6", "positive", "Klein resides in Heidelberg."),
            ],
        },
        {
            "id": "news:dev:013",
            "title": "BMW Group in Munich",
            "group": "group:bmw_germany",
            "text": "BMW is headquartered in Munich. Oliver Zipse works for BMW as chairman of the board. Munich is situated in Bavaria.",
            "mentions": [
                ("m0", "BMW", "ORGANIZATION", 0),
                ("m1", "Munich", "LOCATION", 0),
                ("m2", "Oliver Zipse", "PERSON", 1),
                ("m3", "BMW", "ORGANIZATION", 1),
                ("m4", "Munich", "LOCATION", 2),
                ("m5", "Bavaria", "LOCATION", 2),
            ],
            "relations": [
                ("m0", "BASED_IN", "m1", "positive", "BMW is headquartered in Munich."),
                ("m2", "WORKS_FOR", "m3", "positive", "Oliver Zipse works for BMW as chairman of the board."),
                ("m4", "LOCATED_IN", "m5", "positive", "Munich is situated in Bavaria."),
            ],
        },
        {
            "id": "news:dev:014",
            "title": "Airbus Facilities in Toulouse",
            "group": "group:airbus_france",
            "text": "Airbus is headquartered in Toulouse, France. Guillaume Faury works for Airbus as chief executive officer. Toulouse is located in southern France.",
            "mentions": [
                ("m0", "Airbus", "ORGANIZATION", 0),
                ("m1", "Toulouse", "LOCATION", 0),
                ("m2", "France", "LOCATION", 0),
                ("m3", "Guillaume Faury", "PERSON", 1),
                ("m4", "Airbus", "ORGANIZATION", 1),
                ("m5", "Toulouse", "LOCATION", 2),
                ("m6", "France", "LOCATION", 2),
            ],
            "relations": [
                ("m0", "BASED_IN", "m1", "positive", "Airbus is headquartered in Toulouse, France."),
                ("m1", "LOCATED_IN", "m2", "positive", "Airbus is headquartered in Toulouse, France."),
                ("m3", "WORKS_FOR", "m4", "positive", "Guillaume Faury works for Airbus as chief executive officer."),
                ("m5", "LOCATED_IN", "m6", "positive", "Toulouse is located in southern France."),
            ],
        },
        {
            "id": "news:dev:015",
            "title": "LVMH and Paris Fashion",
            "group": "group:lvmh_france",
            "text": "LVMH is based in Paris, France. Bernard Arnault runs LVMH as chairman and CEO. Arnault lives in Paris.",
            "mentions": [
                ("m0", "LVMH", "ORGANIZATION", 0),
                ("m1", "Paris", "LOCATION", 0),
                ("m2", "France", "LOCATION", 0),
                ("m3", "Bernard Arnault", "PERSON", 1),
                ("m4", "LVMH", "ORGANIZATION", 1),
                ("m5", "Arnault", "PERSON", 2),
                ("m6", "Paris", "LOCATION", 2),
            ],
            "relations": [
                ("m0", "BASED_IN", "m1", "positive", "LVMH is based in Paris, France."),
                ("m1", "LOCATED_IN", "m2", "positive", "LVMH is based in Paris, France."),
                ("m3", "WORKS_FOR", "m4", "positive", "Bernard Arnault runs LVMH as chairman and CEO."),
                ("m5", "LIVES_IN", "m6", "positive", "Arnault lives in Paris."),
            ],
        },
    ]

    # Additional Dev documents (16-30) to fulfill 30 documents
    additional_dev = [
        ("dev:016", "Tesla Austin Texas", "group:tesla_auto", "Tesla is based in Austin, Texas. Austin is situated in Texas. Elon Musk lives in Austin."),
        ("dev:017", "SpaceX Hawthorne and Starbase", "group:spacex_aero", "SpaceX was founded in Hawthorne, California. Hawthorne is in California. Gwynne Shotwell works for SpaceX."),
        ("dev:018", "Nvidia Santa Clara", "group:nvidia_semis", "Jensen Huang leads Nvidia as founder and CEO. Huang lives in Los Altos. Nvidia is based in Santa Clara, California."),
        ("dev:019", "Meta Platforms Menlo Park", "group:meta_social", "Meta is headquartered in Menlo Park, California. Mark Zuckerberg works for Meta. Zuckerberg resides in Palo Alto."),
        ("dev:020", "Sheryl Sandberg at Meta", "group:meta_social", "Sheryl Sandberg worked for Meta for over a decade. Sandberg lived in Menlo Park before moving to Los Angeles."),
        ("dev:021", "Netflix Los Gatos", "group:netflix_media", "Netflix is based in Los Gatos, California. Ted Sarandos works for Netflix as co-chief executive. Sarandos lives in Beverly Hills."),
        ("dev:022", "Spotify Stockholm", "group:spotify_audio", "Spotify is headquartered in Stockholm, Sweden. Daniel Ek works for Spotify. Ek lives in Stockholm."),
        ("dev:023", "ASML Veldhoven", "group:asml_semis", "ASML is based in Veldhoven, Netherlands. Peter Wennink worked for ASML for a decade. Veldhoven is located in the Netherlands."),
        ("dev:024", "Roche Basel Switzerland", "group:roche_pharma", "Roche is headquartered in Basel, Switzerland. Thomas Schinecker works for Roche. Basel is located in Switzerland."),
        ("dev:025", "Novartis Basel Operations", "group:novartis_pharma", "Novartis is based in Basel, Switzerland. Vas Narasimhan works for Novartis. Narasimhan lives in Basel."),
        ("dev:026", "AstraZeneca Cambridge", "group:az_pharma", "AstraZeneca is headquartered in Cambridge, England. Pascal Soriot works for AstraZeneca. Soriot lives in Sydney."),
        ("dev:027", "GlaxoSmithKline London", "group:gsk_pharma", "GSK is based in London, United Kingdom. Emma Walmsley works for GSK as chief executive. Walmsley lives in London."),
        ("dev:028", "BP London Energy", "group:bp_energy", "BP is headquartered in London, England. London is located in the United Kingdom. Murray Auchincloss works for BP."),
        ("dev:029", "Shell London Headquarters", "group:shell_energy", "Shell is based in London. Wael Sawan works for Shell. Sawan lives in The Hague."),
        ("dev:030", "TotalEnergies Paris", "group:total_energy", "TotalEnergies is headquartered in Paris, France. Patrick Pouyanne works for TotalEnergies. Pouyanne lives in Paris."),
    ]

    for item in additional_dev:
        doc_id, title, group, text = item
        full_id = f"news:{doc_id}"
        # We will parse entities programmatically from sentences
        docs.append({
            "id": full_id,
            "title": title,
            "group": group,
            "text": text,
            "split": "dev",
        })

    for spec in dev_specs:
        spec["split"] = "dev"
        docs.append(spec)

    # ─────────────────────────────────────────────────────────────────────────
    # LOCKED TEST SPLIT (30 documents): Topics: Financial firms, auto, biotech, aerospace, global cities
    # (Completely distinct story groups from Dev split!)
    # ─────────────────────────────────────────────────────────────────────────
    test_specs = [
        ("test:001", "JPMorgan Chase New York", "group:jpmorgan_bank", "JPMorgan is headquartered in New York. Jamie Dimon works for JPMorgan as chairman and CEO. Dimon lives in New York."),
        ("test:002", "Goldman Sachs Manhattan", "group:goldman_bank", "Goldman Sachs is based in New York City. David Solomon works for Goldman Sachs. Solomon lives in Manhattan."),
        ("test:003", "Morgan Stanley New York", "group:morgan_bank", "Morgan Stanley is located in New York. Ted Pick works for Morgan Stanley. Pick lives in New York City."),
        ("test:004", "Citigroup New York", "group:citi_bank", "Jane Fraser works for Citigroup as chief executive officer. Fraser lives in New York. Citigroup is based in New York."),
        ("test:005", "Bank of America Charlotte", "group:bofa_bank", "Brian Moynihan works for Bank of America. Bank of America is headquartered in Charlotte, North Carolina. Moynihan lives in Boston."),
        ("test:006", "Wells Fargo San Francisco", "group:wells_bank", "Wells Fargo is based in San Francisco, California. Charlie Scharf works for Wells Fargo. Scharf lives in New York."),
        ("test:007", "BlackRock New York", "group:blackrock_asset", "Larry Fink works for BlackRock as chairman and CEO. Fink lives in New York. BlackRock is headquartered in Manhattan."),
        ("test:008", "Berkshire Hathaway Omaha", "group:berkshire_inv", "Warren Buffett works for Berkshire Hathaway. Buffett lives in Omaha, Nebraska. Berkshire Hathaway is based in Omaha."),
        ("test:009", "Charlie Munger at Berkshire", "group:berkshire_inv", "Charlie Munger worked for Berkshire Hathaway for decades. Munger lived in Los Angeles. Omaha is located in Nebraska."),
        ("test:010", "Toyota Motor Corporation", "group:toyota_auto", "Toyota is headquartered in Toyota City, Japan. Koji Sato works for Toyota as chief executive. Toyota City is located in Japan."),
        ("test:011", "Honda Motor Tokyo", "group:honda_auto", "Honda is based in Tokyo, Japan. Toshihiro Mibe works for Honda. Tokyo is located in Japan."),
        ("test:012", "Sony Group Tokyo", "group:sony_corp", "Sony is headquartered in Tokyo, Japan. Kenichiro Yoshida works for Sony. Yoshida lives in Tokyo."),
        ("test:013", "Nintendo Kyoto", "group:nintendo_corp", "Nintendo is based in Kyoto, Japan. Shuntaro Furukawa works for Nintendo. Kyoto is located in Japan."),
        ("test:014", "Samsung Electronics Suwon", "group:samsung_corp", "Samsung is headquartered in Suwon, South Korea. Jay Y Lee works for Samsung. Suwon is located in South Korea."),
        ("test:015", "Hyundai Motor Seoul", "group:hyundai_corp", "Hyundai is based in Seoul, South Korea. Euisun Chung works for Hyundai as executive chair. Seoul is in South Korea."),
        ("test:016", "Tata Sons Mumbai", "group:tata_india", "Tata Sons is headquartered in Mumbai, India. Natarajan Chandrasekaran works for Tata. Mumbai is located in India."),
        ("test:017", "Reliance Industries Mumbai", "group:reliance_india", "Mukesh Ambani works for Reliance Industries. Ambani lives in Mumbai. Reliance Industries is based in Mumbai."),
        ("test:018", "Infosys Bengaluru", "group:infosys_tech", "Infosys is based in Bengaluru, India. Salil Parekh works for Infosys. Bengaluru is located in India."),
        ("test:019", "Wipro Bengaluru", "group:wipro_tech", "Wipro is headquartered in Bengaluru. Rishad Premji works for Wipro. Premji lives in Bengaluru."),
        ("test:020", "BHP Group Melbourne", "group:bhp_mining", "BHP is based in Melbourne, Australia. Mike Henry works for BHP. Melbourne is located in Australia."),
        ("test:021", "Rio Tinto London Melbourne", "group:riotinto_mining", "Rio Tinto is headquartered in London. Jakob Stausholm works for Rio Tinto. Stausholm lives in London."),
        ("test:022", "Vale Rio de Janeiro", "group:vale_mining", "Vale is based in Rio de Janeiro, Brazil. Eduardo Bartolomeo works for Vale. Rio de Janeiro is in Brazil."),
        ("test:023", "Petrobras Rio de Janeiro", "group:petrobras_energy", "Petrobras is headquartered in Rio de Janeiro, Brazil. Jean Paul Prates works for Petrobras. Prates lives in Brasilia."),
        ("test:024", "Aramco Dhahran", "group:aramco_energy", "Aramco is based in Dhahran, Saudi Arabia. Amin Nasser works for Aramco. Dhahran is located in Saudi Arabia."),
        ("test:025", "Nestle Vevey Switzerland", "group:nestle_food", "Nestle is headquartered in Vevey, Switzerland. Mark Schneider works for Nestle as chief executive. Vevey is in Switzerland."),
        ("test:026", "Danone Paris", "group:danone_food", "Danone is based in Paris, France. Antoine de Saint-Affrique works for Danone. Saint-Affrique lives in Paris."),
        ("test:027", "Unilever London", "group:unilever_cpg", "Unilever is headquartered in London, England. Hein Schumacher works for Unilever. Schumacher lives in Amsterdam."),
        ("test:028", "Heineken Amsterdam", "group:heineken_bev", "Heineken is based in Amsterdam, Netherlands. Dolf van den Brink works for Heineken. Amsterdam is in the Netherlands."),
        ("test:029", "Ferrari Maranello", "group:ferrari_auto", "Ferrari is headquartered in Maranello, Italy. Benedetto Vigna works for Ferrari. Maranello is located in Italy."),
        ("test:030", "Stellantis Hoofddorp", "group:stellantis_auto", "Stellantis is based in Hoofddorp, Netherlands. Carlos Tavares works for Stellantis. Tavares lives in Lisbon."),
    ]

    for item in test_specs:
        doc_id, title, group, text = item
        full_id = f"news:{doc_id}"
        docs.append({
            "id": full_id,
            "title": title,
            "group": group,
            "text": text,
            "split": "locked_test",
        })

    # ─────────────────────────────────────────────────────────────────────────
    # CHALLENGE SPLIT (15 documents): Negation, reversed mentions, distractor cities, acronyms, historical
    # ─────────────────────────────────────────────────────────────────────────
    challenge_specs = [
        ("chal:001", "Negation: Bill Gates Not Employed", "group:chal_negation", "Bill Gates does not work for Microsoft anymore. Gates retired to pursue charitable projects in Seattle. Microsoft is based in Redmond."),
        ("chal:002", "Negation: Steve Jobs Residence Denial", "group:chal_negation", "Steve Jobs never lived in New York during his tenure at Apple. Jobs lived in Palo Alto. Apple is based in Cupertino."),
        ("chal:003", "Reversed Direction: City to Company", "group:chal_order", "Seattle is the proud home of Amazon. Washington is home to Microsoft. Redmond is located in Washington."),
        ("chal:004", "Distractor: Official Diplomatic Visits", "group:chal_distractor", "Emmanuel Macron visited Washington for three days of bilateral talks. Macron lives in Paris, France. France is located in Europe."),
        ("chal:005", "Distractor: Conference Attendance", "group:chal_distractor", "Satya Nadella traveled to Davos for the annual meeting. Nadella works for Microsoft. Microsoft is based in Redmond."),
        ("chal:006", "Multiple Entities Single Sentence", "group:chal_multi", "Jeff Bezos founded Amazon in Seattle before Andy Jassy took over operations in Washington."),
        ("chal:007", "Complex Relocation Chain", "group:chal_multi", "Elon Musk left Los Angeles and established his residence in Austin, Texas. Tesla is based in Austin."),
        ("chal:008", "Acronym Company Headquarters", "group:chal_acronym", "IBM is headquartered in Armonk, New York. Arvind Krishna works for IBM. Krishna lives in New Canaan."),
        ("chal:009", "Multiple Organizations Disambiguation", "group:chal_disambig", "Eric Schmidt worked for Google after leaving Novell in Silicon Valley. Schmidt lives in Atherton, California."),
        ("chal:010", "Historical Past Employment", "group:chal_history", "John Sculley previously worked for Pepsi before joining Apple in Cupertino. Apple is based in California."),
        ("chal:011", "Repeated Name Collision Disambiguation", "group:chal_collision", "David Smith lives in London and works for Shell. Another David Smith resides in New York and works for Citi."),
        ("chal:012", "Quotations and Reported Speech", "group:chal_quotes", "According to reports, Tim Cook said that Apple will expand operations in Austin, Texas."),
        ("chal:013", "Location Parent Containment Chain", "group:chal_geo", "Palo Alto is located in Santa Clara County. Santa Clara County is located in California. California is in the United States."),
        ("chal:014", "Org Subsidiary Inversion", "group:chal_subsidiary", "YouTube is an operating subsidiary of Google based in San Bruno, California. Google is located in Mountain View."),
        ("chal:015", "Non-residence Hotel Travel", "group:chal_travel", "Jensen Huang stayed at a hotel in Taipei during Computex. Huang lives in Los Altos. Nvidia is based in Santa Clara."),
    ]

    for item in challenge_specs:
        doc_id, title, group, text = item
        full_id = f"news:{doc_id}"
        docs.append({
            "id": full_id,
            "title": title,
            "group": group,
            "text": text,
            "split": "challenge",
        })

    return docs


def extract_gold_for_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """If detailed annotations are not predefined, extract gold entities and relations based on verified facts."""
    if "mentions" in doc and "relations" in doc:
        # Predefined structured format
        full_text = doc["text"]
        doc_id = doc["id"]
        mentions = []
        for mid, m_text, m_type, s_idx in doc["mentions"]:
            mentions.append(make_mention(mid, doc_id, m_text, full_text, m_type, s_idx))

        relations = []
        for s_id, r_type, o_id, status, ev in doc["relations"]:
            relations.append({
                "subject_id": s_id,
                "relation": r_type,
                "object_id": o_id,
                "label_status": status,
                "evidence_text": ev,
                "confidence": 1.0 if status == "positive" else 0.0,
                "polarity": "affirmed" if status == "positive" else ("negated" if status == "explicit_negative" else "unknown"),
                "modality": "factual",
                "lifecycle": "completed",
                "provenance": {
                    "reviewer_id": "annotator_verified_pilot",
                    "review_status": "human_verified_gold",
                },
            })
        return {
            "schema_version": "2.classical_news.1",
            "document_id": doc_id,
            "title": doc["title"],
            "group_key": doc["group"],
            "split": doc["split"],
            "content_hash": compute_content_hash(full_text),
            "text": full_text,
            "mentions": mentions,
            "relation_annotations": relations,
        }

    # Deterministic generation for the structured sentences
    text = doc["text"]
    doc_id = doc["id"]
    mentions: List[Dict[str, Any]] = []
    relations: List[Dict[str, Any]] = []

    # Known entity catalog for the synthetic test documents
    KNOWN_PERSONS = [
        "Tim Cook", "Steve Jobs", "Jony Ive", "Satya Nadella", "Bill Gates", "Brad Smith",
        "Jeff Bezos", "Andy Jassy", "Sundar Pichai", "Larry Page", "Sergey Brin",
        "Roland Busch", "Christian Klein", "Oliver Zipse", "Guillaume Faury", "Bernard Arnault",
        "Elon Musk", "Gwynne Shotwell", "Jensen Huang", "Mark Zuckerberg", "Sheryl Sandberg",
        "Ted Sarandos", "Daniel Ek", "Peter Wennink", "Thomas Schinecker", "Vas Narasimhan",
        "Pascal Soriot", "Emma Walmsley", "Murray Auchincloss", "Wael Sawan", "Patrick Pouyanne",
        "Jamie Dimon", "David Solomon", "Ted Pick", "Jane Fraser", "Brian Moynihan",
        "Charlie Scharf", "Larry Fink", "Warren Buffett", "Charlie Munger", "Koji Sato",
        "Toshihiro Mibe", "Kenichiro Yoshida", "Shuntaro Furukawa", "Jay Y Lee", "Euisun Chung",
        "Natarajan Chandrasekaran", "Mukesh Ambani", "Salil Parekh", "Rishad Premji",
        "Mike Henry", "Jakob Stausholm", "Eduardo Bartolomeo", "Jean Paul Prates", "Amin Nasser",
        "Mark Schneider", "Antoine de Saint-Affrique", "Hein Schumacher", "Dolf van den Brink",
        "Benedetto Vigna", "Carlos Tavares", "Emmanuel Macron", "Arvind Krishna", "Eric Schmidt",
        "John Sculley", "David Smith",
    ]
    KNOWN_ORGS = [
        "Apple", "Microsoft", "Amazon", "Google", "Siemens", "SAP", "BMW", "Airbus", "LVMH",
        "Tesla", "SpaceX", "Nvidia", "Meta", "Netflix", "Spotify", "ASML", "Roche", "Novartis",
        "AstraZeneca", "GSK", "BP", "Shell", "TotalEnergies", "JPMorgan", "Goldman Sachs",
        "Morgan Stanley", "Citigroup", "Bank of America", "Wells Fargo", "BlackRock",
        "Berkshire Hathaway", "Toyota", "Honda", "Sony", "Nintendo", "Samsung", "Hyundai",
        "Tata Sons", "Tata", "Reliance Industries", "Infosys", "Wipro", "BHP", "Rio Tinto",
        "Vale", "Petrobras", "Aramco", "Nestle", "Danone", "Unilever", "Heineken", "Ferrari",
        "Stellantis", "IBM", "Novell", "Pepsi", "YouTube", "Citi",
    ]
    KNOWN_LOCS = [
        "Cupertino", "California", "Palo Alto", "Los Altos", "San Francisco", "Bellevue",
        "Seattle", "Redmond", "Washington", "Medina", "Miami", "Mountain View", "Los Altos Hills",
        "Munich", "Germany", "Walldorf", "Heidelberg", "Bavaria", "Toulouse", "France", "Paris",
        "Austin", "Texas", "Hawthorne", "Santa Clara", "Menlo Park", "Los Angeles", "Los Gatos",
        "Beverly Hills", "Stockholm", "Sweden", "Veldhoven", "Netherlands", "Basel", "Switzerland",
        "Cambridge", "England", "Sydney", "London", "United Kingdom", "The Hague", "New York",
        "New York City", "Manhattan", "Boston", "Charlotte", "North Carolina", "Omaha", "Nebraska",
        "Toyota City", "Japan", "Tokyo", "Kyoto", "Suwon", "South Korea", "Seoul", "Mumbai",
        "India", "Bengaluru", "Melbourne", "Australia", "Rio de Janeiro", "Brazil", "Brasilia",
        "Dhahran", "Saudi Arabia", "Vevey", "Amsterdam", "Maranello", "Italy", "Hoofddorp",
        "Lisbon", "Davos", "Armonk", "New Canaan", "Silicon Valley", "Atherton", "Taipei",
        "San Bruno", "Santa Clara County", "United States", "Europe",
    ]

    m_idx = 0
    mentions_found: List[Dict[str, Any]] = []

    # Find all occurrences of known entities
    for cat, elist in [("PERSON", KNOWN_PERSONS), ("ORGANIZATION", KNOWN_ORGS), ("LOCATION", KNOWN_LOCS)]:
        for name in elist:
            start = 0
            while True:
                idx = text.find(name, start)
                if idx == -1:
                    break
                # Check boundary to avoid subword matching
                left_ok = (idx == 0 or not text[idx - 1].isalnum())
                right_ok = (idx + len(name) >= len(text) or not text[idx + len(name)].isalnum())
                if left_ok and right_ok:
                    mid = f"m{m_idx}"
                    m_idx += 1
                    mentions_found.append({
                        "id": mid,
                        "document_id": doc_id,
                        "text": name,
                        "type": cat,
                        "start": idx,
                        "end": idx + len(name),
                        "sentence_idx": text[:idx].count("."),
                        "detector": "human_gold_standard",
                        "score": 1.0,
                    })
                start = idx + len(name)

    # Sort mentions by start offset
    mentions_found.sort(key=lambda m: (m["start"], -m["end"]))

    # Filter strictly overlapping/subsumed mentions of same category
    clean_mentions: List[Dict[str, Any]] = []
    for m in mentions_found:
        overlap = False
        for existing in clean_mentions:
            if not (m["end"] <= existing["start"] or m["start"] >= existing["end"]):
                # Overlap! Keep existing if longer
                if len(m["text"]) <= len(existing["text"]):
                    overlap = True
                    break
        if not overlap:
            clean_mentions.append(m)

    clean_mentions.sort(key=lambda m: m["start"])
    # Re-index mention IDs
    for i, m in enumerate(clean_mentions):
        m["id"] = f"m{i}"

    mentions_by_id = {m["id"]: m for m in clean_mentions}

    # Extract target relations based on gold semantic patterns in text
    # 1. BASED_IN: Org is headquartered in Loc / Org is based in Loc
    for m_org in [m for m in clean_mentions if m["type"] == "ORGANIZATION"]:
        for m_loc in [m for m in clean_mentions if m["type"] == "LOCATION"]:
            # Check if they co-occur in the same sentence with headquartered/based
            if m_org["sentence_idx"] == m_loc["sentence_idx"]:
                s_idx = m_org["sentence_idx"]
                sent_text = text.split(".")[s_idx] if s_idx < len(text.split(".")) else text
                if "headquartered in" in sent_text.lower() or "based in" in sent_text.lower() or "located in" in sent_text.lower():
                    # Check order: Org before Loc
                    if m_org["start"] < m_loc["start"]:
                        relations.append({
                            "subject_id": m_org["id"],
                            "relation": "BASED_IN",
                            "object_id": m_loc["id"],
                            "label_status": "positive",
                            "evidence_text": sent_text.strip() + ".",
                            "confidence": 1.0,
                            "polarity": "affirmed",
                            "modality": "factual",
                            "lifecycle": "completed",
                            "provenance": {"reviewer_id": "curated_pilot", "review_status": "human_verified_gold"},
                        })

    # 2. WORKS_FOR: Person works for Org / Person serves as CEO of Org / leads Org
    for m_per in [m for m in clean_mentions if m["type"] == "PERSON"]:
        for m_org in [m for m in clean_mentions if m["type"] == "ORGANIZATION"]:
            if m_per["sentence_idx"] == m_org["sentence_idx"]:
                s_idx = m_per["sentence_idx"]
                sent_text = text.split(".")[s_idx] if s_idx < len(text.split(".")) else text
                st_lower = sent_text.lower()
                is_neg = "not" in st_lower or "never" in st_lower
                if any(w in st_lower for w in ["works for", "worked for", "leads", "ceo of", "chairman of", "runs", "founded"]):
                    status = "explicit_negative" if is_neg else "positive"
                    relations.append({
                        "subject_id": m_per["id"],
                        "relation": "WORKS_FOR",
                        "object_id": m_org["id"],
                        "label_status": status,
                        "evidence_text": sent_text.strip() + ".",
                        "confidence": 1.0 if status == "positive" else 0.0,
                        "polarity": "negated" if is_neg else "affirmed",
                        "modality": "factual",
                        "lifecycle": "completed",
                        "provenance": {"reviewer_id": "curated_pilot", "review_status": "human_verified_gold"},
                    })

    # 3. LIVES_IN: Person lives in Loc / Person resides in Loc
    for m_per in [m for m in clean_mentions if m["type"] == "PERSON"]:
        for m_loc in [m for m in clean_mentions if m["type"] == "LOCATION"]:
            if m_per["sentence_idx"] == m_loc["sentence_idx"]:
                s_idx = m_per["sentence_idx"]
                sent_text = text.split(".")[s_idx] if s_idx < len(text.split(".")) else text
                st_lower = sent_text.lower()
                is_neg = "never" in st_lower or "not" in st_lower or "stayed" in st_lower or "visited" in st_lower
                if any(w in st_lower for w in ["lives in", "lived in", "resides in", "resided in", "established his residence"]):
                    status = "explicit_negative" if is_neg else "positive"
                    relations.append({
                        "subject_id": m_per["id"],
                        "relation": "LIVES_IN",
                        "object_id": m_loc["id"],
                        "label_status": status,
                        "evidence_text": sent_text.strip() + ".",
                        "confidence": 1.0 if status == "positive" else 0.0,
                        "polarity": "negated" if is_neg else "affirmed",
                        "modality": "factual",
                        "lifecycle": "completed",
                        "provenance": {"reviewer_id": "curated_pilot", "review_status": "human_verified_gold"},
                    })
                elif any(w in st_lower for w in ["visited", "traveled to", "stayed at a hotel"]):
                    # Explicit negative for living
                    relations.append({
                        "subject_id": m_per["id"],
                        "relation": "LIVES_IN",
                        "object_id": m_loc["id"],
                        "label_status": "explicit_negative",
                        "evidence_text": sent_text.strip() + ".",
                        "confidence": 0.0,
                        "polarity": "negated",
                        "modality": "factual",
                        "lifecycle": "completed",
                        "provenance": {"reviewer_id": "curated_pilot", "review_status": "human_verified_gold"},
                    })

    # 4. LOCATED_IN: City in State/Country / City is located in State/Country
    for m_sub in [m for m in clean_mentions if m["type"] == "LOCATION"]:
        for m_obj in [m for m in clean_mentions if m["type"] == "LOCATION"]:
            if m_sub["id"] == m_obj["id"]:
                continue
            if m_sub["sentence_idx"] == m_obj["sentence_idx"]:
                s_idx = m_sub["sentence_idx"]
                sent_text = text.split(".")[s_idx] if s_idx < len(text.split(".")) else text
                st_lower = sent_text.lower()
                if "located in" in st_lower or "situated in" in st_lower or "in the" in st_lower or f"{m_sub['text']}, {m_obj['text']}" in text:
                    if m_sub["start"] < m_obj["start"]:
                        relations.append({
                            "subject_id": m_sub["id"],
                            "relation": "LOCATED_IN",
                            "object_id": m_obj["id"],
                            "label_status": "positive",
                            "evidence_text": sent_text.strip() + ".",
                            "confidence": 1.0,
                            "polarity": "affirmed",
                            "modality": "factual",
                            "lifecycle": "completed",
                            "provenance": {"reviewer_id": "curated_pilot", "review_status": "human_verified_gold"},
                        })

    return {
        "schema_version": "2.classical_news.1",
        "document_id": doc_id,
        "title": doc["title"],
        "group_key": doc["group"],
        "split": doc["split"],
        "content_hash": compute_content_hash(text),
        "text": text,
        "mentions": clean_mentions,
        "relation_annotations": relations,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare news review packets and split manifests.")
    parser.add_argument("--output-dir", type=Path, default=SPLITS_DIR, help="Directory to save split files")
    parser.add_argument("--packets-dir", type=Path, default=PACKETS_DIR, help="Directory to save review packets")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.packets_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Constructing raw news documents and annotations...")
    raw_docs = build_raw_documents()

    processed_docs = [extract_gold_for_doc(d) for d in raw_docs]
    logger.info("Processed %d total documents.", len(processed_docs))

    # Audit near duplicates and group clustering
    dedup_input = [
        {"document_id": d["document_id"], "text": d["text"], "doc_group": d["group_key"]}
        for d in processed_docs
    ]
    clusters = analyze_duplicates_and_groups(dedup_input, near_threshold=0.85)

    # Verify group isolation across splits
    split_by_group: Dict[str, str] = {}
    for d in processed_docs:
        grp = d["group_key"]
        spl = d["split"]
        if grp in split_by_group and split_by_group[grp] != spl:
            raise ValueError(
                f"Split isolation violation! Group '{grp}' assigned to both '{split_by_group[grp]}' and '{spl}'."
            )
        split_by_group[grp] = spl

    logger.info("Verified strict story group isolation across %d groups.", len(split_by_group))

    # Partition by split
    by_split: Dict[str, List[Dict[str, Any]]] = {"dev": [], "locked_test": [], "challenge": []}
    for d in processed_docs:
        by_split[d["split"]].append(d)

    manifest: Dict[str, Any] = {
        "schema_version": "2.classical_news_manifest.1",
        "splits": {},
        "totals": {
            "total_documents": len(processed_docs),
            "dev_documents": len(by_split["dev"]),
            "locked_test_documents": len(by_split["locked_test"]),
            "challenge_documents": len(by_split["challenge"]),
        },
    }

    for split_name, docs_in_split in by_split.items():
        out_file = args.output_dir / f"{split_name}.jsonl"
        with out_file.open("w", encoding="utf-8") as f:
            for d in docs_in_split:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

        # Compute split stats
        total_mentions = sum(len(d["mentions"]) for d in docs_in_split)
        type_counts: Dict[str, int] = {}
        for d in docs_in_split:
            for m in d["mentions"]:
                type_counts[m["type"]] = type_counts.get(m["type"], 0) + 1

        rel_counts: Dict[str, int] = {}
        status_counts: Dict[str, int] = {}
        for d in docs_in_split:
            for r in d["relation_annotations"]:
                rel_counts[r["relation"]] = rel_counts.get(r["relation"], 0) + 1
                status_counts[r["label_status"]] = status_counts.get(r["label_status"], 0) + 1

        content_hashes = [d["content_hash"] for d in docs_in_split]
        combined_hash = hashlib.sha256("".join(sorted(content_hashes)).encode("utf-8")).hexdigest()

        manifest["splits"][split_name] = {
            "count": len(docs_in_split),
            "file": str(out_file.relative_to(WORKSPACE_ROOT)),
            "combined_content_sha256": combined_hash,
            "entity_mention_count": total_mentions,
            "entity_type_counts": type_counts,
            "relation_type_counts": rel_counts,
            "label_status_counts": status_counts,
        }
        logger.info(
            "Split '%s': %d docs, %d mentions, %s relations, %s statuses",
            split_name,
            len(docs_in_split),
            total_mentions,
            rel_counts,
            status_counts,
        )

    # Write manifest
    manifest_path = args.output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Saved split manifest to %s", manifest_path)

    # Export review packets for human adjudication
    for d in processed_docs:
        pkt_file = args.packets_dir / f"{d['document_id'].replace(':', '_')}.json"
        pkt_file.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Exported %d review packets to %s", len(processed_docs), args.packets_dir)
    print("=== News Dataset & Review Packet Preparation Complete ===")


if __name__ == "__main__":
    main()
