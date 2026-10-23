#!/usr/bin/env python
"""CLI document ingestion script.

Usage:
    python scripts/ingest_sample.py --input data/raw/
    python scripts/ingest_sample.py --input data/raw/report.pdf --mode accurate
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Doc2Graph CLI ingestion")
    parser.add_argument(
        "--input", required=True, help="Path to a file or directory of files"
    )
    parser.add_argument(
        "--mode", choices=["fast", "accurate"], default="accurate",
        help="Pipeline mode (default: accurate)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    files: list[Path] = []

    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = list(input_path.glob("*.pdf")) + \
                list(input_path.glob("*.docx")) + \
                list(input_path.glob("*.txt"))
    else:
        print(f"Error: {input_path} does not exist")
        sys.exit(1)

    if not files:
        print("No PDF/DOCX/TXT files found.")
        sys.exit(0)

    print(f"Found {len(files)} file(s) to process.")

    for fpath in files:
        print(f"\n-> Processing: {fpath.name}")
        raw_bytes = fpath.read_bytes()

        # Detect type by extension for CLI (magic bytes checked inside pipeline)
        ext = fpath.suffix.lower()
        ftype_map = {".pdf": "pdf", ".docx": "docx", ".txt": "txt"}
        ftype = ftype_map.get(ext)
        if ftype is None:
            print(f"  Skipping unsupported extension: {ext}")
            continue

        try:
            result = run_pipeline(raw_bytes, fpath.name, ftype, job_id="cli")
            print(f"  [OK] {result['entities']} entities, {result['edges']} edges")
        except Exception as exc:
            print(f"  [ERROR] Failed: {exc}")


if __name__ == "__main__":
    main()
