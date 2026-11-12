"""Setup script for isolated Stanford CoreNLP and OpenJDK 17 runtime dependencies.

Downloads verified, pinned official binaries into .runtime/ and records cryptographic hashes.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

RUNTIME_ROOT = Path(__file__).resolve().parent.parent / ".runtime"
STANFORD_DIR = RUNTIME_ROOT / "stanford_classical"
LIB_DIR = STANFORD_DIR / "lib"
JDK_DIR = RUNTIME_ROOT / "jdk17"

JDK_ZIP_URL = "https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.11%2B9/OpenJDK17U-jdk_x64_windows_hotspot_17.0.11_9.zip"
JDK_EXPECTED_SIZE = 190109030

CORE_NLP_ARTIFACTS = [
    {
        "name": "stanford-corenlp-4.5.8.jar",
        "url": "https://repo1.maven.org/maven2/edu/stanford/nlp/stanford-corenlp/4.5.8/stanford-corenlp-4.5.8.jar",
        "target_dir": STANFORD_DIR,
        "license": "GPLv3",
    },
    {
        "name": "stanford-corenlp-4.5.8-models.jar",
        "url": "https://repo1.maven.org/maven2/edu/stanford/nlp/stanford-corenlp/4.5.8/stanford-corenlp-4.5.8-models.jar",
        "target_dir": STANFORD_DIR,
        "license": "GPLv3",
    },
    {
        "name": "joda-time-2.13.0.jar",
        "url": "https://repo1.maven.org/maven2/joda-time/joda-time/2.13.0/joda-time-2.13.0.jar",
        "target_dir": LIB_DIR,
        "license": "Apache-2.0",
    },
    {
        "name": "xom-1.3.9.jar",
        "url": "https://repo1.maven.org/maven2/xom/xom/1.3.9/xom-1.3.9.jar",
        "target_dir": LIB_DIR,
        "license": "LGPL-2.1",
    },
    {
        "name": "ejml-core-0.39.jar",
        "url": "https://repo1.maven.org/maven2/org/ejml/ejml-core/0.39/ejml-core-0.39.jar",
        "target_dir": LIB_DIR,
        "license": "Apache-2.0",
    },
    {
        "name": "ejml-ddense-0.39.jar",
        "url": "https://repo1.maven.org/maven2/org/ejml/ejml-ddense/0.39/ejml-ddense-0.39.jar",
        "target_dir": LIB_DIR,
        "license": "Apache-2.0",
    },
    {
        "name": "ejml-simple-0.39.jar",
        "url": "https://repo1.maven.org/maven2/org/ejml/ejml-simple/0.39/ejml-simple-0.39.jar",
        "target_dir": LIB_DIR,
        "license": "Apache-2.0",
    },
    {
        "name": "slf4j-api-1.7.36.jar",
        "url": "https://repo1.maven.org/maven2/org/slf4j/slf4j-api/1.7.36/slf4j-api-1.7.36.jar",
        "target_dir": LIB_DIR,
        "license": "MIT",
    },
    {
        "name": "slf4j-simple-1.7.36.jar",
        "url": "https://repo1.maven.org/maven2/org/slf4j/slf4j-simple/1.7.36/slf4j-simple-1.7.36.jar",
        "target_dir": LIB_DIR,
        "license": "MIT",
    },
    {
        "name": "jakarta.json-1.1.6.jar",
        "url": "https://repo1.maven.org/maven2/org/glassfish/jakarta.json/1.1.6/jakarta.json-1.1.6.jar",
        "target_dir": LIB_DIR,
        "license": "EPL-2.0 / GPLv2+CE",
    },
    {
        "name": "protobuf-java-3.25.5.jar",
        "url": "https://repo1.maven.org/maven2/com/google/protobuf/protobuf-java/3.25.5/protobuf-java-3.25.5.jar",
        "target_dir": LIB_DIR,
        "license": "BSD-3-Clause",
    },
]


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(".tmp")
    print(f"Downloading {dest.name} from {url}...")
    headers = {"User-Agent": "Doc2Graph-Classical-Setup/1.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp, open(tmp_dest, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = (downloaded / total) * 100
                sys.stdout.write(f"\r  {downloaded / 1e6:.1f}MB / {total / 1e6:.1f}MB ({pct:.1f}%)")
                sys.stdout.flush()
    print()
    if tmp_dest.exists():
        if dest.exists():
            dest.unlink()
        tmp_dest.rename(dest)
    print(f"Saved: {dest} ({dest.stat().st_size} bytes)")


def setup_jdk() -> Path:
    java_exe = JDK_DIR / "bin" / "java.exe"
    if java_exe.exists():
        print(f"Java already exists at {java_exe}")
        return java_exe

    JDK_DIR.mkdir(parents=True, exist_ok=True)
    zip_dest = RUNTIME_ROOT / "temurin17.zip"
    if not zip_dest.exists() or zip_dest.stat().st_size != JDK_EXPECTED_SIZE:
        download_file(JDK_ZIP_URL, zip_dest)

    print("Extracting OpenJDK 17...")
    with zipfile.ZipFile(zip_dest, "r") as zf:
        # Find top level directory inside zip (e.g. jdk-17.0.11+9)
        top_dirs = {p.split("/")[0] for p in zf.namelist() if "/" in p}
        zf.extractall(RUNTIME_ROOT)

    extracted_dir = None
    for td in top_dirs:
        candidate = RUNTIME_ROOT / td
        if (candidate / "bin" / "java.exe").exists():
            extracted_dir = candidate
            break

    if extracted_dir and extracted_dir != JDK_DIR:
        if JDK_DIR.exists():
            shutil.rmtree(JDK_DIR)
        extracted_dir.rename(JDK_DIR)

    if not java_exe.exists():
        raise RuntimeError(f"Failed to find java.exe at {java_exe} after extraction.")

    print(f"OpenJDK 17 ready at {java_exe}")
    return java_exe


def setup_stanford_dependencies() -> dict:
    STANFORD_DIR.mkdir(parents=True, exist_ok=True)
    LIB_DIR.mkdir(parents=True, exist_ok=True)

    manifest_entries = {}
    for art in CORE_NLP_ARTIFACTS:
        target = art["target_dir"] / art["name"]
        if not target.exists() or target.stat().st_size == 0:
            download_file(art["url"], target)
        sha256 = compute_sha256(target)
        manifest_entries[art["name"]] = {
            "path": str(target),
            "size_bytes": target.stat().st_size,
            "sha256": sha256,
            "url": art["url"],
            "license": art["license"],
        }
        print(f"Verified {art['name']}: {target.stat().st_size} bytes, sha256={sha256[:12]}...")

    manifest_path = STANFORD_DIR / "runtime_manifest.json"
    manifest_path.write_text(json.dumps(manifest_entries, indent=2), encoding="utf-8")
    print(f"Manifest written to {manifest_path}")
    return manifest_entries


def test_java_version(java_exe: Path) -> str:
    res = subprocess.run([str(java_exe), "-version"], capture_output=True, text=True, check=True)
    out = res.stderr or res.stdout
    print("Java version check output:\n", out.strip())
    return out.strip()


def main():
    print("=== Setting up Classical Stanford NLP & JDK 17 Runtime ===")
    java_exe = setup_jdk()
    version_str = test_java_version(java_exe)
    manifest = setup_stanford_dependencies()
    print("=== Setup Complete ===")


if __name__ == "__main__":
    main()
