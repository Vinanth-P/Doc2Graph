"""Python client and lifecycle manager for Stanford Classical CoreNLP Bridge.

Manages persistent subprocess communication over JSONL with explicit UTF-16 code-unit
to Python Unicode code-point offset mapping, health checks, and fail-closed error handling.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
RUNTIME_DIR = WORKSPACE_ROOT / ".runtime"
JDK_BIN = RUNTIME_DIR / "jdk17" / "bin" / ("java.exe" if os.name == "nt" else "java")
STANFORD_DIR = RUNTIME_DIR / "stanford_classical"
BRIDGE_BUILD = WORKSPACE_ROOT / "java" / "stanford-classical-bridge" / "build"


def build_utf16_to_codepoint_map(text: str) -> list[int]:
    """Map each UTF-16 code unit offset to the corresponding Python unicode character index.

    BMP characters (code <= 0xFFFF) take 1 code unit in UTF-16.
    Supplementary characters (code > 0xFFFF, e.g. emoji) take 2 code units in UTF-16.
    The returned list has length equal to (num_utf16_code_units + 1).
    """
    mapping: list[int] = []
    for char_idx, ch in enumerate(text):
        mapping.append(char_idx)
        if ord(ch) > 0xFFFF:
            # Low surrogate maps to the same Python character index
            mapping.append(char_idx)
    mapping.append(len(text))
    return mapping


def convert_utf16_span(mapping: list[int], u16_start: int, u16_end: int) -> Tuple[int, int]:
    """Convert Java UTF-16 code-unit span [start, end] to Python unicode slice indices."""
    if not mapping:
        return 0, 0
    max_idx = len(mapping) - 1
    py_start = mapping[max(0, min(u16_start, max_idx))]
    py_end = mapping[max(0, min(u16_end, max_idx))]
    if py_end < py_start:
        py_end = py_start
    return py_start, py_end


class StanfordClassicalClient:
    """Manages persistent subprocess communication with StanfordClassicalBridge."""

    def __init__(
        self,
        java_bin: Optional[Path] = None,
        max_heap: str = "2g",
        startup_timeout: float = 45.0,
        request_timeout: float = 60.0,
    ):
        self.java_bin = java_bin or JDK_BIN
        self.max_heap = max_heap
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self.proc: Optional[subprocess.Popen] = None
        self._lock = threading.RLock()
        self.ready_metadata: Dict[str, Any] = {}

    def is_available(self) -> bool:
        """Check if all required binaries and jars exist on disk."""
        if not self.java_bin.exists():
            return False
        if not (STANFORD_DIR / "stanford-corenlp-4.5.8.jar").exists():
            return False
        if not (STANFORD_DIR / "stanford-corenlp-4.5.8-models.jar").exists():
            return False
        if not (BRIDGE_BUILD / "edu" / "stanford" / "nlp" / "bridge" / "StanfordClassicalBridge.class").exists():
            return False
        return True

    def start(self) -> None:
        """Launch the Java bridge subprocess and await ready handshake."""
        with self._lock:
            if self.proc is not None and self.proc.poll() is None:
                return  # already running

            if not self.is_available():
                raise RuntimeError(
                    f"Stanford Classical dependencies not found. Ensure OpenJDK 17 and CoreNLP 4.5.8 jars "
                    f"are present in {STANFORD_DIR} and bridge is compiled in {BRIDGE_BUILD}."
                )

            sep = ";" if os.name == "nt" else ":"
            classpath = (
                f"{BRIDGE_BUILD}{sep}"
                f"{STANFORD_DIR / 'stanford-corenlp-4.5.8.jar'}{sep}"
                f"{STANFORD_DIR / 'stanford-corenlp-4.5.8-models.jar'}{sep}"
                f"{STANFORD_DIR / 'lib' / '*'}"
            )

            cmd = [
                str(self.java_bin),
                f"-Xmx{self.max_heap}",
                "-Dfile.encoding=UTF-8",
                "-cp",
                classpath,
                "edu.stanford.nlp.bridge.StanfordClassicalBridge",
            ]

            logger.info("Starting Stanford Classical Bridge subprocess: %s", " ".join(cmd[:3]))
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )

            # Prevent pipe deadlock by reading stderr continuously in a daemon thread
            self._stderr_lines: list[str] = []
            def _drain_stderr(p: subprocess.Popen, logs: list[str]) -> None:
                try:
                    for s_line in iter(p.stderr.readline, ''):
                        if s_line:
                            logs.append(s_line.rstrip())
                            if len(logs) > 300:
                                logs.pop(0)
                except Exception:
                    pass

            self._stderr_thread = threading.Thread(
                target=_drain_stderr,
                args=(self.proc, self._stderr_lines),
                daemon=True,
            )
            self._stderr_thread.start()

            # Read startup ready message
            start_time = time.time()
            ready_line = None
            while time.time() - start_time < self.startup_timeout:
                if self.proc.poll() is not None:
                    err = "\n".join(self._stderr_lines)
                    raise RuntimeError(f"Bridge process terminated unexpectedly during startup: {err}")
                line = self.proc.stdout.readline()
                if line:
                    line = line.strip()
                    if line:
                        try:
                            msg = json.loads(line)
                            if msg.get("status") == "ready":
                                ready_line = msg
                                break
                        except json.JSONDecodeError:
                            logger.warning("Bridge non-JSON startup output: %s", line)
                time.sleep(0.05)

            if not ready_line:
                self.close()
                raise TimeoutError(f"Stanford Classical Bridge did not emit ready signal within {self.startup_timeout}s.")

            self.ready_metadata = ready_line
            logger.info("Stanford Classical Bridge ready: %s", ready_line)

    def annotate(self, text: str, req_id: Optional[str] = None) -> Dict[str, Any]:
        """Annotate text with sentences, CRF entities, and Roth relations."""
        with self._lock:
            if self.proc is None or self.proc.poll() is not None:
                self.start()

            rid = req_id or f"req-{int(time.time() * 1000)}"
            payload = json.dumps({"id": rid, "text": text}, ensure_ascii=False)

            try:
                self.proc.stdin.write(payload + "\n")
                self.proc.stdin.flush()
                resp_line = self.proc.stdout.readline()
                if not resp_line:
                    raise RuntimeError("Received empty response from Stanford Classical Bridge.")
                resp = json.loads(resp_line.strip())
            except Exception as e:
                logger.error("Bridge communication error: %s", e)
                self.close()
                raise

            if resp.get("status") != "ok":
                raise RuntimeError(f"Bridge error: {resp.get('error', 'unknown error')}")

            # Post-process character spans from UTF-16 code units to Python unicode codepoints
            u16_map = build_utf16_to_codepoint_map(text)
            for phrase in resp.get("noun_phrases", []):
                phrase["char_start"], phrase["char_end"] = convert_utf16_span(
                    u16_map, phrase["char_start"], phrase["char_end"])

            for s in resp.get("sentences", []):
                s["char_start"], s["char_end"] = convert_utf16_span(u16_map, s["char_start"], s["char_end"])

            for e in resp.get("entities", []):
                e["char_start"], e["char_end"] = convert_utf16_span(u16_map, e["char_start"], e["char_end"])
                e["head_char_start"], e["head_char_end"] = convert_utf16_span(
                    u16_map, e["head_char_start"], e["head_char_end"]
                )
                # Verify exact span boundary match
                e["mapped_text"] = text[e["char_start"]:e["char_end"]]

            for r in resp.get("relations", []):
                for arg_key in ("arg0", "arg1"):
                    arg = r.get(arg_key, {})
                    arg["char_start"], arg["char_end"] = convert_utf16_span(
                        u16_map, arg["char_start"], arg["char_end"]
                    )
                    arg["head_char_start"], arg["head_char_end"] = convert_utf16_span(
                        u16_map, arg["head_char_start"], arg["head_char_end"]
                    )
                    arg["mapped_text"] = text[arg["char_start"]:arg["char_end"]]

            return resp

    def close(self) -> None:
        """Gracefully shut down the bridge process."""
        with self._lock:
            if self.proc is not None:
                try:
                    if self.proc.poll() is None:
                        self.proc.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
                        self.proc.stdin.flush()
                        self.proc.wait(timeout=5.0)
                except Exception:
                    self.proc.kill()
                finally:
                    self.proc = None

    def __enter__(self) -> "StanfordClassicalClient":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


_SHARED_CLIENT: Optional[StanfordClassicalClient] = None
_SHARED_CLIENT_LOCK = threading.Lock()


def get_shared_stanford_client() -> StanfordClassicalClient:
    """Return a process-wide shared StanfordClassicalClient instance."""
    global _SHARED_CLIENT
    with _SHARED_CLIENT_LOCK:
        if _SHARED_CLIENT is None:
            _SHARED_CLIENT = StanfordClassicalClient()
        return _SHARED_CLIENT


def shutdown_shared_stanford_client() -> None:
    """Shut down the shared bridge subprocess if active."""
    global _SHARED_CLIENT
    with _SHARED_CLIENT_LOCK:
        if _SHARED_CLIENT is not None:
            try:
                _SHARED_CLIENT.close()
            except Exception:
                pass
            _SHARED_CLIENT = None


atexit.register(shutdown_shared_stanford_client)

