"""Pipeline lock — cross-process GPU/CPU resource guard using a file lock.

The lock prevents the CLI and the API server from running two pipeline
instances simultaneously (which would cause memory pressure and slow both).
"""
from __future__ import annotations

import filelock

_LOCK_PATH = "data/.pipeline.lock"
_lock = filelock.FileLock(_LOCK_PATH, timeout=0)


def acquire() -> None:
    """Acquire the pipeline lock. Raises RuntimeError if already held."""
    try:
        _lock.acquire()
    except filelock.Timeout:
        raise RuntimeError(
            "Another pipeline process is already running. "
            "Wait for it to finish before ingesting another document."
        )


def release() -> None:
    """Release the pipeline lock."""
    if _lock.is_locked:
        _lock.release()
