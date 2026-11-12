"""Pipeline lock — cross-process CPU resource guard using a file lock.

The lock prevents concurrent pipeline runs on a single host (which would cause
memory spikes and spaCy model contention).

Architecture & Multi-Container Scaling:
  - In a single-container or local deployment, this filelock guards the extraction pipeline.
  - If scaling horizontally across multiple container replicas in docker-compose or Kubernetes,
    local file locks do not coordinate across containers. To scale horizontally:
    1. Distribute pipeline jobs via a shared queue (e.g. Celery / RQ backed by Redis), or
    2. Use distributed locks via Redis (Redlock) or Neo4j transactional locks.
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
