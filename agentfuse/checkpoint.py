"""Durable run state, so a restart does not disarm the breaker.

The supervisor has been in-memory only. That is not merely inconvenient for
long-horizon runs — it is a **safety hole**, and the spend detector shows it
most clearly: an agent with a 500,000-token ceiling that dies at 480,000 and is
restarted comes back with its budget reset to zero. The restart hands a runaway
run a brand new allowance, and the mechanism whose entire job is bounding
unattended spend has been silently rearmed rather than enforced.

The same argument applies to every counter. Loop repeats, the stall counter, the
Zeno series, the learned calibration baseline and the recovery memory of what has
already been tried all reset, so the agent resumes with a clean slate against a
supervisor that has forgotten why it was worried. For a system whose premise is
*hours to days* of autonomy, "we forget everything on restart" undercuts the
premise.

What is saved
-------------
Detector internals, the calibrator's learned baseline, and the monitor's
accumulated totals — enough that a resumed run keeps counting rather than
starting over. Deliberately **not** saved: the full event history (unbounded, and
detectors only ever consult bounded windows), the embedding cache (rebuildable,
and large), and any callable.

Correctness approach
--------------------
``state_dict`` is written generically, by reflection over instance attributes,
rather than as five hand-written serialisers that drift apart from the classes
they mirror. Reflection can silently miss state, so the guarantee comes from a
*behavioural* test instead of from inspection: drive a detector, checkpoint it,
restore into a fresh instance, and assert the two then behave identically on the
next events. A serialiser that loses something fails that test.

Storage is stdlib ``sqlite3`` — the core stays dependency-free, and a single
file with WAL journalling survives a hard kill in a way a half-written JSON blob
does not.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import deque
from typing import Any, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    updated_at REAL NOT NULL,
    step       INTEGER NOT NULL DEFAULT 0,
    state      TEXT NOT NULL
);
"""

#: Attribute names never persisted, on any object. Callables and live clients
#: cannot be serialised, the embedding cache is rebuildable and large, and the
#: calibrator is shared by several detectors so it is saved once, separately.
TRANSIENT = frozenset({
    "calibrator", "_embedder", "_cache", "_goal_vec", "_client", "tracer",
    "memory", "recovery", "detectors", "history", "_lock", "config",
})


# ---------------------------------------------------------------- encoding
def encode(value: Any) -> Any:
    """JSON-safe form that remembers the container types we actually use."""
    if isinstance(value, deque):
        return {"__deque__": [encode(v) for v in value], "maxlen": value.maxlen}
    if isinstance(value, tuple):
        return {"__tuple__": [encode(v) for v in value]}
    if isinstance(value, set):
        return {"__set__": [encode(v) for v in value]}
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [encode(v) for v in value]
    return value


def decode(value: Any) -> Any:
    if isinstance(value, dict):
        if "__deque__" in value:
            return deque((decode(v) for v in value["__deque__"]),
                         maxlen=value.get("maxlen"))
        if "__tuple__" in value:
            return tuple(decode(v) for v in value["__tuple__"])
        if "__set__" in value:
            return {decode(v) for v in value["__set__"]}
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


def _serialisable(value: Any) -> bool:
    if callable(value):
        return False
    return isinstance(value, (str, int, float, bool, type(None),
                              list, tuple, set, dict, deque))


def state_dict(obj: Any) -> dict:
    """Every persistable instance attribute of ``obj``.

    Each attribute is proven encodable before it is kept, one at a time, and
    silently dropped otherwise. This is not belt-and-braces: the container check
    above passes a *list* without inspecting what is inside it, and a list of
    ``numpy.float32`` — which is what an embedding backend hands back — is not
    JSON. Serialised with a permissive ``default=str`` it round-tripped into a
    list of STRINGS, and the restored detector then failed deep inside a cosine
    similarity with a type error, several layers from the cause.

    Dropping a value costs a counter that restarts. Writing a corrupted one
    costs a supervisor that appears to restore and then breaks the run.
    """
    out: dict = {}
    for key, value in vars(obj).items():
        if key in TRANSIENT or not _serialisable(value):
            continue
        encoded = encode(value)
        try:
            json.dumps(encoded)          # strict: no default= coercion
        except (TypeError, ValueError):
            continue
        out[key] = encoded
    return out


def load_state_dict(obj: Any, state: dict) -> None:
    """Restore attributes saved by :func:`state_dict`, ignoring unknown keys.

    Unknown keys are skipped rather than raising so a checkpoint written by an
    older version still loads. The cost of a missing field is a counter that
    restarts; the cost of refusing to load is losing the whole run's state,
    which is strictly worse.
    """
    for key, value in (state or {}).items():
        if key in TRANSIENT:
            continue
        try:
            setattr(obj, key, decode(value))
        except Exception:
            continue


# ------------------------------------------------------------------ store
class SQLiteCheckpointStore:
    """Run state keyed by ``run_id``, in one stdlib-sqlite file."""

    def __init__(self, path: str = "agentfuse_runs.db"):
        self.path = path
        self._lock = threading.RLock()
        # check_same_thread=False because the monitor may be driven from an
        # executor thread; every access is guarded by the lock above.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        with self._lock:
            # WAL so a hard kill mid-write leaves a recoverable file rather than
            # a truncated one — the crash case is the entire point of this class.
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def save(self, run_id: str, state: dict, step: int = 0) -> None:
        # No default= coercion. Stringifying whatever json cannot handle is how
        # a list of numpy floats became a list of strings and broke the restored
        # detector; state_dict has already dropped anything unencodable, so
        # anything reaching here that still fails is a bug worth surfacing.
        blob = json.dumps(state)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO runs (run_id, updated_at, step, state) "
                "VALUES (?, ?, ?, ?)", (run_id, time.time(), step, blob))
            self._conn.commit()

    def load(self, run_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT state FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return None

    def runs(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT run_id, updated_at, step FROM runs "
                "ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [{"run_id": r[0], "updated_at": r[1], "step": r[2]} for r in rows]

    def delete(self, run_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
