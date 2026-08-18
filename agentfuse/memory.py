"""Recovery memory — what has been tried, and what actually worked.

Phase 1 measured recovery for the first time and exposed the weakness: steering
is *single-shot*. The breaker trips, produces an instruction, injects it, and
never checks whether it helped. If the agent is still stuck two steps later the
same trip fires again, and the same reasoning against the same snapshot produces
substantially the same advice — until ``max_recoveries`` is exhausted and the run
escalates. Three attempts, one idea, tried three times.

This module is the fix's memory half: a record of *(failure signature → steering
attempted → did it work)*, so the next trip on a similar failure can be told what
has already been ruled out.

Backends, following the project's usual shape — stdlib by default, optional
upgrades that never become required:

  * :class:`JSONMemory` — the default. A JSONL file, no dependencies, exact
    signature matching plus a lexical similarity fallback. Fine for a single
    agent over a single long run, which is the common case.
  * :class:`QdrantMemory` — optional (``pip install agentfuse[memory]``). Real
    vector search over failure signatures, so "I have seen a failure *like* this
    before" works across differently-worded but semantically identical failures.
    Uses Qdrant's embedded local mode, so there is still no server to run.

Both satisfy :class:`RecoveryMemory`, and the engine neither knows nor cares
which is in use.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
import threading
from pathlib import Path
from typing import Optional, Protocol

from .events import stable_hash


@dataclass
class RecoveryRecord:
    """One steering attempt and its verified outcome."""

    signature: str                 # fingerprint of the failure being recovered from
    detector: str                  # which detector tripped
    goal: str                      # the original objective
    strategy: str                  # which ladder rung produced this instruction
    instruction: str               # what the agent was actually told
    tool: Optional[str] = None     # the failing tool, when there is one
    worked: Optional[bool] = None  # None until the outcome is verified
    step: int = 0
    ts: float = field(default_factory=time.time)

    @property
    def record_id(self) -> str:
        return stable_hash({"sig": self.signature, "strategy": self.strategy,
                            "instruction": self.instruction, "ts": self.ts})

    def to_dict(self) -> dict:
        d = asdict(self)
        d["record_id"] = self.record_id
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RecoveryRecord":
        d = {k: v for k, v in d.items() if k != "record_id"}
        return cls(**d)


def failure_signature(detector: str, tool: Optional[str], evidence: dict) -> str:
    """A stable fingerprint for 'this kind of failure, on this thing'.

    Deliberately coarse: **detector plus tool, nothing else.** Two loops on the
    same tool must collide even if the step numbers, repeat counts, or internal
    detector signals differ, because the *lesson* transfers — the point of the
    memory is that a steer which failed here will probably fail again on the same
    shape of problem.

    Incidental detector metadata is excluded on purpose. Including it split one
    logical failure across several buckets: the loop detector reports ``signal``
    as either ``call+result`` or ``call-only fallback`` for what is plainly the
    same stuck tool, and keying on that made the ladder restart from the bottom
    each time the detector happened to fire down its other path — retrying
    corrections already known not to work.
    """
    return stable_hash({"detector": detector, "tool": tool})


class RecoveryMemory(Protocol):
    """What the recovery engine needs from a memory, and nothing more.

    ``failed_strategies`` is load-bearing: it is how the engine knows which rungs
    of the steering ladder have already been ruled out. A backend missing it does
    not fail loudly — the engine swallows memory faults so a broken memory cannot
    take down the run it supervises — it just silently reports that nothing has
    been tried, so the ladder never climbs and the same correction is reissued
    forever. Every backend must implement all of these.
    """

    def remember(self, record: RecoveryRecord) -> str: ...
    def recall(self, signature: str, limit: int = 5) -> list[RecoveryRecord]: ...
    def mark_outcome(self, record_id: str, worked: bool) -> None: ...
    def failed_instructions(self, signature: str) -> list[str]: ...
    def failed_strategies(self, signature: str) -> set[str]: ...


# --------------------------------------------------------------------------
# Default backend: no dependencies, no server, no configuration
# --------------------------------------------------------------------------
class JSONMemory:
    """Append-only JSONL memory with in-process indexing.

    Persistence is optional: with no path it stays in memory, which is the right
    default for a single supervised run and keeps tests hermetic.
    """

    def __init__(self, path: Optional[str] = None, max_records: int = 5000):
        # One memory is deliberately shareable across runs — that is the point
        # of it — so it cannot rely on the monitor's lock. _flush() rewrites the
        # WHOLE file, so two concurrent remember() calls can truncate each
        # other's output, and every read here iterates _records while another
        # thread may be appending.
        self._lock = threading.RLock()
        self.path = Path(path) if path else None
        self.max_records = max_records
        self._records: list[RecoveryRecord] = []
        self._by_id: dict[str, RecoveryRecord] = {}
        if self.path and self.path.exists():
            self._load()

    # -- persistence ----------------------------------------------------
    def _load(self) -> None:
        assert self.path is not None
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = RecoveryRecord.from_dict(json.loads(line))
            except (json.JSONDecodeError, TypeError):
                continue
            self._records.append(rec)
            self._by_id[rec.record_id] = rec

    def _flush(self) -> None:
        """Rewrite the whole file. Needed only when existing records changed."""
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            for rec in self._records:
                fh.write(json.dumps(rec.to_dict(), default=str) + "\n")

    def _append(self, record: RecoveryRecord) -> None:
        """Append one record. O(1) per write instead of O(n).

        This class is documented as "append-only JSONL", but every remember()
        used to rewrite the entire file, so a store holding max_records=5000
        performed up to 5000 serialisations per single write -- O(n^2) across a
        run. Measured, same machine, 1500 remember() calls: **42.1s before
        (28.1 ms/write, still climbing), 1.59s after** (1.06 ms/write, flat).
        The remaining cost is one open/append/close per record, which does not
        grow with history.

        That cost lands on the supervision hot path of a system whose whole
        premise is runs lasting hours to days, and it grows exactly as the run
        gets longer, which is the worst possible shape for it.
        """
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict(), default=str) + "\n")

    # -- interface ------------------------------------------------------
    def remember(self, record: RecoveryRecord) -> str:
        with self._lock:
            self._records.append(record)
            self._by_id[record.record_id] = record
            if len(self._records) > self.max_records:
                # An eviction removes a line that is already on disk, so the file
                # genuinely has to be rebuilt. That happens once every
                # max_records writes, not on every one.
                dropped = self._records.pop(0)
                self._by_id.pop(dropped.record_id, None)
                self._flush()
            else:
                self._append(record)
            return record.record_id

    def recall(self, signature: str, limit: int = 5) -> list[RecoveryRecord]:
        """Most recent attempts against this failure shape, newest first."""
        with self._lock:
            hits = [r for r in self._records if r.signature == signature]
        return sorted(hits, key=lambda r: r.ts, reverse=True)[:limit]

    def mark_outcome(self, record_id: str, worked: bool) -> None:
        with self._lock:
            rec = self._by_id.get(record_id)
            if rec is not None:
                rec.worked = worked
                self._flush()

    def failed_instructions(self, signature: str, limit: int = 5) -> list[str]:
        """Steering that has already been tried here and demonstrably failed."""
        with self._lock:
            failed = [r for r in self._records
                      if r.signature == signature and r.worked is False]
        failed.sort(key=lambda r: r.ts, reverse=True)
        return [r.instruction for r in failed[:limit]]

    def failed_strategies(self, signature: str) -> set[str]:
        """Every rung demonstrably ruled out for this failure shape.

        Scans the whole history rather than a recency window. Using a window
        here was a real bug: once enough records accumulated, the failed rungs
        aged out and the ladder silently restarted from the bottom, retrying
        corrections already known not to work.
        """
        with self._lock:
            return {r.strategy for r in self._records
                    if r.signature == signature and r.worked is False}

    def strategies_tried(self, signature: str) -> set[str]:
        with self._lock:
            return {r.strategy for r in self._records if r.signature == signature}

    def __len__(self) -> int:
        return len(self._records)

    @property
    def stats(self) -> dict:
        with self._lock:
            verified = [r for r in self._records if r.worked is not None]
        worked = [r for r in verified if r.worked]
        return {
            "records": len(self._records),
            "verified": len(verified),
            "worked": len(worked),
            "success_rate": (len(worked) / len(verified)) if verified else 0.0,
        }


# --------------------------------------------------------------------------
# Optional backend: semantic recall across differently-worded failures
# --------------------------------------------------------------------------
class QdrantMemory:
    """Vector-backed memory for recall across *similar* rather than identical failures.

    ``JSONMemory`` matches signatures exactly, which is enough when the same tool
    loops the same way twice. It cannot tell that "search_files found nothing" and
    "grep_files returned no matches" are the same lesson. That needs embeddings.

    Uses Qdrant's embedded mode (``path=`` or ``:memory:``), so this is still a
    library and not a service — no container, no daemon. Requires an embedder,
    which in practice means an API key, so the JSON backend stays the default.
    """

    def __init__(self, embedder, path: Optional[str] = None,
                 collection: str = "agentfuse_recovery", dim: Optional[int] = None):
        try:
            from qdrant_client import QdrantClient  # type: ignore
            from qdrant_client.models import Distance, VectorParams  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "QdrantMemory needs the optional extra: pip install agentfuse[memory]"
            ) from exc

        self._embedder = embedder
        self._collection = collection
        self._client = QdrantClient(path=path) if path else QdrantClient(":memory:")
        self._dim = dim or len(embedder("dimension probe"))
        existing = {c.name for c in self._client.get_collections().collections}
        if collection not in existing:
            self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
            )
        self._by_id: dict[str, RecoveryRecord] = {}
        self._rehydrate()

    @staticmethod
    def _point_id(record_id: str) -> int:
        """Stable point id for a record id, across processes.

        This used to be ``abs(hash(record_id)) % 2**63``. Python randomises
        string hashing per process, so the same record produced a DIFFERENT
        point id after every restart. Measured: three processes, three ids.

        In `:memory:` mode nothing notices. With ``path=`` it means a
        ``mark_outcome`` after a restart upserts a SECOND point instead of
        updating the first, leaving the stale ``worked=None`` copy in place —
        so the ladder can read a rung it has already disproved as untried and
        reissue a correction known not to work. That is precisely the failure
        Phase 2 was built to prevent, reintroduced underneath it.
        """
        return int.from_bytes(hashlib.blake2b(record_id.encode("utf-8"),
                                              digest_size=8).digest(),
                              "big") % (2 ** 63)

    def _rehydrate(self) -> None:
        """Reload the exact-match index from the collection.

        ``recall``, ``failed_strategies``, ``failed_instructions`` and
        ``strategies_tried`` all read ``_by_id``, which was in-process only. The
        vectors persisted; the lesson learned from them did not. Measured on a
        real on-disk store: before a restart ``failed_strategies`` returned
        ``{'re-anchor'}``, after it returned ``set()`` — while the record was
        still sitting in the collection with ``worked=False``.

        A memory that silently forgets which corrections already failed is worse
        than no memory, because the engine swallows memory faults by design and
        the ladder just quietly stops climbing.
        """
        try:
            records, _ = self._client.scroll(
                collection_name=self._collection, limit=10_000,
                with_payload=True, with_vectors=False)
        except Exception:                       # noqa: BLE001
            return                              # empty or unreadable: start clean
        for point in records or []:
            payload = getattr(point, "payload", None)
            if not payload:
                continue
            try:
                rec = RecoveryRecord.from_dict(dict(payload))
            except (TypeError, ValueError):
                continue
            self._by_id[rec.record_id] = rec

    def _text_for(self, record: RecoveryRecord) -> str:
        return f"{record.detector} failure on {record.tool or 'unknown tool'}: {record.goal}"

    def remember(self, record: RecoveryRecord) -> str:
        from qdrant_client.models import PointStruct  # type: ignore

        vec = self._embedder(self._text_for(record))
        self._by_id[record.record_id] = record
        self._client.upsert(
            collection_name=self._collection,
            points=[PointStruct(id=self._point_id(record.record_id),
                                vector=vec, payload=record.to_dict())],
        )
        return record.record_id

    def recall(self, signature: str, limit: int = 5) -> list[RecoveryRecord]:
        matches = [r for r in self._by_id.values() if r.signature == signature]
        return sorted(matches, key=lambda r: r.ts, reverse=True)[:limit]

    def recall_similar(self, detector: str, tool: Optional[str], goal: str,
                       limit: int = 5) -> list[RecoveryRecord]:
        """The reason this backend exists: semantically similar past failures.

        Exact signature matching cannot tell that "search_files found nothing"
        and "grep_files returned no matches" are the same lesson. This can.

        The client API is version-dependent: ``search()`` was replaced by
        ``query_points()``. This class shipped calling the removed method, which
        went unnoticed because it had never been run.
        """
        probe = f"{detector} failure on {tool or 'unknown tool'}: {goal}"
        vector = self._embedder(probe)

        if hasattr(self._client, "query_points"):
            response = self._client.query_points(
                collection_name=self._collection, query=vector, limit=limit)
            hits = getattr(response, "points", response)
        else:  # older clients
            hits = self._client.search(collection_name=self._collection,
                                       query_vector=vector, limit=limit)

        out = []
        for h in hits:
            payload = getattr(h, "payload", None)
            if not payload:
                continue
            try:
                out.append(RecoveryRecord.from_dict(dict(payload)))
            except (TypeError, ValueError):
                continue
        return out

    def mark_outcome(self, record_id: str, worked: bool) -> None:
        rec = self._by_id.get(record_id)
        if rec is not None:
            rec.worked = worked
            self.remember(rec)

    def failed_instructions(self, signature: str, limit: int = 5) -> list[str]:
        failed = [r for r in self._by_id.values()
                  if r.signature == signature and r.worked is False]
        failed.sort(key=lambda r: r.ts, reverse=True)
        return [r.instruction for r in failed[:limit]]

    def failed_strategies(self, signature: str) -> set[str]:
        """Rungs demonstrably ruled out for this failure shape.

        This backend shipped without this method. Nothing failed loudly — the
        engine swallows memory faults by design — the ladder simply never climbed
        and the same correction was reissued indefinitely. It went unnoticed
        because the class had never once been executed.
        """
        return {r.strategy for r in self._by_id.values()
                if r.signature == signature and r.worked is False}

    def strategies_tried(self, signature: str) -> set[str]:
        return {r.strategy for r in self._by_id.values() if r.signature == signature}
