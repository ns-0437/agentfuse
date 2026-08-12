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

import json
import time
from dataclasses import dataclass, field, asdict
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
    """What the recovery engine needs from a memory, and nothing more."""

    def remember(self, record: RecoveryRecord) -> str: ...
    def recall(self, signature: str, limit: int = 5) -> list[RecoveryRecord]: ...
    def mark_outcome(self, record_id: str, worked: bool) -> None: ...
    def failed_instructions(self, signature: str) -> list[str]: ...


# --------------------------------------------------------------------------
# Default backend: no dependencies, no server, no configuration
# --------------------------------------------------------------------------
class JSONMemory:
    """Append-only JSONL memory with in-process indexing.

    Persistence is optional: with no path it stays in memory, which is the right
    default for a single supervised run and keeps tests hermetic.
    """

    def __init__(self, path: Optional[str] = None, max_records: int = 5000):
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
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            for rec in self._records:
                fh.write(json.dumps(rec.to_dict(), default=str) + "\n")

    # -- interface ------------------------------------------------------
    def remember(self, record: RecoveryRecord) -> str:
        self._records.append(record)
        self._by_id[record.record_id] = record
        if len(self._records) > self.max_records:
            dropped = self._records.pop(0)
            self._by_id.pop(dropped.record_id, None)
        self._flush()
        return record.record_id

    def recall(self, signature: str, limit: int = 5) -> list[RecoveryRecord]:
        """Most recent attempts against this failure shape, newest first."""
        hits = [r for r in self._records if r.signature == signature]
        return sorted(hits, key=lambda r: r.ts, reverse=True)[:limit]

    def mark_outcome(self, record_id: str, worked: bool) -> None:
        rec = self._by_id.get(record_id)
        if rec is not None:
            rec.worked = worked
            self._flush()

    def failed_instructions(self, signature: str, limit: int = 5) -> list[str]:
        """Steering that has already been tried here and demonstrably failed."""
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
        return {r.strategy for r in self._records
                if r.signature == signature and r.worked is False}

    def strategies_tried(self, signature: str) -> set[str]:
        return {r.strategy for r in self._records if r.signature == signature}

    def __len__(self) -> int:
        return len(self._records)

    @property
    def stats(self) -> dict:
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

    def _text_for(self, record: RecoveryRecord) -> str:
        return f"{record.detector} failure on {record.tool or 'unknown tool'}: {record.goal}"

    def remember(self, record: RecoveryRecord) -> str:
        from qdrant_client.models import PointStruct  # type: ignore

        vec = self._embedder(self._text_for(record))
        self._by_id[record.record_id] = record
        self._client.upsert(
            collection_name=self._collection,
            points=[PointStruct(id=abs(hash(record.record_id)) % (2 ** 63),
                                vector=vec, payload=record.to_dict())],
        )
        return record.record_id

    def recall(self, signature: str, limit: int = 5) -> list[RecoveryRecord]:
        matches = [r for r in self._by_id.values() if r.signature == signature]
        return sorted(matches, key=lambda r: r.ts, reverse=True)[:limit]

    def recall_similar(self, detector: str, tool: Optional[str], goal: str,
                       limit: int = 5) -> list[RecoveryRecord]:
        """The reason this backend exists: semantically similar past failures."""
        probe = f"{detector} failure on {tool or 'unknown tool'}: {goal}"
        hits = self._client.search(collection_name=self._collection,
                                   query_vector=self._embedder(probe), limit=limit)
        out = []
        for h in hits:
            try:
                out.append(RecoveryRecord.from_dict(dict(h.payload)))
            except (TypeError, ValueError):
                continue
        return out

    def mark_outcome(self, record_id: str, worked: bool) -> None:
        rec = self._by_id.get(record_id)
        if rec is not None:
            rec.worked = worked
            self.remember(rec)

    def failed_instructions(self, signature: str) -> list[str]:
        return [r.instruction for r in self.recall(signature, limit=20)
                if r.worked is False]

    def strategies_tried(self, signature: str) -> set[str]:
        return {r.strategy for r in self.recall(signature, limit=20)}
