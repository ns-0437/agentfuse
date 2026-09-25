"""Local, fail-closed journal for effects in the plain OpenAI adapter.

The journal records only tool names and operation state. Tool arguments and
results are deliberately excluded because they may contain credentials or PII.
It does not claim to roll back or reconcile an external side effect.
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import closing
from typing import Optional


class SQLiteOperationLedger:
    def __init__(self, path: str):
        self.path = path
        with closing(sqlite3.connect(path, timeout=30)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tool_operations (
                    operation_id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    owner TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (state IN ('pending', 'completed')),
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    completed_at TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS tool_operations_scope "
                         "ON tool_operations(scope)")
            conn.commit()

    def prior(self, scope: str) -> Optional[tuple[str, str]]:
        with closing(sqlite3.connect(self.path, timeout=30)) as conn:
            return conn.execute(
                "SELECT tool_name, state FROM tool_operations WHERE scope = ? "
                "ORDER BY CASE state WHEN 'pending' THEN 0 ELSE 1 END, "
                "created_at, rowid LIMIT 1", (scope,)).fetchone()

    def begin(self, scope: str, owner: str, tool_name: str
              ) -> tuple[Optional[str], Optional[tuple[str, str]]]:
        """Commit intent before the tool runs; reject another process's scope."""
        with closing(sqlite3.connect(self.path, timeout=30)) as conn:
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                prior = conn.execute(
                    "SELECT tool_name, state FROM tool_operations "
                    "WHERE scope = ? AND owner != ? "
                    "ORDER BY CASE state WHEN 'pending' THEN 0 ELSE 1 END "
                    "LIMIT 1", (scope, owner),
                ).fetchone()
                if prior is not None:
                    return None, prior
                operation_id = uuid.uuid4().hex
                conn.execute(
                    "INSERT INTO tool_operations "
                    "(operation_id, scope, owner, tool_name, state) "
                    "VALUES (?, ?, ?, ?, 'pending')",
                    (operation_id, scope, owner, tool_name),
                )
                return operation_id, None

    def complete(self, operation_id: str) -> None:
        with closing(sqlite3.connect(self.path, timeout=30)) as conn:
            with conn:
                conn.execute(
                    "UPDATE tool_operations SET state = 'completed', "
                    "completed_at = CURRENT_TIMESTAMP WHERE operation_id = ?",
                    (operation_id,),
                )
