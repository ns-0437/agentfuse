"""Offline write-recovery walkthrough using a real SQLite invoice store.

Run: python examples/sqlite_write_recovery.py

The scripted model is free; the OpenAI-compatible adapter, operation journal,
and invoice database are real. Two failure timings show why a timeout cannot
prove whether a write happened. The example uses separate logical job scopes
and verifies the authoritative invoice table before any retry.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentfuse.adapters.openai_sdk import guarded_tool_loop  # noqa: E402
from agentfuse.operation_ledger import SQLiteOperationLedger  # noqa: E402


class ScriptedCompletions:
    def __init__(self, invoice_id: int):
        self.invoice_id = invoice_id
        self.calls = 0

    def create(self, *, messages, **kwargs):
        self.calls += 1
        if any(message.get("role") == "tool" for message in messages):
            message = SimpleNamespace(content="Invoice created.", tool_calls=None)
        else:
            call = SimpleNamespace(
                id=f"call-{self.calls}",
                function=SimpleNamespace(
                    name="create_invoice",
                    arguments=json.dumps({"invoice_id": self.invoice_id}),
                ),
            )
            message = SimpleNamespace(content="Creating invoice.", tool_calls=[call])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


def run_demo(directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    invoices = directory / "invoices.sqlite3"
    journal = directory / "agentfuse-operations.sqlite3"
    with closing(sqlite3.connect(invoices)) as conn:
        with conn:
            conn.execute("CREATE TABLE IF NOT EXISTS invoices "
                         "(invoice_id INTEGER PRIMARY KEY)")

    def exists(invoice_id: int) -> bool:
        with closing(sqlite3.connect(invoices)) as conn:
            return conn.execute("SELECT 1 FROM invoices WHERE invoice_id = ?",
                                (invoice_id,)).fetchone() is not None

    def run(invoice_id: int, failure: str | None) -> tuple[dict, ScriptedCompletions]:
        completions = ScriptedCompletions(invoice_id)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        def route(name: str, args: dict) -> str:
            assert name == "create_invoice"
            assert args == {"invoice_id": invoice_id}
            if failure == "before_commit":
                raise TimeoutError("request never reached the invoice store")
            with closing(sqlite3.connect(invoices)) as conn:
                with conn:
                    conn.execute("INSERT INTO invoices (invoice_id) VALUES (?)",
                                 (invoice_id,))
            if failure == "after_commit":
                raise TimeoutError("response was lost after commit")
            return f"invoice {invoice_id} created"

        result = guarded_tool_loop(
            client, model="offline-scripted", system_prompt="Create invoices.",
            user_input=f"Create invoice {invoice_id}.", tools=[], tool_router=route,
            tool_effects={"create_invoice": "write"}, max_turns=3, echo=False,
            operation_ledger_path=str(journal),
            operation_scope=f"invoice-{invoice_id}",
        )
        return result, completions

    # Case A: the write committed, but its response was lost.
    after_commit, _ = run(42, "after_commit")
    blocked_retry, retry_model = run(42, None)
    assert after_commit["status"] == blocked_retry["status"] == "recovery_blocked"
    assert retry_model.calls == 0  # stopped before model or tool execution
    assert exists(42)

    # Case B: the request failed before the write. The application checks the
    # authoritative store, then explicitly clears only this pending intent.
    before_commit, _ = run(43, "before_commit")
    assert before_commit["status"] == "recovery_blocked"
    assert not exists(43)
    ledger = SQLiteOperationLedger(str(journal))
    pending = ledger.pending("invoice-43")
    assert len(pending) == 1
    ledger.confirm_not_applied(pending[0].operation_id)
    retried, _ = run(43, None)
    assert retried["status"] == "complete" and exists(43)

    with closing(sqlite3.connect(invoices)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM invoices").fetchone()[0]
    assert count == 2
    return {
        "ok": True,
        "committed_then_timeout": after_commit["status"],
        "blind_retry": blocked_retry["status"],
        "failed_before_commit": before_commit["status"],
        "verified_absent_retry": retried["status"],
        "invoice_rows": count,
    }


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="agentfuse-write-recovery-") as temp_dir:
        print(json.dumps(run_demo(Path(temp_dir)), sort_keys=True))
