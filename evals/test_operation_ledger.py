"""Durable tool-effect boundaries for the framework-free adapter."""

import sqlite3

import pytest

from agentfuse.adapters.openai_sdk import guarded_tool_loop
from agentfuse.monitor import Directive
from agentfuse.operation_ledger import SQLiteOperationLedger
from evals.test_adapters_untested import FakeOpenAI, _Msg, _Resp, _ToolCall


class QuietMonitor:
    def observe(self, event):
        return Directive()

    def finish(self, status):
        return {"status": status}


def client_with_tool(name="create_invoice"):
    return FakeOpenAI([
        _Resp(_Msg("working", [_ToolCall(name, {"id": 42}, 0)])),
        _Resp(_Msg("done")),
    ])


def run(client, router, path, scope="invoice-42", effects=None):
    return guarded_tool_loop(
        client, "fake", "Assist", "Create invoice 42", [], router,
        monitor=QuietMonitor(), max_turns=3, tool_effects=effects,
        operation_ledger_path=str(path), operation_scope=scope,
    )


def test_completed_write_blocks_a_restarted_run_before_model_or_tool(tmp_path):
    path = tmp_path / "operations.db"
    writes = []
    first = run(client_with_tool(),
                lambda *_: writes.append("committed") or "invoice created", path)
    assert first["status"] == "complete"
    assert SQLiteOperationLedger(str(path)).prior("invoice-42") == (
        "create_invoice", "completed")

    second_client = client_with_tool()
    second = run(second_client,
                 lambda *_: writes.append("duplicate") or "invoice created", path)
    assert second["status"] == "recovery_blocked"
    assert second["blocked_tool"] == "create_invoice"
    assert second_client.seen == []
    assert writes == ["committed"]


def test_ledger_does_not_persist_tool_results(tmp_path):
    path = tmp_path / "operations.db"
    result = run(client_with_tool(), lambda *_: "private-customer-token", path)
    assert result["status"] == "complete"
    assert b"private-customer-token" not in path.read_bytes()


def test_tool_exception_leaves_pending_intent_and_blocks_retry(tmp_path):
    path = tmp_path / "operations.db"
    writes = []

    def uncertain_write(*_):
        writes.append("possibly committed")
        raise TimeoutError("connection dropped after write")

    first = run(client_with_tool(), uncertain_write, path)
    assert first["status"] == "recovery_blocked"
    assert "outcome is unknown" in first["blocked_reason"]
    assert SQLiteOperationLedger(str(path)).prior("invoice-42") == (
        "create_invoice", "pending")

    second_client = client_with_tool()
    second = run(second_client, lambda *_: writes.append("duplicate"), path)
    assert second["status"] == "recovery_blocked"
    assert second_client.seen == []
    assert writes == ["possibly committed"]


def test_verified_not_applied_intent_can_be_retried(tmp_path):
    path = tmp_path / "operations.db"
    external_invoices = []

    def failed_before_write(*_):
        raise TimeoutError("service unavailable before request was sent")

    first = run(client_with_tool(), failed_before_write, path)
    assert first["status"] == "recovery_blocked"
    ledger = SQLiteOperationLedger(str(path))
    pending = ledger.pending("invoice-42")
    assert len(pending) == 1
    assert pending[0].tool_name == "create_invoice"
    # The application checks its authoritative store before making this claim.
    assert external_invoices == []
    ledger.confirm_not_applied(pending[0].operation_id)
    assert ledger.pending("invoice-42") == []

    second = run(client_with_tool(),
                 lambda *_: external_invoices.append(42) or "created", path)
    assert second["status"] == "complete"
    assert external_invoices == [42]
    assert ledger.prior("invoice-42") == ("create_invoice", "completed")


def test_completed_effect_cannot_be_cleared_as_not_applied(tmp_path):
    path = tmp_path / "operations.db"
    run(client_with_tool(), lambda *_: "created", path)
    ledger = SQLiteOperationLedger(str(path))
    with sqlite3.connect(path) as conn:
        operation_id = conn.execute(
            "SELECT operation_id FROM tool_operations").fetchone()[0]
    with pytest.raises(ValueError, match="not pending"):
        ledger.confirm_not_applied(operation_id)
    assert ledger.prior("invoice-42") == ("create_invoice", "completed")


def test_existing_journal_is_migrated_without_losing_pending_intent(tmp_path):
    path = tmp_path / "operations.db"
    with sqlite3.connect(path) as conn:
        conn.execute("""
            CREATE TABLE tool_operations (
                operation_id TEXT PRIMARY KEY, scope TEXT NOT NULL,
                owner TEXT NOT NULL, tool_name TEXT NOT NULL,
                state TEXT NOT NULL CHECK (state IN ('pending', 'completed')),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                completed_at TEXT)
        """)
        conn.execute("INSERT INTO tool_operations "
                     "(operation_id, scope, owner, tool_name, state) "
                     "VALUES ('old-op', 'invoice-42', 'old-owner', "
                     "'create_invoice', 'pending')")
    ledger = SQLiteOperationLedger(str(path))
    assert ledger.pending("invoice-42")[0].operation_id == "old-op"
    ledger.confirm_not_applied("old-op")
    assert ledger.prior("invoice-42") is None


def test_read_only_tools_do_not_poison_operation_scope(tmp_path):
    path = tmp_path / "operations.db"
    for _ in range(2):
        result = run(client_with_tool("get_invoice"), lambda *_: "invoice 42",
                     path, effects={"get_invoice": "read"})
        assert result["status"] == "complete"
    assert SQLiteOperationLedger(str(path)).prior("invoice-42") is None


def test_other_process_cannot_begin_effect_in_claimed_scope(tmp_path):
    ledger = SQLiteOperationLedger(str(tmp_path / "operations.db"))
    operation_id, conflict = ledger.begin("invoice-42", "owner-a", "create_invoice")
    assert operation_id and conflict is None
    other_id, conflict = ledger.begin("invoice-42", "owner-b", "send_email")
    assert other_id is None
    assert conflict == ("create_invoice", "pending")
    ledger.complete(operation_id)
    assert ledger.prior("invoice-42") == ("create_invoice", "completed")


@pytest.mark.parametrize("path,scope", [(None, "invoice-42"), ("journal.db", None)])
def test_ledger_requires_both_path_and_scope(path, scope):
    with pytest.raises(ValueError, match="must be set together"):
        guarded_tool_loop(client_with_tool(), "fake", "Assist", "Task", [],
                          lambda *_: None, monitor=QuietMonitor(),
                          operation_ledger_path=path, operation_scope=scope)


@pytest.mark.parametrize("path,scope,field", [
    (" ", "invoice-42", "operation_ledger_path"),
    ("journal.db", " ", "operation_scope"),
])
def test_ledger_rejects_blank_identity(path, scope, field):
    with pytest.raises(ValueError, match=field):
        guarded_tool_loop(client_with_tool(), "fake", "Assist", "Task", [],
                          lambda *_: None, monitor=QuietMonitor(),
                          operation_ledger_path=path, operation_scope=scope)
