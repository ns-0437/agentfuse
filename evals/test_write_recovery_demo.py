"""The documented write-recovery walkthrough must exercise real SQLite state."""

import sqlite3

from agentfuse.operation_ledger import SQLiteOperationLedger
from examples.sqlite_write_recovery import run_demo


def test_write_recovery_walkthrough_distinguishes_both_timeout_timings(tmp_path):
    result = run_demo(tmp_path)
    assert result == {
        "ok": True,
        "committed_then_timeout": "recovery_blocked",
        "blind_retry": "recovery_blocked",
        "failed_before_commit": "recovery_blocked",
        "verified_absent_retry": "complete",
        "invoice_rows": 2,
    }
    with sqlite3.connect(tmp_path / "invoices.sqlite3") as conn:
        assert conn.execute("SELECT invoice_id FROM invoices ORDER BY invoice_id"
                            ).fetchall() == [(42,), (43,)]
    ledger = SQLiteOperationLedger(str(tmp_path / "agentfuse-operations.sqlite3"))
    assert len(ledger.pending("invoice-42")) == 1
    assert ledger.pending("invoice-43") == []
    assert ledger.prior("invoice-43") == ("create_invoice", "completed")
