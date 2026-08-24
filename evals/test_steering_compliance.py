"""Tests for the compliance measurement behind REPORT.md sections 3.5-3.6.

This is the actual evidence for this project's central "does steering work"
claim, so it gets the same scrutiny as a detector: constructed traces with a
known answer, not just a smoke test that it runs.
"""

from __future__ import annotations

import json
from pathlib import Path

from evals.steering_compliance import compliance_from_trace, tool_signature


def _write(tmp_path: Path, records: list[dict]) -> Path:
    trace = tmp_path / "trace.jsonl"
    trace.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return trace


def _call(tool: str, args: dict) -> dict:
    return {"kind": "event", "type": "tool_call", "tool_name": tool, "tool_args": args}


def _trip(reason: str = "loop") -> dict:
    return {"kind": "trip", "detector": "loop", "reason": reason}


def test_a_different_next_call_counts_as_complied(tmp_path):
    trace = _write(tmp_path, [
        _call("search_files", {"dir": "./config"}),
        _trip(),
        _call("read_secret", {"name": "prod/db/primary"}),
    ])
    assert compliance_from_trace(trace) == (1, 1)


def test_the_identical_next_call_counts_as_ignored(tmp_path):
    trace = _write(tmp_path, [
        _call("search_files", {"dir": "./config"}),
        _trip(),
        _call("search_files", {"dir": "./config"}),
    ])
    assert compliance_from_trace(trace) == (0, 1)


def test_same_tool_different_args_counts_as_complied():
    """The signature is (tool, args) together -- same tool with a genuinely
    different argument is not the loop being repeated."""
    a = tool_signature({"tool_name": "search_files", "tool_args": {"dir": "./config"}})
    b = tool_signature({"tool_name": "search_files", "tool_args": {"dir": "./logs"}})
    assert a != b


def test_a_trip_with_no_following_call_is_not_counted(tmp_path):
    """The run ended or escalated right after the trip -- no behaviour exists
    to judge yet, so this must not silently count as either verdict."""
    trace = _write(tmp_path, [
        _call("search_files", {"dir": "./config"}),
        _trip(),
    ])
    assert compliance_from_trace(trace) == (0, 0)


def test_multiple_trips_in_one_trace_are_each_judged_independently(tmp_path):
    trace = _write(tmp_path, [
        _call("search_files", {"dir": "./config"}),
        _trip(),
        _call("search_files", {"dir": "./config"}),   # ignored
        _trip(),
        _call("read_secret", {"name": "prod/db/primary"}),  # complied
    ])
    assert compliance_from_trace(trace) == (1, 2)


def test_argument_key_order_does_not_matter():
    """json.dumps(..., sort_keys=True) is load-bearing: two dicts written in a
    different key order must still compare equal, or an adapter that happens
    to emit args in a different order would look like a different call."""
    a = tool_signature({"tool_name": "search_files", "tool_args": {"dir": "x", "n": 1}})
    b = tool_signature({"tool_name": "search_files", "tool_args": {"n": 1, "dir": "x"}})
    assert a == b


def test_malformed_lines_are_skipped_not_fatal(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps(_call("search_files", {"dir": "./config"})) + "\n"
        + "not json at all\n"
        + json.dumps(_trip()) + "\n"
        + json.dumps(_call("read_secret", {"name": "prod/db/primary"})) + "\n",
        encoding="utf-8")
    assert compliance_from_trace(trace) == (1, 1)
